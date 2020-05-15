#include "../NDS.h"
#include "../GPU.h"
#include "../version.h"
#include "../Config.h"
#include "../OpenGLSupport.h"
#include "../SPU.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <dirent.h>
#include <unistd.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include <unordered_map>
#include <algorithm>
#include <vector>

#include <deko3d.hpp>

#include "vec.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_deko3d.h"

#include "dr_wav.h"

#include "compat_switch.h"

#include "profiler.h"

#ifdef GDB_ENABLED
#include <gdbstub.h>
#endif

extern std::unordered_map<u32, u32> arm9BlockFrequency;
extern std::unordered_map<u32, u32> arm7BlockFrequency;

namespace Config
{
int CursorMode;
int CursorClickMode;

int ScreenRotation;
int ScreenGap;
int ScreenLayout;
int ScreenSizing;

int IntegerScaling;

int Filtering;

char LastROMFolder[512];

int SwitchOverclock;

int DirectBoot;

int GlobalRotation;

ConfigEntry PlatformConfigFile[] =
{
    {"ScreenRotation", 0, &ScreenRotation, 0, NULL, 0},
    {"ScreenGap",      0, &ScreenGap,      0, NULL, 0},
    {"ScreenLayout",   0, &ScreenLayout,   0, NULL, 0},
    {"ScreenSizing",   0, &ScreenSizing,   0, NULL, 0},
    {"Filtering",      0, &Filtering,      1, NULL, 0},
    {"IntegerScaling", 0, &IntegerScaling, 0, NULL, 0},
    {"GlobalRotation", 0, &GlobalRotation, 0, NULL, 0},
    {"CursorMode",     0, &CursorMode,     0, NULL, 0},

    {"LastROMFolder", 1, LastROMFolder, 0, (char*)"/", 511},

    {"SwitchOverclock", 0, &SwitchOverclock, 0, NULL, 0},

    {"DirectBoot",   0, &DirectBoot,     1, NULL, 0},

    {"", -1, NULL, 0, NULL, 0}
};
}

struct StupidStackBuffer
{
    dk::MemBlock block;
    u32 offset;

    StupidStackBuffer(){}

    StupidStackBuffer(dk::Device gDevice, u32 flags, u32 size)
    {
        block = dk::MemBlockMaker{gDevice, size}.setFlags(flags).create();
        
        offset = 0;
    }

    void Destroy()
    {
        block.destroy();
    }

    u32 Allocate(u32 size, u32 align = 0)
    {
        offset = (offset + align - 1) & ~(align - 1);
        size = (size + align - 1) & ~(align - 1);
        assert(offset + size <= block.getSize());
        u32 result = offset;
        offset += size;
        return result;
    }

    void Reset()
    {
        offset = 0;
    }
};

dk::Device gDevice;
dk::Queue gQueue;

dk::CmdBuf gCmdbuf;
dk::Fence gCmdFence[2];
ImGui_GfxDataBlock gCmdbufData[2];

dk::Image gFramebuffers[2];

dk::Swapchain gSwapchain;

StupidStackBuffer textureBuffer;
StupidStackBuffer codeBuffer;
StupidStackBuffer dataBuffer;
StupidStackBuffer tempBuffer;

dk::Image rotatedFb, screenTexture;

ImGui_GfxDataBlock fullscreenQuadMem;
ImGui_GfxDataBlock screenVerticesMem;

static const ImDrawVert fullscreenQuadData[] =
{
    {{-1.f, -1.f}, {0.f, 1.f}, 0xFFFFFFFF},
    {{-1.f, 1.f}, {0.f, 0.f}, 0xFFFFFFFF},
    {{1.f, 1.f}, {1.f, 0.f}, 0xFFFFFFFF},
    {{1.f, -1.f}, {1.f, 1.f}, 0xFFFFFFFF},
};

ImGui_GfxDataBlock allocTexture(u32 size, u32 align)
{
    size = (size + align - 1) & ~(align - 1);
    ImGui_GfxDataBlock block;
    block.offset = textureBuffer.Allocate(size, align);
    block.mem = textureBuffer.block;
    block.size = size;
    return block;
}
ImGui_GfxDataBlock allocShader(u32 size, u32 align)
{
    size = (size + align - 1) & ~(align - 1);
    ImGui_GfxDataBlock block;
    block.offset = codeBuffer.Allocate(size, align);
    block.mem = codeBuffer.block;
    block.size = size;
    return block;
}
ImGui_GfxDataBlock allocData(u32 size, u32 align)
{
    size = (size + align - 1) & ~(align - 1);
    ImGui_GfxDataBlock block;
    block.offset = dataBuffer.Allocate(size, align);
    block.mem = dataBuffer.block;
    block.size = size;
    return block;
}
ImGui_GfxDataBlock allocTmp(u32 size, u32 align)
{
    size = (size + align - 1) & ~(align - 1);
    ImGui_GfxDataBlock block;
    block.offset = tempBuffer.Allocate(size, align);
    block.mem = tempBuffer.block;
    block.size = size;
    return block;
}
void resetTmp()
{
    tempBuffer.Reset();
}

void dkError(void* userData, const char* context, DkResult result, const char* message)
{
    printf("errorrrrrr %d %s\n", result, message);
}

void graphicsInitialize()
{
    gDevice = dk::DeviceMaker{}.setCbDebug(dkError).create();

    gQueue = dk::QueueMaker{gDevice}.setFlags(DkQueueFlags_Graphics).create();

    textureBuffer = StupidStackBuffer(gDevice, DkMemBlockFlags_Image | DkMemBlockFlags_GpuCached, 16*1024*1024);
    codeBuffer = StupidStackBuffer(gDevice, DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached | DkMemBlockFlags_Code, 128*1024);
    dataBuffer = StupidStackBuffer(gDevice, DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached, 1024*1024);
    tempBuffer = StupidStackBuffer(gDevice, DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached, 1024*1024);

    gCmdbuf = dk::CmdBufMaker{gDevice}.create();
    gCmdbufData[0] = allocData(0x10000, 0x10000);
    gCmdbufData[1] = allocData(0x10000, 0x10000);

    dk::ImageLayout fbLayout;
    dk::ImageLayoutMaker{gDevice}
        .setFlags(DkImageFlags_UsageRender | DkImageFlags_UsagePresent | DkImageFlags_HwCompression)
        .setFormat(DkImageFormat_RGBA8_Unorm)
        .setDimensions(1280, 720)
        .initialize(fbLayout);

    u64 fbSize = fbLayout.getSize();
    u32 fbAlign = fbLayout.getAlignment();
    std::array<DkImage const*, 2> fbArray;
    for (int i = 0; i < 2; i++)
    {
        ImGui_GfxDataBlock mem = allocTexture(fbSize, fbAlign);

        gFramebuffers[i].initialize(fbLayout, mem.mem, mem.offset);

        fbArray[i] = &gFramebuffers[i];
    }

    gSwapchain = dk::SwapchainMaker{gDevice, nwindowGetDefault(), fbArray}.create();

    dk::ImageLayout rotatedFbLayout;
    dk::ImageLayoutMaker{gDevice}
        .setFlags(DkImageFlags_UsageRender | DkImageFlags_HwCompression)
        .setFormat(DkImageFormat_RGBA8_Unorm)
        .setDimensions(1280, 1280)
        .initialize(rotatedFbLayout);

    ImGui_GfxDataBlock rotatedFbMem = allocTexture(rotatedFbLayout.getSize(), rotatedFbLayout.getAlignment());
    rotatedFb.initialize(rotatedFbLayout, rotatedFbMem.mem, rotatedFbMem.offset);

    fullscreenQuadMem = allocData(sizeof(fullscreenQuadData), alignof(fullscreenQuadData[0]));
    memcpy(fullscreenQuadMem.GetCpuAddr(), fullscreenQuadData, sizeof(fullscreenQuadData));

    screenVerticesMem = allocData(sizeof(ImDrawVert) * 8, alignof(ImDrawVert));

    dk::ImageLayout screenTextureLayout;
    dk::ImageLayoutMaker{gDevice}
        .setFlags(0)
        .setFormat(DkImageFormat_BGRX8_Unorm)
        .setDimensions(256, 192*2)
        .initialize(screenTextureLayout);

    ImGui_GfxDataBlock screenTextureMem = allocTexture(screenTextureLayout.getSize(), screenTextureLayout.getAlignment());
    screenTexture.initialize(screenTextureLayout, screenTextureMem.mem, screenTextureMem.offset);
}

void graphicsUpdate(int guiState, int screenWidth, int screenHeight)
{
    int slot = gQueue.acquireImage(gSwapchain);

    gCmdbuf.clear();
    gCmdFence[slot].wait();
    gCmdbuf.addMemory(dataBuffer.block, gCmdbufData[slot].offset, gCmdbufData[slot].size);
    
    if (guiState == 1)
    {
        ImGui_GfxDataBlock stageBuffer = allocTmp(256*192*2, 16);
        memcpy(stageBuffer.GetCpuAddr(), GPU::Framebuffer[GPU::FrontBuffer][0], 256*192*4);
        memcpy(((uint32_t*)stageBuffer.GetCpuAddr()) + 256*192, GPU::Framebuffer[GPU::FrontBuffer][1], 256*192*4);

        dk::ImageView screenTexView {screenTexture};
        gCmdbuf.copyBufferToImage({stageBuffer.GetGpuAddr()}, screenTexView, {0, 0, 0, 256, 192*2, 1});

        gQueue.submitCommands(gCmdbuf.finishList());
        gQueue.waitIdle();
    }

    dk::ImageView colorTarget {gFramebuffers[slot]};
    gCmdbuf.bindRenderTargets(&colorTarget);
    gCmdbuf.setViewports(0, {{0, 0, 1280, 720}});
    gCmdbuf.setScissors(0, {{0, 0, 1280, 720}});

    if (guiState == 1)
        gCmdbuf.clearColor(0, DkColorMask_RGBA, 0.0f, 0.0f, 0.0f, 0.f);
    else
        gCmdbuf.clearColor(0, DkColorMask_RGBA, 117.f/255.f, 38.f/255.f, 50.f/255.f, 0.f);

    ImGui_ImplDeko3D_SetupRenderState(gCmdbuf);

    ImGui_GfxTransform transform = {0};
    xm4_orthographic(transform.ProjMtx, -1280.f/2, 1280.f/2, 720.f/2.f, -720.f/2, -1.f, 1.f);
    float rot[16];
    float trans[16];
    xm4_translatev(trans, -screenWidth/2, -screenHeight/2, 0.f);
    xm4_rotatef(rot, -M_PI_2 * Config::GlobalRotation, 0.f, 0.f, 1.f);
    xm4_mul(rot, trans, rot);
    xm4_mul(transform.ProjMtx, rot, transform.ProjMtx);
    transform.TexMtx[0] = 1.f;
    transform.TexMtx[5] = 1.f;

    ImGui_ImplDeko3D_SetTransform(gCmdbuf, &transform);

    if (guiState > 0)
    {
        gCmdbuf.bindTextures(DkStage_Fragment, 0, dkMakeTextureHandle(2, Config::Filtering ? 0 : 1));
        gCmdbuf.bindVtxBuffer(0, screenVerticesMem.GetGpuAddr(), screenVerticesMem.size);
        gCmdbuf.draw(DkPrimitive_Quads, 8, 1, 0, 0);

        resetTmp();
    }

    ImGui::Render();
    ImGui_ImplDeko3D_RenderDrawData(ImGui::GetDrawData(), gCmdbuf, &transform, Config::GlobalRotation);

    gCmdbuf.signalFence(gCmdFence[slot]);

    gQueue.submitCommands(gCmdbuf.finishList());

    gQueue.presentImage(gSwapchain, slot);
}

void graphicsExit()
{
    gQueue.waitIdle();
    gQueue.destroy();

    gCmdbuf.clear();
    gCmdbuf.destroy();

    gSwapchain.destroy();

    textureBuffer.Destroy();
    codeBuffer.Destroy();
    dataBuffer.Destroy();

    gDevice.destroy();
}

void applyOverclock(bool usePCV, ClkrstSession* session, int setting)
{
    const int clockSpeeds[] = { 1020000000, 1224000000, 1581000000, 1785000000 };
    if (usePCV)
        pcvSetClockRate(PcvModule_CpuBus, clockSpeeds[setting]);
    else
        clkrstSetClockRate(session, clockSpeeds[setting]);
}

float botX, botY, botWidth, botHeight;
int AutoScreenSizing = 0;

void updateScreenLayout(int screenWidth, int screenHeight)
{
    const ImDrawVert verticesSingleScreen[] =
    {
        {{-256.f/2, -192.f/2}, {0.f, 0.f}, 0xFFFFFFFF},
        {{-256.f/2, 192.f/2}, {0.f, 0.5f}, 0xFFFFFFFF},
        {{256.f/2, 192.f/2}, {1.f, 0.5f}, 0xFFFFFFFF},
        {{256.f/2, -192.f/2}, {1.f, 0.f}, 0xFFFFFFFF},
    };

    ImDrawVert vertices[8];
    memcpy(&vertices[0], verticesSingleScreen, sizeof(ImDrawVert)*4);
    memcpy(&vertices[4], verticesSingleScreen, sizeof(ImDrawVert)*4);

    int layout = Config::ScreenLayout == 0
        ? ((Config::ScreenRotation % 2 == 0) ? 0 : 1)
        : Config::ScreenLayout - 1;
    int rotation = Config::ScreenRotation;

    int sizing = Config::ScreenSizing == 3 ? AutoScreenSizing : Config::ScreenSizing;

    {
        float rotmat[4];
        xm2_rotate(rotmat, M_PI_2 * rotation);
        
        for (int i = 0; i < 8; i++)
            xm2_transform(&vertices[i].pos[0], rotmat, &vertices[i].pos[0]);
    }

    // move screens apart
    {
        const float screenGaps[] = {0.f, 1.f, 8.f, 64.f, 90.f, 128.f};
        int idx = layout == 0 ? 1 : 0;
        float offset =
            (((layout == 0 && (rotation % 2 == 0)) || (layout == 1 && (rotation % 2 == 1)) 
                ? 192.f : 256.f)
            + screenGaps[Config::ScreenGap]) / 2.f;
        if (rotation >= 2)
            offset *= -1.f;
        for (int i = 0; i < 4; i++)
            vertices[i].pos[idx] -= offset;
        for (int i = 0; i < 4; i++)
        {
            vertices[i + 4].pos[idx] += offset;
            vertices[i + 4].uv[1] += 0.5f;
        }
    }

    // scale
    {
        if (sizing == 0)
        {
            float minX = 100000.f, maxX = -100000.f;
            float minY = 100000.f, maxY = -100000.f;

            for (int i = 0; i < 8; i++)
            {
                minX = std::min(minX, vertices[i].pos[0]);
                minY = std::min(minY, vertices[i].pos[1]);
                maxX = std::max(maxX, vertices[i].pos[0]);
                maxY = std::max(maxY, vertices[i].pos[1]);
            }

            float hSize = maxX - minX;
            float vSize = maxY - minY;

            // scale evenly
            float scale = std::min(screenWidth / hSize, screenHeight / vSize);

            if (Config::IntegerScaling)
                scale = floor(scale);

            for (int i = 0; i < 8; i++)
            {
                vertices[i].pos[0] *= scale;
                vertices[i].pos[1] *= scale;
            }
        }
        else
        {
            int primOffset = sizing == 1 ? 0 : 4;
            int secOffset = sizing == 1 ? 4 : 0;

            float primMinX = 100000.f, primMaxX = -100000.f;
            float primMinY = 100000.f, primMaxY = -100000.f;
            float secMinX = 100000.f, secMaxX = -100000.f;
            float secMinY = 100000.f, secMaxY = -100000.f;

            for (int i = 0; i < 4; i++)
            {
                primMinX = std::min(primMinX, vertices[i + primOffset].pos[0]);
                primMinY = std::min(primMinY, vertices[i + primOffset].pos[1]);
                primMaxX = std::max(primMaxX, vertices[i + primOffset].pos[0]);
                primMaxY = std::max(primMaxY, vertices[i + primOffset].pos[1]);
            }
            for (int i = 0; i < 4; i++)
            {
                secMinX = std::min(secMinX, vertices[i + secOffset].pos[0]);
                secMinY = std::min(secMinY, vertices[i + secOffset].pos[1]);
                secMaxX = std::max(secMaxX, vertices[i + secOffset].pos[0]);
                secMaxY = std::max(secMaxY, vertices[i + secOffset].pos[1]);
            }

            float primHSize = layout == 1 ? std::max(primMaxX, -primMinX) : primMaxX - primMinX;
            float primVSize = layout == 0 ? std::max(primMaxY, -primMinY) : primMaxY - primMinY;

            float secHSize = layout == 1 ? std::max(secMaxX, -secMinX) : secMaxX - secMinX;
            float secVSize = layout == 0 ? std::max(secMaxY, -secMinY) : secMaxY - secMinY;

            float primScale = std::min(screenWidth / primHSize, screenHeight / primVSize);
            float secScale = 1.f;

            if (layout == 0)
            {
                if (screenHeight - primVSize * primScale < secVSize)
                    primScale = std::min((screenWidth - secHSize) / primHSize, (screenHeight - secVSize) / primVSize);
                else
                    secScale = std::min((screenHeight - primVSize * primScale) / secVSize, screenWidth / secHSize);
            }
            else
            {
                if (screenWidth - primHSize * primScale < secHSize)
                    primScale = std::min((screenWidth - secHSize) / primHSize, (screenHeight - secVSize) / primVSize);
                else
                    secScale = std::min((screenWidth - primHSize * primScale) / secHSize, screenHeight / secVSize);
            }

            if (Config::IntegerScaling)
                primScale = floor(primScale);
            if (Config::IntegerScaling)
                secScale = floor(secScale);

            for (int i = 0; i < 4; i++)
            {
                vertices[i + primOffset].pos[0] *= primScale;
                vertices[i + primOffset].pos[1] *= primScale;
            }
            for (int i = 0; i < 4; i++)
            {
                vertices[i + secOffset].pos[0] *= secScale;
                vertices[i + secOffset].pos[1] *= secScale;
            }
        }
    }

    // position
    {
        float minX = 100000.f, maxX = -100000.f;
        float minY = 100000.f, maxY = -100000.f;

        for (int i = 0; i < 8; i++)
        {
            minX = std::min(minX, vertices[i].pos[0]);
            minY = std::min(minY, vertices[i].pos[1]);
            maxX = std::max(maxX, vertices[i].pos[0]);
            maxY = std::max(maxY, vertices[i].pos[1]);
        }

        float width = maxX - minX;
        float height = maxY - minY;

        float botMaxX = -1000000.f, botMaxY = -1000000.f;
        float botMinX = 1000000.f, botMinY = 1000000.f;
        for (int i = 0; i < 8; i++)
        {
            vertices[i].pos[0] = floor(vertices[i].pos[0] - minX + screenWidth / 2 - width / 2);
            vertices[i].pos[1] = floor(vertices[i].pos[1] - minY + screenHeight / 2 - height / 2);

            if (i >= 4)
            {
                botMinX = std::min(vertices[i].pos[0], botMinX);
                botMinY = std::min(vertices[i].pos[1], botMinY);
                botMaxX = std::max(vertices[i].pos[0], botMaxX);
                botMaxY = std::max(vertices[i].pos[1], botMaxY);
            }
        }

        botX = botMinX;
        botY = botMinY;
        botWidth = botMaxX - botMinX;
        botHeight = botMaxY - botMinY;
    }

    memcpy(screenVerticesMem.GetCpuAddr(), vertices, sizeof(ImDrawVert)*8);
}

const u32 keyMappings[] = {
    KEY_A,
    KEY_B,
    KEY_MINUS,
    KEY_PLUS,
    KEY_DRIGHT | KEY_LSTICK_RIGHT,
    KEY_DLEFT  | KEY_LSTICK_LEFT,
    KEY_DUP    | KEY_LSTICK_UP,
    KEY_DDOWN  | KEY_LSTICK_DOWN,
    KEY_R,
    KEY_L,
    KEY_X,
    KEY_Y
};


static u64 MicWavLength;
static u64 MicBufferReadPos;
static s16* MicWavBuffer = NULL;

void loadMicSample()
{
    unsigned int channels, sampleRate;
    drwav_uint64 totalSamples;
    drwav_int16* result = drwav_open_file_and_read_pcm_frames_s16("/melonds/micsample.wav", 
        &channels, &sampleRate, &totalSamples, NULL);
    
    const u64 dstfreq = 44100;

    if (result && channels == 1 && totalSamples >= 735 && sampleRate == dstfreq)
    {
        MicWavBuffer = result;
        MicWavLength = totalSamples;
    }
}

void freeMicSample()
{
    free(MicWavBuffer);
}

void feedMicAudio(u32 state)
{
    if (!MicWavBuffer)
        return;
    if (state == 0)
    {
        NDS::MicInputFrame(NULL, 0);
        return;
    }
    if ((MicBufferReadPos + 735) > MicWavLength)
    {
        s16 tmp[735];
        u32 len1 = MicWavLength - MicBufferReadPos;
        memcpy(&tmp[0], &MicWavBuffer[MicBufferReadPos], len1*sizeof(s16));
        memcpy(&tmp[len1], &MicWavBuffer[0], (735 - len1)*sizeof(s16));

        NDS::MicInputFrame(tmp, 735);
        MicBufferReadPos = 735 - len1;
    }
    else
    {
        NDS::MicInputFrame(&MicWavBuffer[MicBufferReadPos], 735);
        MicBufferReadPos += 735;
    }
}

static bool running = true;
static bool paused = true;
static void* audMemPool = NULL;
static AudioDriver audDrv;

const int AudioSampleSize = 768 * 2 * sizeof(s16);

/*void LoadState(int slot)
{
    int prevstatus = EmuRunning;
    EmuRunning = 2;
    while (EmuStatus != 2);

    char filename[1024];

    if (slot > 0)
    {
        GetSavestateName(slot, filename, 1024);
    }
    else
    {
        char* file = uiOpenFile(MainWindow, "melonDS savestate (any)|*.ml1;*.ml2;*.ml3;*.ml4;*.ml5;*.ml6;*.ml7;*.ml8;*.mln", Config::LastROMFolder);
        if (!file)
        {
            EmuRunning = prevstatus;
            return;
        }

        strncpy(filename, file, 1023);
        filename[1023] = '\0';
        uiFreeText(file);
    }

    if (!Platform::FileExists(filename))
    {
        char msg[64];
        if (slot > 0) sprintf(msg, "State slot %d is empty", slot);
        else          sprintf(msg, "State file does not exist");
        OSD::AddMessage(0xFFA0A0, msg);

        EmuRunning = prevstatus;
        return;
    }

    u32 oldGBACartCRC = GBACart::CartCRC;

    // backup
    Savestate* backup = new Savestate("timewarp.mln", true);
    NDS::DoSavestate(backup);
    delete backup;

    bool failed = false;

    Savestate* state = new Savestate(filename, false);
    if (state->Error)
    {
        delete state;

        uiMsgBoxError(MainWindow, "Error", "Could not load savestate file.");

        // current state might be crapoed, so restore from sane backup
        state = new Savestate("timewarp.mln", false);
        failed = true;
    }

    NDS::DoSavestate(state);
    delete state;

    if (!failed)
    {
        if (Config::SavestateRelocSRAM && ROMPath[0][0]!='\0')
        {
            strncpy(PrevSRAMPath[0], SRAMPath[0], 1024);

            strncpy(SRAMPath[0], filename, 1019);
            int len = strlen(SRAMPath[0]);
            strcpy(&SRAMPath[0][len], ".sav");
            SRAMPath[0][len+4] = '\0';

            NDS::RelocateSave(SRAMPath[0], false);
        }

        bool loadedPartialGBAROM = false;

        // in case we have a GBA cart inserted, and the GBA ROM changes
        // due to having loaded a save state, we do not want to reload
        // the previous cartridge on reset, or commit writes to any
        // loaded save file. therefore, their paths are "nulled".
        if (GBACart::CartInserted && GBACart::CartCRC != oldGBACartCRC)
        {
            ROMPath[1][0] = '\0';
            SRAMPath[1][0] = '\0';
            loadedPartialGBAROM = true;
        }

        char msg[64];
        if (slot > 0) sprintf(msg, "State loaded from slot %d%s",
                        slot, loadedPartialGBAROM ? " (GBA ROM header only)" : "");
        else          sprintf(msg, "State loaded from file%s",
                        loadedPartialGBAROM ? " (GBA ROM header only)" : "");
        OSD::AddMessage(0, msg);

        SavestateLoaded = true;
        uiMenuItemEnable(MenuItem_UndoStateLoad);
    }

    EmuRunning = prevstatus;
}

int SaveState(int slot)
{
    char filename[1024];

    if (slot > 0)
    {
        GetSavestateName(slot, filename, 1024);
    }
    else
    {
        char* file = uiSaveFile(MainWindow, "melonDS savestate (*.mln)|*.mln", Config::LastROMFolder);
        if (!file)
        {
            EmuRunning = prevstatus;
            return;
        }

        strncpy(filename, file, 1023);
        filename[1023] = '\0';
        uiFreeText(file);
    }

    Savestate* state = new Savestate(filename, true);
    if (state->Error)
    {
        delete state;

        return 0;
    }
    else
    {
        NDS::DoSavestate(state);
        delete state;

        if (slot > 0)
            uiMenuItemEnable(MenuItem_LoadStateSlot[slot-1]);

        if (Config::SavestateRelocSRAM && ROMPath[0][0]!='\0')
        {
            strncpy(SRAMPath[0], filename, 1019);
            int len = strlen(SRAMPath[0]);
            strcpy(&SRAMPath[0][len], ".sav");
            SRAMPath[0][len+4] = '\0';

            NDS::RelocateSave(SRAMPath[0], true);
        }
    }

    char msg[64];
    if (slot > 0) sprintf(msg, "State saved to slot %d", slot);
    else          sprintf(msg, "State saved to file");
    OSD::AddMessage(0, msg);

    EmuRunning = prevstatus;
}

void UndoStateLoad()
{
    if (!SavestateLoaded) return;

    int prevstatus = EmuRunning;
    EmuRunning = 2;
    while (EmuStatus != 2);

    // pray that this works
    // what do we do if it doesn't???
    // but it should work.
    Savestate* backup = new Savestate("timewarp.mln", false);
    NDS::DoSavestate(backup);
    delete backup;

    if (ROMPath[0][0]!='\0')
    {
        strncpy(SRAMPath[0], PrevSRAMPath[0], 1024);
        NDS::RelocateSave(SRAMPath[0], false);
    }

    OSD::AddMessage(0, "State load undone");

    EmuRunning = prevstatus;
}*/

void setupAudio()
{
    static const AudioRendererConfig arConfig =
    {
        .output_rate     = AudioRendererOutputRate_48kHz,
        .num_voices      = 4,
        .num_effects     = 0,
        .num_sinks       = 1,
        .num_mix_objs    = 1,
        .num_mix_buffers = 2,
    };

    Result code;
    if (!R_SUCCEEDED(code = audrenInitialize(&arConfig)))
    {
        printf("audren init failed! %d\n", code);
        abort();
    }

    if (!R_SUCCEEDED(code = audrvCreate(&audDrv, &arConfig, 2)))
    {
        printf("audrv create failed! %d\n", code);
        abort();
    }

    const int poolSize = (AudioSampleSize * 2 + (AUDREN_MEMPOOL_ALIGNMENT-1)) & ~(AUDREN_MEMPOOL_ALIGNMENT-1);
    audMemPool = memalign(AUDREN_MEMPOOL_ALIGNMENT, poolSize);

    int mpid = audrvMemPoolAdd(&audDrv, audMemPool, poolSize);
    audrvMemPoolAttach(&audDrv, mpid);

    static const u8 sink_channels[] = { 0, 1 };
    audrvDeviceSinkAdd(&audDrv, AUDREN_DEFAULT_DEVICE_NAME, 2, sink_channels);

    audrvUpdate(&audDrv);

    if (!R_SUCCEEDED(code = audrenStartAudioRenderer()))
        printf("audrv create failed! %d\n", code);

    if (!audrvVoiceInit(&audDrv, 0, 2, PcmFormat_Int16, 32823)) // cheating
        printf("failed to create voice\n");

    audrvVoiceSetDestinationMix(&audDrv, 0, AUDREN_FINAL_MIX_ID);
    audrvVoiceSetMixFactor(&audDrv, 0, 1.0f, 0, 0);
    audrvVoiceSetMixFactor(&audDrv, 0, 1.0f, 1, 1);
    audrvVoiceStart(&audDrv, 0);
}

void audioOutput(void *args)
{
    AudioDriverWaveBuf buffers[2];
    memset(&buffers[0], 0, sizeof(AudioDriverWaveBuf) * 2);
    for (int i = 0; i < 2; i++)
    {
        buffers[i].data_pcm16 = (s16*)audMemPool;
        buffers[i].size = AudioSampleSize;
        buffers[i].start_sample_offset = i * AudioSampleSize / 2 / sizeof(s16);
        buffers[i].end_sample_offset = buffers[i].start_sample_offset + AudioSampleSize / 2 / sizeof(s16);
    }

    while (running)
    {
        while (paused && running)
        {
            svcSleepThread(17000000); // a bit more than a frame...
        }
        while (!paused && running)
        {
            AudioDriverWaveBuf* refillBuf = NULL;
            for (int i = 0; i < 2; i++)
            {
                if (buffers[i].state == AudioDriverWaveBufState_Free || buffers[i].state == AudioDriverWaveBufState_Done)
                {
                    refillBuf = &buffers[i];
                    break;
                }
            }

            if (refillBuf)
            {
                s16* data = (s16*)audMemPool + refillBuf->start_sample_offset * 2;
                
                int nSamples = 0;
                while (running && !(nSamples = SPU::ReadOutput(data, 768)))
                    svcSleepThread(1000);
                
                u32 last = ((u32*)data)[nSamples - 1];
                while (nSamples < 768)
                    ((u32*)data)[nSamples++] = last;

                armDCacheFlush(data, nSamples * 2 * sizeof(u16));
                refillBuf->end_sample_offset = refillBuf->start_sample_offset + nSamples;

                audrvVoiceAddWaveBuf(&audDrv, 0, refillBuf);
                audrvVoiceStart(&audDrv, 0);
            }

            audrvUpdate(&audDrv);
            audrenWaitFrame();
        }
    }
}

u64 sectionStartTick;
u64 sectionTicksTotal;
int entered = 0;

void EnterProfileSection()
{
    entered++;
    sectionStartTick = armGetSystemTick();
}

void CloseProfileSection()
{
    sectionTicksTotal += armGetSystemTick() - sectionStartTick;
}

ClkrstSession cpuOverclockSession;
bool usePCV;
void onAppletHook(AppletHookType hook, void *param)
{
    if (hook == AppletHookType_OnOperationMode || hook == AppletHookType_OnPerformanceMode
        || hook == AppletHookType_OnRestart || hook == AppletHookType_OnExitRequest)
    {
        applyOverclock(usePCV, &cpuOverclockSession, Config::SwitchOverclock);
    }
}

// tbh idk why I even bother with C strings
struct Filebrowser
{
    Filebrowser()
    {
        Entry entry;
        entry.isDir = true;
        entry.name = new char[3];
        strcpy(entry.name, "..");
        entries.push_back(entry);
    }

    ~Filebrowser()
    {
        delete[] entries[0].name;
    }

    void EnterDirectory(const char* path)
    {
        DIR* dir = opendir(path);
        if (dir == NULL)
        {
            path = "/";
            dir = opendir(path);
        }

        for (int i = 1; i < entries.size(); i++)
            delete[] entries[i].name;
        entries.resize(1);

        strcpy(curdir, path);

        curfile[0] = '\0';
        entryselected = NULL;
        struct dirent* cur;
        while (cur = readdir(dir))
        {
            Entry entry;
            int nameLen = strlen(cur->d_name);
            if (nameLen == 1 && cur->d_name[0] == '.')
                continue;
            if (cur->d_type == DT_REG)
            {
                if (nameLen < 4)
                    continue;
                if (cur->d_name[nameLen - 4] != '.' 
                    || cur->d_name[nameLen - 3] != 'n' 
                    || cur->d_name[nameLen - 2] != 'd' 
                    || cur->d_name[nameLen - 1] != 's')
                    continue;

                entry.name = new char[nameLen+1];
                strcpy(entry.name, cur->d_name);
                entry.isDir = false;
            }
            else if (cur->d_type == DT_DIR)
            {
                entry.name = new char[nameLen+1];
                strcpy(entry.name, cur->d_name);
                entry.isDir = true;
            }
            entries.push_back(entry);
        }

        closedir(dir);
    }

    void MoveIntoDirectory(const char* name)
    {
        int curpathlen = strlen(curdir);
        if (curpathlen > 1)
            curdir[curpathlen] = '/';
        else
            curpathlen = 0;
        strcpy(curdir + curpathlen + 1, name);
        EnterDirectory(curdir);
    }
    void MoveUpwards()
    {
        int len = strlen(curdir);
        if (len > 1)
        {
            for (int i = len - 1; i >= 0; i--)
            {
                if (curdir[i] == '/')
                {
                    if (i == 0)
                        curdir[i + 1] = '\0';
                    else
                        curdir[i] = '\0';
                    break;
                }
            }
            EnterDirectory(curdir);
        }
    }

    void Draw()
    {
        if (ImGui::BeginCombo("Browse files", curfile[0] == '\0' ? curdir : curfile))
        {
            for (int i = 0; i < entries.size(); i++)
            {
                ImGui::PushID(entries[i].name);
                if (ImGui::Selectable(entries[i].name, entryselected == entries[i].name))
                {
                    if (entries[i].isDir)
                    {
                        if (i == 0)
                            MoveUpwards();
                        else
                            MoveIntoDirectory(entries[i].name);
                    }
                    else
                    {
                        entryselected = entries[i].name;
                        strcpy(curfile, curdir);
                        int dirlen = strlen(curdir);
                        curfile[dirlen] = '/';
                        strcpy(curfile + dirlen + 1, entries[i].name);
                    }
                }
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
    }

    bool HasFileSelected()
    {
        return entryselected != NULL;
    }

    struct Entry
    {
        char* name;
        bool isDir;
    };
    int curItemsCount;
    std::vector<Entry> entries;
    char curdir[256];
    char curfile[256];
    char* entryselected;
};


const int clockSpeeds[] = { 1020000000, 1224000000, 1581000000, 1785000000 };

int main(int argc, char* argv[])
{
    /*setenv("EGL_LOG_LEVEL", "debug", 1);
    setenv("MESA_VERBOSE", "all", 1);
    setenv("NOUVEAU_MESA_DEBUG", "1", 1);

    setenv("NV50_PROG_OPTIMIZE", "0", 1);
    setenv("NV50_PROG_DEBUG", "1", 1);
    setenv("NV50_PROG_CHIPSET", "0x120", 1);*/
//#ifdef GDB_ENABLED
    socketInitializeDefault();
    int nxlinkSocket = nxlinkStdio();
    //GDBStub_Init();
    //GDBStub_Breakpoint();
//#endif

    romfsInit();

    AppletHookCookie aptCookie;
    appletLockExit();
    appletHook(&aptCookie, onAppletHook, NULL);

    Config::Load();

    loadMicSample();

    int screenWidth, screenHeight;
    int cursorX = 0, cursorY = 0;
    float cursorVelX = 0.f, cursorVelY = 0.f;
    int touchscreenInputSource = 0;

    if (Config::GlobalRotation % 2 == 0)
    {
        screenWidth = 1280;
        screenHeight = 720;
    }
    else
    {
        screenWidth = 720;
        screenHeight = 1280;
    }

    usePCV = hosversionBefore(8, 0, 0);
    if (usePCV)
    {
        pcvInitialize();
    }
    else
    {
        clkrstInitialize();
        clkrstOpenSession(&cpuOverclockSession, PcvModuleId_CpuBus, 0);
    }
    applyOverclock(usePCV, &cpuOverclockSession, Config::SwitchOverclock);

    graphicsInitialize();

    ImGui::CreateContext();
    ImGui::StyleColorsClassic();

    ImGuiStyle& style = ImGui::GetStyle();
    style.TouchExtraPadding = ImVec2(4, 4);
    style.ScaleAllSizes(2.f);

    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = 1.5f;

    ImGui_ImplDeko3D_Init(gDevice, gQueue, allocShader, allocData, allocTexture, allocTmp, resetTmp);

    ImGui_ImplDeko3D_GetImageDescriptor(1)->initialize(dk::ImageView{rotatedFb});
    ImGui_ImplDeko3D_GetImageDescriptor(2)->initialize(dk::ImageView{screenTexture});

    updateScreenLayout(screenWidth, screenHeight);

    Thread audioThread;
    setupAudio();
    threadCreate(&audioThread, audioOutput, NULL, NULL, 0x8000, 0x30, 2);
    threadStart(&audioThread);

    printf("melonDS " MELONDS_VERSION "\n");
    printf(MELONDS_URL "\n");

    NDS::Init();

    Config::JIT_Enable = true;

    Config::Threaded3D = true;

    GPU3D::InitRenderer(false);

    float frametimeHistogram[60] = {0};
    float frametimeDiffHistogram[60] = {0};
    float customTimeHistogram[60] = {0};

    int guiState = 0;
    float frametimeSum = 0.f;
    float frametimeSum2 = 0.f;
    float frametimeMax = 0.f;
    float frametimeStddev = 0.f;

    const char* requiredFiles[] = {"romlist.bin", "bios9.bin", "bios7.bin", "firmware.bin"};
    int filesReady = 0;
    {
        FILE* f;
        for (int i = 0; i < sizeof(requiredFiles)/sizeof(requiredFiles[0]); i++)
        {
            if ((f = Platform::OpenLocalFile(requiredFiles[i], "rb")))
            {
                fclose(f);
                filesReady |= 1 << i;
            }
        }
    }

    bool showGui = true;
    bool navInput = true;

    FILE* perfRecord = NULL;
    int perfRecordMode = 0;

    std::vector<std::pair<u32, u32>> jitFreqResults;

    Filebrowser filebrowser;
    filebrowser.EnterDirectory(Config::LastROMFolder);
    char* romSramPath = NULL;

    bool lidClosed = false;
    u32 microphoneState = 0;

    int mainScreenPos[3];

    float joyconRightCalibration[3] = {1.f, 0.f, 0.f};
    float joyconUpCalibration[3] = {0.f, 0.f, 1.f};
    bool touchDownKey = false;

    u32 joyconSixaxisHandles[2];
    hidGetSixAxisSensorHandles(joyconSixaxisHandles, 2, CONTROLLER_PLAYER_1, TYPE_JOYCON_PAIR);
    hidStartSixAxisSensor(joyconSixaxisHandles[0]);
    hidStartSixAxisSensor(joyconSixaxisHandles[1]);
    SixAxisSensorValues sixaxisValues;

    while (appletMainLoop() && running)
    {
        hidScanInput();

        u32 keysDown = hidKeysDown(CONTROLLER_P1_AUTO);
        u32 keysUp = hidKeysUp(CONTROLLER_P1_AUTO);
        u32 keysHeld = hidKeysHeld(CONTROLLER_P1_AUTO);

        bool touchDown = false;

        if (guiState > 0 && keysDown & KEY_ZL)
        {
            if (!showGui)
            {
                for (int i = 0; i < 12; i++)
                    NDS::ReleaseKey(i > 9 ? i + 6 : i);
                NDS::ReleaseScreen();

                NDS::MicInputFrame(NULL, 0);
                microphoneState = 0;
            }

            showGui ^= true;
            navInput = showGui;
        }

        {
            ImGuiIO& io = ImGui::GetIO();
            io.DisplaySize = ImVec2(screenWidth, screenHeight);
            io.MouseDown[0] = false;

            if (!navInput)
            {
                u32 rotatedKeyMappings[12];
                memcpy(rotatedKeyMappings, keyMappings, 4*12);
                switch (Config::ScreenRotation)
                {
                    case 0: // nothing needs to be handled
                        break;
                    case 1: // 90 degrees
                        rotatedKeyMappings[4] = keyMappings[6]; // right -> up
                        rotatedKeyMappings[5] = keyMappings[7]; // left -> down
                        rotatedKeyMappings[6] = keyMappings[5]; // up -> left
                        rotatedKeyMappings[7] = keyMappings[4]; // down -> right

                        rotatedKeyMappings[0] = keyMappings[10]; // X -> A
                        rotatedKeyMappings[1] = keyMappings[0]; // A -> B
                        rotatedKeyMappings[10] = keyMappings[11]; // X -> Y
                        rotatedKeyMappings[11] = keyMappings[1]; // Y -> B
                        break;
                    case 2: // 180 degrees
                        rotatedKeyMappings[4] = keyMappings[5]; // right -> left
                        rotatedKeyMappings[5] = keyMappings[4]; // left -> right
                        rotatedKeyMappings[6] = keyMappings[7]; // up -> down
                        rotatedKeyMappings[7] = keyMappings[6]; // down -> up

                        rotatedKeyMappings[0] = keyMappings[11]; // Y -> A
                        rotatedKeyMappings[1] = keyMappings[10]; // X -> B
                        rotatedKeyMappings[10] = keyMappings[1]; // B -> X
                        rotatedKeyMappings[11] = keyMappings[0]; // A -> Y
                        break;
                    case 3: // 270 degrees
                        rotatedKeyMappings[4] = keyMappings[7]; // right -> down
                        rotatedKeyMappings[5] = keyMappings[6]; // left -> up
                        rotatedKeyMappings[6] = keyMappings[4]; // up -> right
                        rotatedKeyMappings[7] = keyMappings[5]; // down -> left

                        rotatedKeyMappings[0] = keyMappings[1]; // A -> B
                        rotatedKeyMappings[1] = keyMappings[11]; // B -> Y
                        rotatedKeyMappings[10] = keyMappings[0]; // X -> A
                        rotatedKeyMappings[11] = keyMappings[10]; // Y -> X
                        break;
                }

                for (int i = 0; i < 12; i++)
                {
                    if (keysDown & rotatedKeyMappings[i])
                        NDS::PressKey(i > 9 ? i + 6 : i);
                    if (keysUp & rotatedKeyMappings[i])
                        NDS::ReleaseKey(i > 9 ? i + 6 : i);
                }

                if (keysDown & KEY_LSTICK)
                    microphoneState = 1;
                if (keysUp & KEY_LSTICK)
                    microphoneState = 0;

                feedMicAudio(microphoneState);

                if (keysDown & KEY_RSTICK)
                {
                    switch (Config::ScreenSizing)
                    {
                    case 0:
                        Config::ScreenSizing = AutoScreenSizing == 0 ? 1 : AutoScreenSizing;
                        break;
                    case 1:
                        Config::ScreenSizing = 2;
                        break;
                    case 2:
                        Config::ScreenSizing = 1;
                        break;
                    case 3:
                        Config::ScreenSizing = AutoScreenSizing == 2 ? 1 : 2;
                        break;
                    }
                    updateScreenLayout(screenWidth, screenHeight);
                }
            }
            else
            {
                JoystickPosition lstick;
                hidJoystickRead(&lstick, CONTROLLER_P1_AUTO, JOYSTICK_LEFT);
            #define MAPNAV(name, key) io.NavInputs[ImGuiNavInput_##name] = keysHeld & KEY_##key ? 1.f : 0.f
                MAPNAV(Activate, A);
                MAPNAV(Cancel, B);
                MAPNAV(Input, X);
                MAPNAV(Menu, Y);
                MAPNAV(DpadLeft, DLEFT);
                MAPNAV(DpadRight, DRIGHT);
                MAPNAV(DpadUp, DUP);
                MAPNAV(DpadDown, DDOWN);
                MAPNAV(FocusNext, R);
                MAPNAV(FocusPrev, L);
                if (lstick.dy < 0)
                    io.NavInputs[ImGuiNavInput_LStickDown] = (float)lstick.dy / JOYSTICK_MIN;
                if (lstick.dy > 0)
                    io.NavInputs[ImGuiNavInput_LStickUp] = (float)lstick.dy / JOYSTICK_MAX;
                if (lstick.dx < 0)
                    io.NavInputs[ImGuiNavInput_LStickLeft] = (float)lstick.dx / JOYSTICK_MIN;
                if (lstick.dx > 0)
                    io.NavInputs[ImGuiNavInput_LStickRight] = (float)lstick.dx / JOYSTICK_MAX;
            }

            if (Config::CursorClickMode == 0)
                touchDownKey = keysHeld & KEY_ZR;
            else if (keysDown & KEY_ZR)
                touchDownKey ^= true;

            JoystickPosition rstick;
            hidJoystickRead(&rstick, CONTROLLER_P1_AUTO, JOYSTICK_RIGHT);

            hidSixAxisSensorValuesRead(&sixaxisValues, CONTROLLER_P1_AUTO, 1);

            if (Config::CursorMode < 2)
            {
                if (rstick.dx * rstick.dx + rstick.dy * rstick.dy < (JOYSTICK_MAX / 10) * (JOYSTICK_MAX / 10))
                {
                    cursorVelX = 0.f;
                    cursorVelY = 0.f;
                }
                else
                {
                    touchscreenInputSource = 0;

                    cursorVelX += (float)rstick.dx / JOYSTICK_MAX * std::max(botWidth, botHeight) / 8.f;
                    cursorVelY += (float)-rstick.dy / JOYSTICK_MAX * std::max(botWidth, botHeight) / 8.f;

                    float maxSpeed = std::max(botWidth, botHeight) * 1.5f;

                    if (cursorVelX < -maxSpeed)
                        cursorVelX = -maxSpeed;
                    if (cursorVelX > maxSpeed)
                        cursorVelX = maxSpeed;
                    if (cursorVelY < -maxSpeed)
                        cursorVelY = -maxSpeed;
                    if (cursorVelY > maxSpeed)
                        cursorVelY = maxSpeed;

                    // allow for quick turns
                    if ((cursorVelX > 0.f && rstick.dx < 0) || (cursorVelX < 0.f && rstick.dx > 0))
                        cursorVelX = 0.f;
                    if ((cursorVelY > 0.f && rstick.dy > 0) || (cursorVelY < 0.f && rstick.dy < 0))
                        cursorVelY = 0.f;
                }

                if (Config::CursorMode == 0) // mouse mode
                {
                    cursorX += cursorVelX / 60.f; // framerate independent eehhh
                    cursorY += cursorVelY / 60.f;
                }
                else // offset mode
                {
                    cursorX = (botX + botWidth / 2) + (float)rstick.dx / JOYSTICK_MAX / 2.f * botWidth;
                    cursorY = (botY + botHeight / 2) - (float)rstick.dy / JOYSTICK_MAX / 2.f * botHeight;
                }
            }
            else
            {
                float forward[] = {sixaxisValues.orientation[1].x, sixaxisValues.orientation[1].y,
                    sixaxisValues.orientation[1].z};
                xv_norm(forward, forward, 3);

                float xAngle = xv3_dot(forward, joyconRightCalibration);
                float zAngle = xv3_dot(forward, joyconUpCalibration);

                touchscreenInputSource = 0;

                cursorX = (botX + botWidth / 2) + xAngle / M_PI * 8.f * std::max(botWidth, botHeight);
                cursorY = (botY + botHeight / 2) - zAngle / M_PI * 8.f * std::max(botWidth, botHeight);
            
                if (keysHeld & KEY_ZR && keysHeld & KEY_ZL)
                {
                    joyconUpCalibration[0] = sixaxisValues.orientation[2].x;
                    joyconUpCalibration[1] = sixaxisValues.orientation[2].y;
                    joyconUpCalibration[2] = sixaxisValues.orientation[2].z;
                    joyconRightCalibration[0] = sixaxisValues.orientation[0].x;
                    joyconRightCalibration[1] = sixaxisValues.orientation[0].y;
                    joyconRightCalibration[2] = sixaxisValues.orientation[0].z;
                    
                    xv_norm(joyconUpCalibration, joyconUpCalibration, 3);
                    xv_norm(joyconRightCalibration, joyconRightCalibration, 3);
                }
            }

            if (cursorX < botX)
            {
                cursorX = botX - 1;
                touchscreenInputSource = 1;
            }
            else if (cursorX >= botX + botWidth)
            {
                cursorX = botX + botWidth;
                touchscreenInputSource = 1;
            }
            if (cursorY < botY)
            {
                cursorY = botY - 1;
                touchscreenInputSource = 1;
            }
            else if (cursorY >= botY + botHeight)
            {
                cursorY = botY + botHeight;
                touchscreenInputSource = 1;
            }

            if (hidTouchCount() > 0)
            {
                io.MouseDrawCursor = false;
                touchPosition pos;
                hidTouchRead(&pos, 0);

                float rotatedTouch[2];
                switch (Config::GlobalRotation)
                {
                case 0: rotatedTouch[0] = pos.px; rotatedTouch[1] = pos.py; break;
                case 1: rotatedTouch[0] = pos.py; rotatedTouch[1] = 1280.f - pos.px; break;
                case 2: rotatedTouch[0] = 1280.f - pos.px; rotatedTouch[1] = 720.f - pos.py; break;
                case 3: rotatedTouch[0] = 720.f - pos.py; rotatedTouch[1] = pos.px; break;
                }

                if (showGui)
                {
                    io.MousePos = ImVec2(rotatedTouch[0], rotatedTouch[1]);
                    io.MouseDown[0] = true;
                }

                if (!io.WantCaptureMouse && rotatedTouch[0] >= botX && rotatedTouch[0] < (botX + botWidth) && rotatedTouch[1] >= botY && rotatedTouch[1] < (botY + botHeight))
                {
                    touchscreenInputSource = 1;

                    cursorX = rotatedTouch[0];
                    cursorY = rotatedTouch[1];

                    touchDownKey = false;
                    touchDown = true;
                }
            }

            touchDown |= touchDownKey;

            if (touchDown)
            {
                int x, y;
                if (Config::ScreenRotation == 0) // 0
                {
                    x = (cursorX - botX) * 256.0f / botWidth;
                    y = (cursorY - botY) * 256.0f / botWidth;
                }
                else if (Config::ScreenRotation == 1) // 90
                {
                    x = (cursorY - botY) * -192.0f / botWidth;
                    y = (cursorX - botX) *  192.0f / botWidth;
                }
                else if (Config::ScreenRotation == 2) // 180
                {
                    x =       (cursorX - botX) * -256.0f / botWidth;
                    y = 192 - (cursorY - botY) *  256.0f / botWidth;
                }
                else // 270
                {
                    x =       (cursorY - botY) * 192.0f / botWidth;
                    y = 192 - (cursorX - botX) * 192.0f / botWidth;
                }
                NDS::PressKey(16 + 6);
                NDS::TouchScreen(x, y);
            }
            else
            {
                NDS::ReleaseKey(16 + 6);
                NDS::ReleaseScreen();
            }
        }

        ImGui::NewFrame();

        paused = guiState != 1;

        if (guiState == 1)
        {
            entered = 0;
            sectionTicksTotal = 0;

            if (touchDown && touchscreenInputSource == 0)
            {
                ImGui::GetForegroundDrawList()->AddCircleFilled(ImVec2(cursorX, cursorY), 14.f, ImColor(255, 200, 0));
            }
            else if (!touchDown && touchscreenInputSource == 0)
                ImGui::GetForegroundDrawList()->AddRect(ImVec2(cursorX - 8, cursorY - 8), 
                    ImVec2(cursorX + 8, cursorY + 8), ImColor(127, 127, 127), 0.f, 15, 4.f);

            //arm9BlockFrequency.clear();
            //arm7BlockFrequency.clear();

            u64 frameStartTime = armGetSystemTick();
            NDS::RunFrame();
            u64 frameEndTime = armGetSystemTick();

            {
                mainScreenPos[2] = mainScreenPos[1];
                mainScreenPos[1] = mainScreenPos[0];
                mainScreenPos[0] = NDS::PowerControl9 >> 15;
                int guess;
                if (mainScreenPos[0] == mainScreenPos[2] &&
                    mainScreenPos[0] != mainScreenPos[1])
                {
                    // constant flickering, likely displaying 3D on both screens
                    // TODO: when both screens are used for 2D only...???
                    guess = 0;
                }
                else
                {
                    if (mainScreenPos[0] == 1)
                        guess = 1;
                    else
                        guess = 2;
                }

                if (guess != AutoScreenSizing)
                {
                    AutoScreenSizing = guess;
                    updateScreenLayout(screenWidth, screenHeight);
                }
            }

            profiler::Frame();

            for (int i = 0; i < 59; i++)
                customTimeHistogram[i] = customTimeHistogram[i + 1];
            customTimeHistogram[59] = (float)armTicksToNs(sectionTicksTotal) / 1000000.f;

            frametimeMax = 0.f;
            frametimeStddev = 0.f;
            frametimeSum = 0.f;
            frametimeSum2 = 0.f;
            for (int i = 0; i < 30; i++)
            {
                frametimeSum += frametimeHistogram[i + 1];
                frametimeHistogram[i] = frametimeHistogram[i + 1];
            }
            for (int i = 30; i < 59; i++)
            {
                frametimeSum += frametimeHistogram[i + 1];
                frametimeSum2 += frametimeHistogram[i + 1];
                frametimeHistogram[i] = frametimeHistogram[i + 1];
            }
            frametimeHistogram[59] = (float)armTicksToNs(frameEndTime - frameStartTime) / 1000000.f;
            frametimeSum += frametimeHistogram[59];
            frametimeSum /= 60.f;
            frametimeSum2 += frametimeHistogram[59];
            frametimeSum2 /= 30.f;
            for (int i = 0; i < 60; i++)
            {
                frametimeMax = std::max(frametimeHistogram[i], frametimeMax);
                float stdDevPartSqrt = frametimeHistogram[i] - frametimeSum;
                frametimeStddev += stdDevPartSqrt * stdDevPartSqrt;
            }
            frametimeStddev = sqrt(frametimeStddev / 60.f);

            if (perfRecordMode == 1)
                fwrite(&frametimeHistogram[59], 4, 1, perfRecord);
            else if (perfRecordMode == 2)
            {
                for (int i = 0; i < 59; i++)
                {
                    frametimeDiffHistogram[i] = frametimeDiffHistogram[i + 1];
                }
                float compValue;
                fread(&compValue, 4, 1, perfRecord);

                frametimeDiffHistogram[59] = compValue - frametimeHistogram[59];
            }

            /*jitFreqResults.clear();
            for (auto res : arm9BlockFrequency)
                jitFreqResults.push_back(res);
            std::sort(jitFreqResults.begin(), jitFreqResults.end(), [](std::pair<u32, u32>& a, std::pair<u32, u32>& b)
            {
                return a.second > b.second;
            });
            int totalBlockCalls = 0;
            for (int i = 0; i < jitFreqResults.size(); i++)
            {
                totalBlockCalls += jitFreqResults[i].second;
            }
            printf("top 20 blocks frametime(%f) %d out of %d\n", frametimeHistogram[59], totalBlockCalls, jitFreqResults.size());
            for (int i = 0; i < 20; i++)
            {
                printf("%x hit %dx\n", jitFreqResults[i].first, jitFreqResults[i].second);
            }*/
        }
        else if (filesReady != 0xF)
        {
            if (ImGui::Begin("Files missing!"))
            {
                ImGui::TextColored(ImVec4(1.f, 1.f, 0.f, 1.f), "Some files couldn't. Please make sure they're at the exact place:");
                for (int i = 0; i < sizeof(requiredFiles)/sizeof(requiredFiles[0]); i++)
                {
                    if (!(filesReady & (1 << i)))
                        ImGui::Text("File: /melonds/%s is missing", requiredFiles[i]);
                }
                if (ImGui::Button("Exit"))
                    running = false;
                
            }
            ImGui::End();
        }
        else if (guiState == 0)
        {
            if (ImGui::Begin("Select rom..."))
            {
                filebrowser.Draw();

                if (filebrowser.HasFileSelected() && ImGui::Button("Load!"))
                {
                    AutoScreenSizing = 0;
                    memset(mainScreenPos, 0, sizeof(int)*3);

                    int romNameLen = strlen(filebrowser.curfile);
                    if (romSramPath)
                        delete[] romSramPath;
                    romSramPath = new char[romNameLen + 4 + 1];
                    strcpy(romSramPath, filebrowser.curfile);
                    strcpy(romSramPath + romNameLen, ".sav");
                    NDS::LoadROM(filebrowser.curfile, romSramPath, Config::DirectBoot);

                    if (perfRecordMode == 1)
                        perfRecord = fopen("melonds_perf", "wb");
                    else if (perfRecordMode == 2)
                        perfRecord = fopen("melonds_perf", "rb");

                    guiState = 1;
                }
                
                if (ImGui::Button("Exit"))
                {
                    running = false;
                }

            }
            ImGui::End();

            if (ImGui::Begin("Settings"))
            {
                int globalRotation = Config::GlobalRotation;
                ImGui::Combo("Global rotation", &globalRotation, "0°\0" "90°\0" "180°\0" "270°\0");
                if (globalRotation != Config::GlobalRotation)
                {
                    Config::GlobalRotation = globalRotation;
                    if (Config::GlobalRotation % 2 == 0)
                    {
                        screenWidth = 1280;
                        screenHeight = 720;
                    }
                    else
                    {
                        screenWidth = 720;
                        screenHeight = 1280;
                    }
                    updateScreenLayout(screenWidth, screenHeight);
                }

                bool directBoot = Config::DirectBoot;
                ImGui::Checkbox("Boot games directly", &directBoot);
                Config::DirectBoot = directBoot;

                int newOverclock = Config::SwitchOverclock;
                ImGui::Combo("Overclock", &newOverclock, "1020 MHz\0" "1224 MHz\0" "1581 MHz\0" "1785 MHz\0");
                if (newOverclock != Config::SwitchOverclock)
                {
                    applyOverclock(usePCV, &cpuOverclockSession, newOverclock);
                    Config::SwitchOverclock = newOverclock;
                }
                ImGui::SliderInt("Block size", &Config::JIT_MaxBlockSize, 1, 32);

                bool enableBranchInlining = Config::JIT_BrancheOptimisations > 0;
                bool enableBranchLinking = Config::JIT_BrancheOptimisations == 2;
                ImGui::Checkbox("Branch optimisations", &enableBranchInlining);
                if (enableBranchInlining)
                    ImGui::Checkbox("Branch linking", &enableBranchLinking);

                if (enableBranchLinking)
                    Config::JIT_BrancheOptimisations = 2;
                else if (enableBranchInlining)
                    Config::JIT_BrancheOptimisations = 1;
                else
                    Config::JIT_BrancheOptimisations = 0;

                bool literalOptimisations = Config::JIT_LiteralOptimisations;
                ImGui::Checkbox("Literal optimisations", &literalOptimisations);
                Config::JIT_LiteralOptimisations = literalOptimisations;
            }
            ImGui::End();

            if (ImGui::Begin("Profiling"))
            {
                ImGui::Combo("Mode", &perfRecordMode, "No comparision\0Write frametimes\0Compare frametimes\0");
            }
            ImGui::End();

            if (ImGui::Begin("Help"))
            {
                ImGui::BulletText("Use the Dpad to navigate the GUI");
                ImGui::BulletText("Press A to select");
                ImGui::BulletText("Press B to cancel");
                ImGui::BulletText("Use Y and...");
                ImGui::BulletText("L/R to switch between windows");
                ImGui::BulletText("the left analogstick to move windows");
                ImGui::BulletText("the Dpad to resize windows");
            }
            ImGui::End();

            if (!MicWavBuffer)
            {
                if (ImGui::Begin("Couldn't load mic sample"))
                {
                    ImGui::BulletText("You can proceed but microphone input won't be available\n");
                    ImGui::BulletText("Make sure to put the sample into /melonds/micsample.wav");
                    ImGui::BulletText("The file has to be saved as 44100Hz mono 16-bit signed pcm and be atleast 1/60s long");
                }
                ImGui::End();
            }
        }

        if (guiState > 0)
        {
            if (showGui)
            {
                ImGui::Begin("Navigation");
                ImGui::Combo("Touch mode", &Config::CursorMode, "Mouse mode\0" "Offset mode\0" "Motion control!\0");
                ImGui::Combo("Touch click mode", &Config::CursorClickMode, "Hold\0" "Toggle\0");
                ImGui::BulletText("Using the Switch's touchscreen is always possible");
                ImGui::BulletText("Hide/unhide GUI using ZL");
                ImGui::BulletText("Press the right stick to quickly switch screen emphasis");
                ImGui::BulletText("Use ZL+ZR to recenter the gyro cursor");
                ImGui::BulletText("In mouse or motion control mode move the cursor offscreen to hide it");
                if (navInput)
                    navInput = navInput && !ImGui::Button("Give key input back to game");
                else
                    ImGui::Text("Hide and unhide the GUI to regain key input");
                ImGui::End();

                if (ImGui::Begin("Perf", NULL, ImGuiWindowFlags_AlwaysAutoResize))
                {
                    ImGui::Text("frametime avg1: %fms avg2: %fms std dev: +/%fms max: %fms %d", frametimeSum, frametimeSum2, frametimeStddev, frametimeMax, entered);
                    ImGui::PlotHistogram("Frametime history", frametimeHistogram, 60, 0, NULL, 0.f, 25.f, ImVec2(0, 50.f));
                    
                    ImGui::PlotHistogram("Custom counter", customTimeHistogram, 60, 0, NULL, 0.f, 25.f, ImVec2(0, 50.f));

                    if (perfRecordMode == 2)
                    {
                        ImGui::PlotHistogram("Frametime diff", frametimeDiffHistogram, 60, 0, NULL, -25.f, 25.f, ImVec2(0, 50.f));
                    }

                    profiler::Render();

                }
                ImGui::End();

                if (ImGui::Begin("Display settings"))
                {
                    bool displayDirty = false;

                    int globalRotation = Config::GlobalRotation;
                    ImGui::Combo("Global rotation", &globalRotation, "0°\0" "90°\0" "180°\0" "270°\0");
                    displayDirty |= globalRotation != Config::GlobalRotation;

                    int newSizing = Config::ScreenSizing;
                    ImGui::Combo("Screen Sizing", &newSizing, "Even\0Emphasise top\0Emphasise bottom\0Auto\0");
                    displayDirty |= newSizing != Config::ScreenSizing;

                    int newRotation = Config::ScreenRotation;
                    const char* rotations[] = {"0°", "90°", "180°", "270°"};
                    ImGui::Combo("Screen Rotation", &newRotation, rotations, 4);
                    displayDirty |= newRotation != Config::ScreenRotation;

                    int newGap = Config::ScreenGap;
                    const char* screenGaps[] = {"0px", "1px", "8px", "64px", "90px", "128px"};
                    ImGui::Combo("Screen Gap", &newGap, screenGaps, 6);
                    displayDirty |= newGap != Config::ScreenGap;

                    int newLayout = Config::ScreenLayout;
                    ImGui::Combo("Screen Layout", &newLayout, "Natural\0Vertical\0Horizontal\0");
                    displayDirty |= newLayout != Config::ScreenLayout;

                    bool newIntegerScale = Config::IntegerScaling;
                    ImGui::Checkbox("Integer Scaling", &newIntegerScale);
                    displayDirty |= newIntegerScale != Config::IntegerScaling;

                    if (displayDirty)
                    {
                        Config::GlobalRotation = globalRotation;
                        if (Config::GlobalRotation % 2 == 0)
                        {
                            screenWidth = 1280;
                            screenHeight = 720;
                        }
                        else
                        {
                            screenWidth = 720;
                            screenHeight = 1280;
                        }

                        Config::ScreenSizing = newSizing;
                        Config::ScreenRotation = newRotation;
                        Config::ScreenGap = newGap;
                        Config::ScreenLayout = newLayout;
                        Config::IntegerScaling = newIntegerScale;

                        updateScreenLayout(screenWidth, screenHeight);
                    }

                    bool newFiltering = Config::Filtering;
                    ImGui::Checkbox("Filtering", &newFiltering);
                    Config::Filtering = newFiltering;
                }
                ImGui::End();

                if (ImGui::Begin("Emusettings", NULL, ImGuiWindowFlags_AlwaysAutoResize))
                {
                    if (ImGui::Checkbox("Lid closed", &lidClosed))
                        NDS::SetLidClosed(lidClosed);
                    if (ImGui::Button("Reset"))
                    {
                        NDS::LoadROM(filebrowser.curfile, romSramPath, true);

                        if (perfRecord)
                        {
                            fseek(perfRecord, SEEK_SET, 0);
                        }
                    }
                    if (ImGui::Button("Stop"))
                    {
                        if (perfRecord)
                        {
                            fclose(perfRecord);
                            perfRecord = NULL;
                        }
                        guiState = 0;
                        navInput = true;
                    }
                    if (guiState == 1 && ImGui::Button("Pause"))
                        guiState = 2;
                    if (guiState == 2 && ImGui::Button("Unpause"))
                        guiState = 1;

                }
                ImGui::End();
            }
        }

        graphicsUpdate(guiState, screenWidth, screenHeight);
    }

    running = false;

    if (perfRecord)
    {
        fclose(perfRecord);
        perfRecord = NULL;
    }

    NDS::DeInit();

    strcpy(Config::LastROMFolder, filebrowser.curdir);

    Config::Save();

    if (romSramPath)
        delete[] romSramPath;
    
    hidStopSixAxisSensor(joyconSixaxisHandles[0]);
    hidStopSixAxisSensor(joyconSixaxisHandles[1]);

    threadWaitForExit(&audioThread);
    threadClose(&audioThread);

    audrvClose(&audDrv);
    audrenExit();

    free(audMemPool);

    ImGui_ImplDeko3D_Shutdown();

    ImGui::DestroyContext();

    graphicsExit();

    applyOverclock(usePCV, &cpuOverclockSession, 0);
    if (usePCV)
    {
        pcvExit();
    }
    else
    {
        clkrstCloseSession(&cpuOverclockSession);
        clkrstExit();
    }

    freeMicSample();

    appletUnhook(&aptCookie);
    appletUnlockExit();

//#ifdef GDB_ENABLED
    close(nxlinkSocket);
    socketExit();
    //GDBStub_Shutdown();
//#endif
    romfsExit();

    return 0;
}