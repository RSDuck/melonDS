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
#include <math.h>

#include <unordered_map>
#include <algorithm>
#include <vector>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <glad/glad.h>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_opengl3.h"

#include "dr_wav.h"

#include "compat_switch.h"

#include "profiler.h"

#ifdef GDB_ENABLED
#include <gdbstub.h>
#endif

static EGLDisplay eglDisplay;
static EGLContext eglCtx;
static EGLSurface eglSurface;

extern std::unordered_map<u32, u32> arm9BlockFrequency;
extern std::unordered_map<u32, u32> arm7BlockFrequency;

namespace Config
{

int ScreenRotation;
int ScreenGap;
int ScreenLayout;
int ScreenSizing;

int ControlsRotation;

bool Filtering;

char LastROMFolder[512];

int SwitchOverclock;

int DirectBoot;

ConfigEntry PlatformConfigFile[] =
{
    {"ScreenRotation", 0, &ScreenRotation, 0, NULL, 0},
    {"ScreenGap",      0, &ScreenGap,      0, NULL, 0},
    {"ScreenLayout",   0, &ScreenLayout,   0, NULL, 0},
    {"ScreenSizing",   0, &ScreenSizing,   0, NULL, 0},
    {"Filtering",      0, &Filtering,      1, NULL, 0},

    {"ControlsRotation", 0, &ControlsRotation, 0, NULL, 0},

    {"LastROMFolder", 1, LastROMFolder, 0, (char*)"sdmc:/", 511},

    {"SwitchOverclock", 0, &SwitchOverclock, 0, NULL, 0},

    {"DirectBoot",   0, &DirectBoot,     1, NULL, 0},

    {"", -1, NULL, 0, NULL, 0}
};
}

void InitEGL(NWindow* window)
{
    setenv("EGL_LOG_LEVEL", "debug", 1);
    setenv("MESA_VERBOSE", "all", 1);
    setenv("NOUVEAU_MESA_DEBUG", "1", 1);

    eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    
    eglInitialize(eglDisplay, NULL, NULL);

    eglBindAPI(EGL_OPENGL_API);

    EGLConfig config;
    EGLint numConfigs;
    static const EGLint framebufferAttributeList[] =
    {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE,     8,
        EGL_GREEN_SIZE,   8,
        EGL_BLUE_SIZE,    8,
        EGL_ALPHA_SIZE,   8,
        EGL_DEPTH_SIZE,   24,
        EGL_STENCIL_SIZE, 8,
        EGL_NONE
    };
    eglChooseConfig(eglDisplay, framebufferAttributeList, &config, 1, &numConfigs);

    eglSurface = eglCreateWindowSurface(eglDisplay, config, window, NULL);

    static const EGLint contextAttributeList[] =
    {
        EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR,
        EGL_CONTEXT_MAJOR_VERSION_KHR, 4,
        EGL_CONTEXT_MINOR_VERSION_KHR, 3,
        EGL_NONE
    };
    eglCtx = eglCreateContext(eglDisplay, config, EGL_NO_CONTEXT, contextAttributeList);

    eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglCtx);
}

void DeInitEGL()
{
    eglMakeCurrent(eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(eglDisplay, eglCtx);
    eglDestroySurface(eglDisplay, eglSurface);
    eglTerminate(eglDisplay);
}

struct Vertex
{
    float position[2];
    float uv[2];
};

void applyOverclock(bool usePCV, ClkrstSession* session, int setting)
{
    const int clockSpeeds[] = { 1020000000, 1224000000, 1581000000, 1785000000 };
    if (usePCV)
        pcvSetClockRate(PcvModule_CpuBus, clockSpeeds[setting]);
    else
        clkrstSetClockRate(session, clockSpeeds[setting]);
}

float topX, topY, topWidth, topHeight, botX, botY, botWidth, botHeight;

void updateScreenLayout(GLint vbo)
{
    int gapSizes[] = { 0, 1, 8, 64, 90, 128 };
    float gap = gapSizes[Config::ScreenGap];

    if (Config::ScreenLayout == 0) // Natural, choose based on rotation
        Config::ScreenLayout = (Config::ScreenRotation % 2 == 0) ? 1 : 2;

    if (Config::ScreenLayout == 1) // Vertical
    {
        if (Config::ScreenSizing == 0) // Even
        {
            topHeight = botHeight = 360 - gap / 2;
            if (Config::ScreenRotation % 2 == 0)
                topWidth = botWidth = topHeight * 4 / 3;
            else
                topWidth = botWidth = topHeight * 3 / 4;
        }
        else if (Config::ScreenSizing == 1) // Emphasize top
        {
            if (Config::ScreenRotation % 2 == 0) // 0, 180
            {
                botWidth = 256;
                botHeight = 192;
                topHeight = 720 - botHeight - gap;
                topWidth = topHeight * 4 / 3;
            }
            else // 90, 270
            {
                botWidth = 192;
                botHeight = 256;
                topHeight = 720 - botHeight - gap;
                topWidth = topHeight * 3 / 4;
            }
        }
        else // Emphasize bottom
        {
            if (Config::ScreenRotation % 2 == 0) // 0, 180
            {
                topWidth = 256;
                topHeight = 192;
                botHeight = 720 - topHeight - gap;
                botWidth = botHeight * 4 / 3;
            }
            else // 90, 270
            {
                botWidth = 192;
                botHeight = 256;
                topHeight = 720 - botHeight - gap;
                topWidth = topHeight * 3 / 4;
            }
        }

        topX = 640 - topWidth / 2;
        botX = 640 - botWidth / 2;
        topY = 0;
        botY = 720 - botHeight;
    }
    else // Horizontal
    {
        if (Config::ScreenRotation % 2 == 0) // 0, 180
        {
            topWidth = botWidth = 640 - gap / 2;
            topHeight = botHeight = topWidth * 3 / 4;
            topX = 0;
            botX = 1280 - topWidth;
        }
        else // 90, 270
        {
            topHeight = botHeight = 720;
            topWidth = botWidth = topHeight * 3 / 4;
            topX = 640 - topWidth - gap / 2;
            botX = 640 + gap / 2;
        }

        topY = botY = 360 - topHeight / 2;

        if (Config::ScreenSizing == 1) // Emphasize top
        {
            if (Config::ScreenRotation % 2 == 0) // 0, 180
            {
                botWidth = 256;
                botHeight = 192;
                topWidth = 1280 - botWidth - gap;
                if (topWidth > 960)
                    topWidth = 960;
                topHeight = topWidth * 3 / 4;
                topX = 640 - (botWidth + topWidth + gap) / 2;
                botX = topX + topWidth + gap;
                topY = 360 - topHeight / 2;
                botY = topY + topHeight - botHeight;
            }
            else // 90, 270
            {
                botWidth = 192;
                botHeight = 256;
                topX += (topWidth - botWidth) / 2;
                botX += (topWidth - botWidth) / 2;
                botY = 720 - botHeight;
            }
        }
        else if (Config::ScreenSizing == 2) // Emphasize bottom
        {
            if (Config::ScreenRotation % 2 == 0) // 0, 180
            {
                topWidth = 256;
                topHeight = 192;
                botWidth = 1280 - topWidth - gap;
                if (botWidth > 960)
                    botWidth = 960;
                botHeight = botWidth * 3 / 4;
                topX = 640 - (botWidth + topWidth + gap) / 2;
                botX = topX + topWidth + gap;
                botY = 360 - botHeight / 2;
                topY = botY + botHeight - topHeight;
            }
            else // 90, 270
            {
                topWidth = 192;
                topHeight = 256;
                topX += (botWidth - topWidth) / 2;
                botX -= (botWidth - topWidth) / 2;
                topY = 720 - topHeight;
            }
        }
    }

    // Swap the top and bottom screens for 90 and 180 degrees
    if (Config::ScreenRotation == 1 || Config::ScreenRotation == 2)
    {
        std::swap(topX, botX);
        std::swap(topY, botY);
        std::swap(topWidth, botWidth);
        std::swap(topHeight, botHeight);
    }

    float scwidth, scheight;

    float x0, y0, x1, y1;
    float s0, s1, s2, s3;
    float t0, t1, t2, t3;

    Vertex GL_ScreenVertices[6 * 2];

#define SETVERTEX(i, x, y, s, t) \
    GL_ScreenVertices[i].position[0] = x; \
    GL_ScreenVertices[i].position[1] = y; \
    GL_ScreenVertices[i].uv[0] = s; \
    GL_ScreenVertices[i].uv[1] = t;

    x0 = topX;
    y0 = topY;
    x1 = topX + topWidth;
    y1 = topY + topHeight;

    scwidth = 256;
    scheight = 192;

    switch (Config::ScreenRotation)
    {
        case 0:
            s0 = 0; t0 = 0;
            s1 = scwidth; t1 = 0;
            s2 = 0; t2 = scheight;
            s3 = scwidth; t3 = scheight;
            break;

        case 1:
            s0 = 0; t0 = scheight;
            s1 = 0; t1 = 0;
            s2 = scwidth; t2 = scheight;
            s3 = scwidth; t3 = 0;
            break;

        case 2:
            s0 = scwidth; t0 = scheight;
            s1 = 0; t1 = scheight;
            s2 = scwidth; t2 = 0;
            s3 = 0; t3 = 0;
            break;

        default:
            s0 = scwidth; t0 = 0;
            s1 = scwidth; t1 = scheight;
            s2 = 0; t2 = 0;
            s3 = 0; t3 = scheight;
            break;
    }


    SETVERTEX(0, x0, y0, s0, t0);
    SETVERTEX(1, x1, y1, s3, t3);
    SETVERTEX(2, x1, y0, s1, t1);
    SETVERTEX(3, x0, y0, s0, t0);
    SETVERTEX(4, x0, y1, s2, t2);
    SETVERTEX(5, x1, y1, s3, t3);

    x0 = botX;
    y0 = botY;
    x1 = botX + botWidth;
    y1 = botY + botHeight;

    scwidth = 256;
    scheight = 192;

    switch (Config::ScreenRotation)
    {
        case 0:
            s0 = 0; t0 = 192;
            s1 = scwidth; t1 = 192;
            s2 = 0; t2 = 192 + scheight;
            s3 = scwidth; t3 = 192 + scheight;
            break;

        case 1:
            s0 = 0; t0 = 192 + scheight;
            s1 = 0; t1 = 192;
            s2 = scwidth; t2 = 192 + scheight;
            s3 = scwidth; t3 = 192;
            break;

        case 2:
            s0 = scwidth; t0 = 192 + scheight;
            s1 = 0; t1 = 192 + scheight;
            s2 = scwidth; t2 = 192;
            s3 = 0; t3 = 192;
            break;

        default:
            s0 = scwidth; t0 = 192;
            s1 = scwidth; t1 = 192 + scheight;
            s2 = 0; t2 = 192;
            s3 = 0; t3 = 192 + scheight;
            break;
    }

    SETVERTEX(6, x0, y0, s0, t0);
    SETVERTEX(7, x1, y1, s3, t3);
    SETVERTEX(8, x1, y0, s1, t1);
    SETVERTEX(9, x0, y0, s0, t0);
    SETVERTEX(10, x0, y1, s2, t2);
    SETVERTEX(11, x1, y1, s3, t3);

#undef SETVERTEX

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * 12, GL_ScreenVertices, GL_STATIC_DRAW);
}

const char* vtxShader = 
    "#version 330 core\n"
    "layout (location=0) in vec2 in_position;\n"
    "layout (location=1) in vec2 in_uv;\n"
    "out vec2 out_uv;\n"
    "uniform mat4 proj;\n"
    "uniform float uvshift;\n"
    "void main() {\n"
    "   gl_Position = proj * vec4(in_position, 0.0, 1.0);\n"
    "   out_uv = in_uv / vec2(256.0, 192.0*2.0);\n"
    "}";
const char* frgShader =
    "#version 330 core\n"
    "out vec4 out_color;\n"
    "in vec2 out_uv;\n"
    "uniform sampler2D inTexture;"
    "void main() {\n"
    "   out_color = vec4(texture(inTexture, out_uv).bgr, 1.0);\n"
    "}";

#define xv_zero_array(p,n) xv_zero_size(p, (n) * (int)sizeof((p)[0]))
void xv_zero_size(void *ptr, int size)
{
    memset(ptr, 0, size);
}
void
xm4_orthographic(float *m, float left, float right, float bottom, float top,
    float near, float far)
{
    #define M(col, row) m[(col<<2)+row]
    xv_zero_array(m, 16);
    M(0,0) = 2.0f/(right-left);
    M(1,1) = 2.0f/(top-bottom);
    M(2,2) = -2.0f/(far - near);
    M(3,0) = -(right+left)/(right-left);
    M(3,1) = -(top+bottom)/(top-bottom);
    M(3,2) = -(far+near)/(far-near);
    M(3,3) = 1.0f;
    #undef M
}

#define MMX_MEMCPY memcpy
void
xm4_mul(float *product, const float *m1, const float *m2)
{
    int i;
    float a[16], b[16], o[16];
    #define A(col, row) a[(col << 2)+row]
    #define B(col, row) b[(col << 2)+row]
    #define P(col, row) o[(col << 2)+row]

    /* load */
    MMX_MEMCPY(a, m1, sizeof(a));
    MMX_MEMCPY(b, m2, sizeof(b));

    /* calculate */
    for (i = 0; i < 4; ++i) {
        const float ai0 = A(i,0), ai1 = A(i,1), ai2 = A(i,2), ai3 = A(i,3);
        P(i,0) = ai0 * B(0,0) + ai1 * B(1,0) + ai2 * B(2,0) + ai3 * B(3,0);
        P(i,1) = ai0 * B(0,1) + ai1 * B(1,1) + ai2 * B(2,1) + ai3 * B(3,1);
        P(i,2) = ai0 * B(0,2) + ai1 * B(1,2) + ai2 * B(2,2) + ai3 * B(3,2);
        P(i,3) = ai0 * B(0,3) + ai1 * B(1,3) + ai2 * B(2,3) + ai3 * B(3,3);
    }

    /* store */
    MMX_MEMCPY(product, o, sizeof(o));
    #undef A
    #undef B
    #undef P
}

void
xm4_scalev(float *m, float x, float y, float z)
{
    #define M(col, row) m[(col<<2)+row]
    xv_zero_array(m, 16);
    M(0,0) = x;
    M(1,1) = y;
    M(2,2) = z;
    M(3,3) = 1.0f;
    #undef M
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
        printf("audren init failed! %d\n", code);

    if (!R_SUCCEEDED(code = audrvCreate(&audDrv, &arConfig, 2)))
        printf("audrv create failed! %d\n", code);

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

const int clockSpeeds[] = { 1020000000, 1224000000, 1581000000, 1785000000 };

int main(int argc, char* argv[])
{
    setenv("MESA_NO_ERROR", "1", 1);

    /*setenv("EGL_LOG_LEVEL", "debug", 1);
    setenv("MESA_VERBOSE", "all", 1);
    setenv("NOUVEAU_MESA_DEBUG", "1", 1);

    setenv("NV50_PROG_OPTIMIZE", "0", 1);
    setenv("NV50_PROG_DEBUG", "1", 1);
    setenv("NV50_PROG_CHIPSET", "0x120", 1);*/

#ifdef GDB_ENABLED
    socketInitializeDefault();
    int nxlinkSocket = nxlinkStdio();
    //GDBStub_Init();
    //GDBStub_Breakpoint();
#endif

    InitEGL(nwindowGetDefault());

    gladLoadGL();

    AppletHookCookie aptCookie;
    appletLockExit();
    appletHook(&aptCookie, onAppletHook, NULL);

    Config::Load();

    loadMicSample();

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

    ImGui::CreateContext();
    ImGui::StyleColorsClassic();

    ImGuiStyle& style = ImGui::GetStyle();
    style.TouchExtraPadding = ImVec2(4, 4);
    style.ScaleAllSizes(2.f);

    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = 1.5f;

    ImGui_ImplOpenGL3_Init();

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint vtxBuffer;
    glGenBuffers(1, &vtxBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vtxBuffer);
    updateScreenLayout(vtxBuffer);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(offsetof(Vertex, position)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(offsetof(Vertex, uv)));
    glEnableVertexAttribArray(1);

    GLuint shaders[3];
    GLint projectionUniformLoc = -1, textureUniformLoc = -1, uvShiftUniformLoc = -1;
    if (!OpenGL_BuildShaderProgram(vtxShader, frgShader, shaders, "GUI"))
        printf("ahhh shaders didn't compile!!!\n");
    OpenGL_LinkShaderProgram(shaders);

    projectionUniformLoc = glGetUniformLocation(shaders[2], "proj");
    textureUniformLoc = glGetUniformLocation(shaders[2], "inTexture");

    GLuint screenTexture;
    glGenTextures(1, &screenTexture);
    glBindTexture(GL_TEXTURE_2D, screenTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, Config::Filtering ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, Config::Filtering ? GL_LINEAR : GL_NEAREST);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 256, 192 * 2);

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

    char* romNames[100];
    memset(romNames, 0, sizeof(romNames));
    char* curSelectedRom = NULL;
    char* romFullpath = NULL;
    char* romSramPath = NULL;
    int romsCount = 0;
    {
        DIR* dir = opendir("/roms/ds");
        
        int i = 0;
        struct dirent* cur;
        while (i++ < 100 && (cur = readdir(dir)))
        {
            if (cur->d_type == DT_REG)
            {
                int nameLen = strlen(cur->d_name);
                if (nameLen < 4)
                    continue;
                if (cur->d_name[nameLen - 4] != '.' 
                    || cur->d_name[nameLen - 3] != 'n' 
                    || cur->d_name[nameLen - 2] != 'd' 
                    || cur->d_name[nameLen - 1] != 's')
                    continue;
                romNames[romsCount] = new char[nameLen + 1];
                strcpy(romNames[romsCount], cur->d_name);
                romsCount++;
            }
        }

        closedir(dir);
    }

    bool lidClosed = false;
    u32 microphoneState = 0;

    while (appletMainLoop())
    {
        hidScanInput();

        u32 keysDown = hidKeysDown(CONTROLLER_P1_AUTO);
        u32 keysUp = hidKeysUp(CONTROLLER_P1_AUTO);
        u32 keysHeld = hidKeysHeld(CONTROLLER_P1_AUTO);


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
            io.DisplaySize = ImVec2(1280.f, 720.f);
            io.MouseDown[0] = false;

            if (!navInput)
            {
                for (int i = 0; i < 12; i++)
                {
                    int rotatedKeyMapping = keyMappings[i];

                    if (Config::ControlsRotation > 0 && i >= 4 && i <= 7)
                    {
                        switch (Config::ControlsRotation)
                        {
                            case 1: // 90 degrees
                                if (i == 4) rotatedKeyMapping = keyMappings[7]; // right -> down
                                else if (i == 5) rotatedKeyMapping = keyMappings[6]; // left -> up
                                else if (i == 6) rotatedKeyMapping = keyMappings[4]; // up -> right
                                else if (i == 7) rotatedKeyMapping = keyMappings[5]; // down -> left
                                break;
                            case 2: // 180 degrees
                                if (i == 4) rotatedKeyMapping = keyMappings[5]; // right -> left
                                else if (i == 5) rotatedKeyMapping = keyMappings[4]; // left -> right
                                else if (i == 6) rotatedKeyMapping = keyMappings[7]; // up -> down
                                else if (i == 7) rotatedKeyMapping = keyMappings[6]; // down -> up
                                break;
                            case 3: // 270 degrees
                                if (i == 4) rotatedKeyMapping = keyMappings[6]; // right -> up
                                else if (i == 5) rotatedKeyMapping = keyMappings[7]; // left -> down
                                else if (i == 6) rotatedKeyMapping = keyMappings[5]; // up -> left
                                else if (i == 7) rotatedKeyMapping = keyMappings[4]; // down -> right
                                break;
                        }
                    }

                    if (keysDown & rotatedKeyMapping)
                        NDS::PressKey(i > 9 ? i + 6 : i);
                    if (keysUp & rotatedKeyMapping)
                        NDS::ReleaseKey(i > 9 ? i + 6 : i);
                }

                if (keysDown & KEY_LSTICK)
                    microphoneState = 1;
                if (keysUp & KEY_LSTICK)
                    microphoneState = 0;

                feedMicAudio(microphoneState);
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

            if (hidTouchCount() > 0)
            {
                io.MouseDrawCursor = false;
                touchPosition pos;
                hidTouchRead(&pos, 0);

                if (showGui)
                {
                    io.MousePos = ImVec2((float)pos.px, (float)pos.py);
                    io.MouseDown[0] = true;
                }

                if (!io.WantCaptureMouse && pos.px >= botX && pos.px < (botX + botWidth) && pos.py >= botY && pos.py < (botY + botHeight))
                {
                    int x, y;
                    if (Config::ScreenRotation == 0) // 0
                    {
                        x = (pos.px - botX) * 256.0f / botWidth;
                        y = (pos.py - botY) * 256.0f / botWidth;
                    }
                    else if (Config::ScreenRotation == 1) // 90
                    {
                        x =       (pos.py - botY) * 192.0f / botWidth;
                        y = 192 - (pos.px - botX) * 192.0f / botWidth;
                    }
                    else if (Config::ScreenRotation == 2) // 180
                    {
                        x =       (pos.px - botX) * -256.0f / botWidth;
                        y = 192 - (pos.py - botY) *  256.0f / botWidth;
                    }
                    else // 270
                    {
                        x = (pos.py - botY) * -192.0f / botWidth;
                        y = (pos.px - botX) *  192.0f / botWidth;
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
            else
            {
                NDS::ReleaseKey(16 + 6);
                NDS::ReleaseScreen();
            }
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui::NewFrame();

        glViewport(0, 0, 1280, 720);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        paused = guiState != 1;

        if (guiState == 1)
        {
            entered = 0;
            sectionTicksTotal = 0;

            //arm9BlockFrequency.clear();
            //arm7BlockFrequency.clear();

            u64 frameStartTime = armGetSystemTick();
            NDS::RunFrame();
            u64 frameEndTime = armGetSystemTick();

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
                    break;
                
            }
            ImGui::End();
        }
        else if (guiState == 0)
        {
            if (ImGui::Begin("Select rom..."))
            {
                if (ImGui::BeginCombo("File", curSelectedRom ? curSelectedRom : "Selected a rom"))
                {
                    for (int i = 0; i < romsCount; i++)
                    {
                        ImGui::PushID(romNames[i]);
                        if (ImGui::Selectable(romNames[i], curSelectedRom == romNames[i]))
                            curSelectedRom = romNames[i];
                        ImGui::PopID();
                    }
                    ImGui::EndCombo();
                }

                if (ImGui::Button("Load!") && curSelectedRom)
                {
                    const int romPrefixLen = strlen("/roms/ds/");
                    int romNameLen = strlen(curSelectedRom);
                    int romFullpathLen = romPrefixLen + romNameLen;
                    romFullpath = new char[romFullpathLen + 1];
                    strcpy(romFullpath, "/roms/ds/");
                    strcpy(romFullpath + romPrefixLen, curSelectedRom);
                    romSramPath = new char[romFullpathLen + 4 + 1];
                    strcpy(romSramPath, romFullpath);
                    romSramPath[romFullpathLen + 0] = '.';
                    romSramPath[romFullpathLen + 1] = 's';
                    romSramPath[romFullpathLen + 2] = 'a';
                    romSramPath[romFullpathLen + 3] = 'v';
                    romSramPath[romFullpathLen + 4] = '\0';
                    NDS::LoadROM(romFullpath, romSramPath, Config::DirectBoot);

                    if (perfRecordMode == 1)
                        perfRecord = fopen("melonds_perf", "wb");
                    else if (perfRecordMode == 2)
                        perfRecord = fopen("melonds_perf", "rb");

                    guiState = 1;
                }
                
                if (ImGui::Button("Exit"))
                {
                    break;
                }

            }
            ImGui::End();

            if (ImGui::Begin("Settings"))
            {
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
                ImGui::Checkbox("Branch optimisations", &Config::JIT_BrancheOptimisations);
                ImGui::Checkbox("Literal optimisations", &Config::JIT_LiteralOptimisations);
            }
            ImGui::End();

            if (ImGui::Begin("Profiling"))
            {
                ImGui::Combo("Mode", &perfRecordMode, "No comparision\0Write frametimes\0Compare frametimes\0");
            }
            ImGui::End();

            if (ImGui::Begin("Help"))
            {
                ImGui::BulletText("Put roms into /roms/ds");
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
            OpenGL_UseShaderProgram(shaders);
            glBindVertexArray(vao);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 8);

            for (int i = 0; i < 2; i++)
            {
                glBindTexture(GL_TEXTURE_2D, screenTexture);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 192 * i, 256, 192, GL_RGBA, GL_UNSIGNED_BYTE, GPU::Framebuffer[GPU::FrontBuffer][i]);
            }
            glUniform1i(textureUniformLoc, 0);
            float proj[16];
            xm4_orthographic(proj, 0.f, 1280.f, 720.f, 0.f, 1.f, -1.f);
            glUniformMatrix4fv(projectionUniformLoc, 1, GL_FALSE, proj);
            glDrawArrays(GL_TRIANGLES, 0, 12);

            glBindVertexArray(0);

            if (showGui)
            {
                ImGui::Begin("Navigation");
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

                    int newSizing = Config::ScreenSizing;
                    ImGui::Combo("Screen Sizing", &newSizing, "Even\0Emphasise top\0Emphasise bottom\0");
                    displayDirty |= newSizing != Config::ScreenSizing;

                    int newRotation = Config::ScreenRotation;
                    const char* rotations[] = {"0", "90", "180", "270"};
                    ImGui::Combo("Screen Rotation", &newRotation, rotations, 4);
                    displayDirty |= newRotation != Config::ScreenRotation;

                    int newGap = Config::ScreenGap;
                    const char* screenGaps[] = {"0px", "1px", "8px", "64px", "90px", "128px"};
                    ImGui::Combo("Screen Gap", &newGap, screenGaps, 6);
                    displayDirty |= newGap != Config::ScreenGap;

                    int newLayout = Config::ScreenLayout;
                    ImGui::Combo("Screen Layout", &newLayout, "Natural\0Vertical\0Horizontal\0");
                    displayDirty |= newLayout != Config::ScreenLayout;

                    if (displayDirty)
                    {
                        Config::ScreenSizing = newSizing;
                        Config::ScreenRotation = newRotation;
                        Config::ScreenGap = newGap;
                        Config::ScreenLayout = newLayout;

                        updateScreenLayout(vtxBuffer);
                    }

                    bool newFiltering = Config::Filtering;
                    ImGui::Checkbox("Filtering", &newFiltering);
                    if (newFiltering != Config::Filtering)
                    {
                        glBindTexture(GL_TEXTURE_2D, screenTexture);
                        GLenum glFilter = newFiltering ? GL_LINEAR : GL_NEAREST;
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, glFilter);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, glFilter);
                        glBindTexture(GL_TEXTURE_2D, 0);
                        Config::Filtering = newFiltering;
                    }
                }
                ImGui::End();

                if (ImGui::Begin("Controller settings"))
                {
                    bool updateRequired = false;

                    int newRotation = Config::ControlsRotation;
                    const char* rotations[] = {"0", "90", "180", "270"};
                    ImGui::Combo("Controls Rotation", &newRotation, rotations, 4);
                    updateRequired |= newRotation != Config::ControlsRotation;

                    if (updateRequired)
                    {
                        Config::ControlsRotation = newRotation;
                    }
                }
                ImGui::End();

                if (ImGui::Begin("Emusettings", NULL, ImGuiWindowFlags_AlwaysAutoResize))
                {
                    if (ImGui::Checkbox("Lid closed", &lidClosed))
                        NDS::SetLidClosed(lidClosed);
                    if (ImGui::Button("Reset"))
                    {
                        NDS::LoadROM(romFullpath, romSramPath, true);

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

        ImGui::Render();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        eglSwapBuffers(eglDisplay, eglSurface);
    }

    if (perfRecord)
    {
        fclose(perfRecord);
        perfRecord = NULL;
    }

    NDS::DeInit();

    Config::Save();

    if (romSramPath)
        delete[] romSramPath;
    if (romFullpath)
        delete[] romFullpath;

    for (int i = 0; i < romsCount; i++)
        delete[] romNames[i];

    running = false;
    threadWaitForExit(&audioThread);
    threadClose(&audioThread);

    audrvClose(&audDrv);
    audrenExit();

    free(audMemPool);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();

    DeInitEGL();

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

    //close(nxlinkSocket);
#ifdef GDB_ENABLED
    socketExit();
    //GDBStub_Shutdown();
#endif

    return 0;
}
