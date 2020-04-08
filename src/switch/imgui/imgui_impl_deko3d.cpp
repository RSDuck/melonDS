#include "imgui.h"
#include "imgui_impl_deko3d.h"
#include <stdio.h>

#include <deko3d.hpp>

static dk::Device g_Device;

static ImGui_GfxAllocatorFunc g_AllocShader, g_AllocData, g_AllocTexture, g_AllocTmp;
static ImGui_GfxResetTmpAllocatorFunc g_ResetTmp;

static ImGui_GfxDataBlock g_TransformUniform, g_VertexBuffers[2], g_ImageDescriptor, g_SamplerDescriptor;
static dk::Fence g_VertexBufferFence[2];
static uint32_t g_CurVertexBuffer = 0;

static dk::Shader g_Vsh, g_Fsh;

static dk::Image g_FontTexture;

void ImGui_ImplDeko3D_CreateDeviceObjects(dk::Queue queue);

void ImGui_ImplDeko3D_Init(dk::Device device,
    dk::Queue queue,
    ImGui_GfxAllocatorFunc allocShader, 
    ImGui_GfxAllocatorFunc allocData,
    ImGui_GfxAllocatorFunc allocTexture,
    ImGui_GfxAllocatorFunc allocTmp,
    ImGui_GfxResetTmpAllocatorFunc resetTmpAlloc)
{
    ImGuiIO& io = ImGui::GetIO();
    io.BackendRendererName = "imgui_impl_deko3D";
    io.BackendFlags |= ImDrawListFlags_AllowVtxOffset;

    g_Device = device;
    
    g_AllocShader = allocShader;
    g_AllocData = allocData;
    g_AllocTexture = allocTexture;
    g_AllocTmp = allocTmp;
    g_ResetTmp = resetTmpAlloc;

    ImGui_ImplDeko3D_CreateDeviceObjects(queue);
}

struct DkshHeader
{
    uint32_t magic; // DKSH_MAGIC
    uint32_t header_sz; // sizeof(DkshHeader)
    uint32_t control_sz;
    uint32_t code_sz;
    uint32_t programs_off;
    uint32_t num_programs;
};

void ImGui_ImplDeko3D_LoadShader(const char* path, dk::Shader& out)
{
    FILE* f = fopen(path, "rb");
    if (f)
    {
        DkshHeader header;
        fread(&header, sizeof(DkshHeader), 1, f);

        rewind(f);
        void* ctrlmem = malloc(header.control_sz);
        size_t read = fread(ctrlmem, header.control_sz, 1, f);
        assert (read == header.control_sz);

        ImGui_GfxDataBlock data = g_AllocShader(header.code_sz, DK_SHADER_CODE_ALIGNMENT);
        read = fread(data.GetCpuAddr(), header.code_sz, 1, f);
        assert(read == header.code_sz);

        dk::ShaderMaker{data.mem, data.offset}
            .setControl(ctrlmem)
            .setProgramId(0)
            .initialize(out);

        free(ctrlmem);
        fclose(f);
    }
    else
    {
        printf("couldn't open %s\n", path);
        assert (false);
    }
}

void ImGui_ImplDeko3D_Shutdown()
{
}

void ImGui_ImplDeko3D_SetupRenderState(dk::CmdBuf cmdbuf)
{
    cmdbuf.bindShaders(DkStageFlag_GraphicsMask, {&g_Vsh, &g_Fsh});
    cmdbuf.bindColorWriteState(dk::ColorWriteState{});
    cmdbuf.bindRasterizerState(
        dk::RasterizerState{}.setCullMode(DkFace_None));
    cmdbuf.bindDepthStencilState(
        dk::DepthStencilState{}.setDepthTestEnable(false));
    cmdbuf.bindColorState(dk::ColorState{}.setBlendEnable(0, true));
    cmdbuf.bindBlendStates(0,
        dk::BlendState{}
            .setColorBlendOp(DkBlendOp_Add)
            .setSrcColorBlendFactor(DkBlendFactor_SrcAlpha)
            .setDstColorBlendFactor(DkBlendFactor_InvSrcAlpha));

    cmdbuf.bindImageDescriptorSet(g_ImageDescriptor.GetGpuAddr(), 4);
    cmdbuf.bindSamplerDescriptorSet(g_SamplerDescriptor.GetGpuAddr(), 4);

    cmdbuf.bindUniformBuffer(DkStage_Vertex, 0, g_TransformUniform.GetGpuAddr(), g_TransformUniform.size);

    cmdbuf.bindVtxAttribState({
        DkVtxAttribState{0, 0, offsetof(ImDrawVert, pos), DkVtxAttribSize_2x32, DkVtxAttribType_Float, 0},
        DkVtxAttribState{0, 0, offsetof(ImDrawVert, uv), DkVtxAttribSize_2x32, DkVtxAttribType_Float, 0},
        DkVtxAttribState{0, 0, offsetof(ImDrawVert, col), DkVtxAttribSize_4x8, DkVtxAttribType_Unorm, 0},
    });
    cmdbuf.bindVtxBufferState({{sizeof(ImDrawVert), 0}});
}

void ImGui_ImplDeko3D_SetTransform(dk::CmdBuf cmdbuf, ImGui_GfxTransform* transform)
{
    cmdbuf.pushConstants(g_TransformUniform.GetGpuAddr(), g_TransformUniform.size, 0, sizeof(ImGui_GfxTransform), transform);
}

void ImGui_ImplDeko3D_CreateDeviceObjects(dk::Queue queue)
{
    ImGui_ImplDeko3D_LoadShader("romfs:/shaders/imgui_vsh.dksh", g_Vsh);
    ImGui_ImplDeko3D_LoadShader("romfs:/shaders/imgui_fsh.dksh", g_Fsh);

    g_TransformUniform = g_AllocData(sizeof(ImGui_GfxTransform), DK_UNIFORM_BUF_ALIGNMENT);

    for (int i = 0; i < 2; i++)
    {
        g_VertexBuffers[i] = g_AllocData(sizeof(ImDrawVert) * 8192, alignof(ImDrawVert));
    }

    {
        dk::CmdBuf tmpcmdbuf = dk::CmdBufMaker{g_Device}.create();
        ImGui_GfxDataBlock tmpcmdbufmem = g_AllocTmp(DK_MEMBLOCK_ALIGNMENT, DK_MEMBLOCK_ALIGNMENT);
        tmpcmdbuf.addMemory(tmpcmdbufmem.mem, tmpcmdbufmem.offset, tmpcmdbufmem.size);

        ImGuiIO& io = ImGui::GetIO();
        unsigned char* pixels;
        int width, height;
        io.Fonts->GetTexDataAsAlpha8(&pixels, &width, &height);

        ImGui_GfxDataBlock stagebuffer = g_AllocTmp(width * height, DK_IMAGE_LINEAR_STRIDE_ALIGNMENT);
        memcpy(stagebuffer.GetCpuAddr(), pixels, width * height);

        dk::ImageLayout layout;
        dk::ImageLayoutMaker{g_Device}
            .setFlags(0)
            .setFormat(DkImageFormat_R8_Unorm)
            .setDimensions(width, height)
            .initialize(layout);

        ImGui_GfxDataBlock imagemem = g_AllocTexture(layout.getSize(), layout.getAlignment());
        g_FontTexture.initialize(layout, imagemem.mem, imagemem.offset);

        dk::ImageView imageView{g_FontTexture};
        imageView.setSwizzle(DkImageSwizzle_One, DkImageSwizzle_One, DkImageSwizzle_One, DkImageSwizzle_Red);
        tmpcmdbuf.copyBufferToImage({stagebuffer.GetGpuAddr()}, imageView, {0, 0, 0, (uint32_t)width, (uint32_t)(height), 1});

        queue.submitCommands(tmpcmdbuf.finishList());
        queue.waitIdle();

        tmpcmdbuf.clear();
        tmpcmdbuf.addMemory(tmpcmdbufmem.mem, tmpcmdbufmem.offset, tmpcmdbufmem.size);

        g_ImageDescriptor = g_AllocData(sizeof(DkImageDescriptor) * 4, DK_IMAGE_DESCRIPTOR_ALIGNMENT);
        g_SamplerDescriptor = g_AllocData(sizeof(DkSamplerDescriptor) * 4, DK_SAMPLER_DESCRIPTOR_ALIGNMENT);

        ImGui_ImplDeko3D_GetImageDescriptor(0)->initialize(imageView);
        ((dk::SamplerDescriptor*)g_SamplerDescriptor.GetCpuAddr())[0].initialize(dk::Sampler{}.setFilter(DkFilter_Linear, DkFilter_Linear));
        ((dk::SamplerDescriptor*)g_SamplerDescriptor.GetCpuAddr())[1].initialize(dk::Sampler{}.setFilter(DkFilter_Nearest, DkFilter_Nearest));

        tmpcmdbuf.destroy();
        g_ResetTmp();
    }
}

dk::ImageDescriptor* ImGui_ImplDeko3D_GetImageDescriptor(uint32_t i)
{
    return ((dk::ImageDescriptor*)g_ImageDescriptor.GetCpuAddr()) + i;
}

void rotate(float* out, float* in, int rotation)
{
    switch (rotation)
    {
    case 0: out[0] = in[0]; out[1] = in[1]; break;
    case 1: out[0] = 1280.f - in[1]; out[1] = in[0]; break;
    case 2: out[0] = 1280.f - in[0]; out[1] = 720.f - in[1]; break;
    case 3: out[0] = in[1]; out[1] = 720.f - in[0]; break;
    }
}

void ImGui_ImplDeko3D_RenderDrawData(ImDrawData* drawData, dk::CmdBuf cmdbuf, ImGui_GfxTransform* transform, int clipRotation)
{
    int fb_width = (int)(drawData->DisplaySize.x * drawData->FramebufferScale.x);
    int fb_height = (int)(drawData->DisplaySize.y * drawData->FramebufferScale.y);
    if (fb_width <= 0 || fb_height <= 0)
        return;

    g_VertexBufferFence[g_CurVertexBuffer].wait();
    ImGui_GfxDataBlock vtxBuffer = g_VertexBuffers[g_CurVertexBuffer];

    ImVec2 clip_off = drawData->DisplayPos;         // (0,0) unless using multi-viewports
    ImVec2 clip_scale = drawData->FramebufferScale; // (1,1) unless using retina display which are often (2,2)

    uint32_t vtxBufferOffset = 0;
    // Render command lists
    for (int n = 0; n < drawData->CmdListsCount; n++)
    {
        const ImDrawList* cmd_list = drawData->CmdLists[n];

        uint32_t vtxBufferSize = cmd_list->VtxBuffer.Size * sizeof(ImDrawVert);
        memcpy((uint8_t*)vtxBuffer.GetCpuAddr() + vtxBufferOffset, cmd_list->VtxBuffer.Data, vtxBufferSize);
        cmdbuf.bindVtxBuffer(0, vtxBuffer.GetGpuAddr() + vtxBufferOffset, vtxBufferSize);
        vtxBufferOffset += vtxBufferSize;

        uint32_t idxBufferSize = cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx);
        memcpy((uint8_t*)vtxBuffer.GetCpuAddr() + vtxBufferOffset, cmd_list->IdxBuffer.Data, idxBufferSize);
        cmdbuf.bindIdxBuffer(DkIdxFormat_Uint16, vtxBuffer.GetGpuAddr() + vtxBufferOffset);
        vtxBufferOffset += idxBufferSize;
        // Upload vertex/index buffers

        for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++)
        {
            const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];
            if (pcmd->UserCallback != NULL)
            {
                // User callback, registered via ImDrawList::AddCallback()
                // (ImDrawCallback_ResetRenderState is a special callback value used by the user to request the renderer to reset render state.)
                if (pcmd->UserCallback == ImDrawCallback_ResetRenderState)
                    ImGui_ImplDeko3D_SetupRenderState(cmdbuf);
                else
                    pcmd->UserCallback(cmd_list, pcmd);
            }
            else
            {
                // Project scissor/clipping rectangles into framebuffer space
                ImVec4 clip_rect;
                clip_rect.x = (pcmd->ClipRect.x - clip_off.x) * clip_scale.x;
                clip_rect.y = (pcmd->ClipRect.y - clip_off.y) * clip_scale.y;
                clip_rect.z = (pcmd->ClipRect.z - clip_off.x) * clip_scale.x;
                clip_rect.w = (pcmd->ClipRect.w - clip_off.y) * clip_scale.y;

                if (clip_rect.x < fb_width && clip_rect.y < fb_height && clip_rect.z >= 0.0f && clip_rect.w >= 0.0f)
                {
                    ImVec4 rotatedClipRect;
                    rotate(&rotatedClipRect[0], &clip_rect[0], clipRotation);
                    rotate(&rotatedClipRect[2], &clip_rect[2], clipRotation);

                    uint32_t x0 = std::min(rotatedClipRect.x, rotatedClipRect.z);
                    uint32_t y0 = std::min(rotatedClipRect.y, rotatedClipRect.w);
                    uint32_t x1 = std::max(rotatedClipRect.x, rotatedClipRect.z);
                    uint32_t y1 = std::max(rotatedClipRect.y, rotatedClipRect.w);
                    // Apply scissor/clipping rectangle
                    cmdbuf.setScissors(0, {{x0, y0, x1 - x0, y1 - y0}});

                    cmdbuf.bindTextures(DkStage_Fragment, 0, dkMakeTextureHandle(0, 0));
                    // Bind texture, Draw
                    //printf("drawing %d\n", pcmd->ElemCount);
                    cmdbuf.drawIndexed(DkPrimitive_Triangles, pcmd->ElemCount, 1, pcmd->IdxOffset, pcmd->VtxOffset, 0);
                }
            }
        }
    }

    cmdbuf.signalFence(g_VertexBufferFence[g_CurVertexBuffer]);
    g_CurVertexBuffer ^= 1;
}