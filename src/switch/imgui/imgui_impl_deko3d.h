#pragma once

#include <deko3d.hpp>

struct ImGui_GfxDataBlock
{
    DkGpuAddr GetGpuAddr()
    {
        return mem.getGpuAddr() + offset;
    }
    void* GetCpuAddr()
    {
        return (void*)(((uint8_t*)mem.getCpuAddr()) + offset);
    }

    dk::MemBlock mem;
    uint32_t offset, size;
};
typedef ImGui_GfxDataBlock (*ImGui_GfxAllocatorFunc)(uint32_t size, uint32_t alignment);
typedef void (*ImGui_GfxResetTmpAllocatorFunc)();

void ImGui_ImplDeko3D_Init(dk::Device device, 
    dk::Queue queue,
    ImGui_GfxAllocatorFunc allocShader, 
    ImGui_GfxAllocatorFunc allocData, 
    ImGui_GfxAllocatorFunc allocTexture,
    ImGui_GfxAllocatorFunc allocTmp,
    ImGui_GfxResetTmpAllocatorFunc resetTmpAlloc);

void ImGui_ImplDeko3D_Shutdown();

void ImGui_ImplDeko3D_RenderDrawData(ImDrawData* drawData, dk::CmdBuf cmdbuf);
