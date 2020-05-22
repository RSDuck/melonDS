#include "GPU2D.h"

#include "NDS.h"
#include "GPU.h"

#include <string.h>
#include <arm_neon.h>

#include <assert.h>

#include "switch/custom_counter.h"
#include "switch/profiler.h"

/*
    optimised GPU2D for aarch64 devices
    which are usually less powerful, but support the NEON vector instruction set

    BGOBJLine format:
        * when palette index:
            * byte 0: color index
            * byte 1: palette index
            * byte 2: bit 7: 0
        * when direct 6-bit color:
            * byte 0: red
            * byte 1: green
            * byte 2:
                * bit 0-5: blue
                * bit 6: is a bitmap sprite
                * bit 7: 1
        * byte 3 (same as regular BGOBJLine):
            * with alpha:
                * bit 0-4: alpha
                * bit 5: unused
                * bit 6: if semitransparent sprite: bitmap sprite otherwise: 3D layer
                * bit 7: semi transparent sprite
            * without alpha (same as in BlendCnt):
                * bit 0-5: BG0-BG3, OBJ, backdrop
                * bit 6-7: unused

    OBJLine format:
        * when palette index:
            * byte 0: color index
            * byte 1: palette index
        * when direct 5-bit color:
            * byte 0-1: color
        * byte 2:
            * bit 0-1: layer
            * bit 2: opaqueness
            * bit 3: apply sprite mosaic here
            * bit 4: some sprite pixel exists here
            * bit 5-6: unused
            * bit 7: format (0=palette index 1=direct color)
        * byte 3:
            * bit 0-4: bitmap sprite alpha
            * bit 5-7: source (same as BGOBJLine)
*/

#define unroll8(n, body) \
    { const int n = 0; body } \
    { const int n = 1; body } \
    { const int n = 2; body } \
    { const int n = 3; body } \
    { const int n = 4; body } \
    { const int n = 5; body } \
    { const int n = 6; body } \
    { const int n = 7; body }
#define unroll4(n, body) \
    { const int n = 0; body } \
    { const int n = 1; body } \
    { const int n = 2; body } \
    { const int n = 3; body }
#define unroll2(n, body) \
    { const int n = 0; body } \
    { const int n = 1; body }

GPU2DNeon::GPU2DNeon(u32 num)
    : GPU2DBase(num)
{
}

void GPU2DNeon::Reset()
{
    GPU2DBase::Reset();

    BGExtPalStatus = 0;
}

void GPU2DNeon::DoSavestate(Savestate* file)
{
    GPU2DBase::DoSavestate(file);
}

void GPU2DNeon::SetDisplaySettings(bool accel)
{
    // OGL renderer is unsupported in conjunction with the Neon renderer
}

void GPU2DNeon::BGExtPalDirty(u32 base)
{
    BGExtPalStatus &= ~((u64)0xFFFFFFFF << (base * 16));
}

void GPU2DNeon::OBJExtPalDirty()
{
    OBJExtPalStatus = 0;
}

inline uint8x16_t ColorBrightnessDown(uint8x16_t val, uint8x16_t factor)
{
    uint8x8_t lo = vshrn_n_u16(vmull_u8(vget_low_u8(val), vget_low_u8(factor)), 4);
    return vsubq_u8(
        val,
        vshrn_high_n_u16(lo, vmull_high_u8(val, factor), 4));
}

inline uint8x16_t ColorBrightnessUp(uint8x16_t val, uint8x16_t factor)
{
    uint8x16_t invVal = vsubq_u8(vdupq_n_u8(0x3F), val);

    uint8x8_t lo = vshrn_n_u16(vmull_u8(vget_low_u8(invVal), vget_low_u8(factor)), 4);
    return vaddq_u8(
        val,
        vshrn_high_n_u16(lo, vmull_high_u8(invVal, factor), 4));
}

inline uint8x16_t ColorBlend4(uint8x16_t src1, uint8x16_t src2, uint8x16_t eva, uint8x16_t evb)
{
    uint16x8_t lo = vaddq_u16(vmull_u8(vget_low_u8(src1), vget_low_u8(eva)),
        vmull_u8(vget_low_u8(src2), vget_low_u8(evb)));
    uint16x8_t hi = vaddq_u16(vmull_high_u8(src1, eva),
        vmull_high_u8(src2, evb));

    uint16x8_t satVal = vdupq_n_u16(0x3F0);
    lo = vminq_u16(lo, satVal);
    hi = vminq_u16(hi, satVal);

    return vshrn_high_n_u16(vshrn_n_u16(lo, 4), hi, 4);
}

inline uint8x16_t ColorBlend5(uint8x16_t src1, uint8x16_t src2, uint8x16_t alpha)
{
    uint8x16_t eva = vaddq_u8(alpha, vdupq_n_u8(1));
    uint8x16_t evb = vsubq_u8(vdupq_n_u8(32), eva);

    uint16x8_t lo = vaddq_u16(vmull_u8(vget_low_u8(src1), vget_low_u8(eva)),
        vmull_u8(vget_low_u8(src2), vget_low_u8(evb)));
    uint16x8_t hi = vaddq_u16(vmull_high_u8(src1, eva),
        vmull_high_u8(src2, evb));

    uint16x8_t satVal = vdupq_n_u16(0x7E0);
    lo = vminq_u16(lo, satVal);
    hi = vminq_u16(hi, satVal);

    return vshrn_high_n_u16(vshrn_n_u16(lo, 5), hi, 5);
}

inline void RGB5ToRGB6(uint8x16_t lo, uint8x16_t hi, uint8x16_t& red, uint8x16_t& green, uint8x16_t& blue)
{
    uint8x16_t rgb6Mask = vdupq_n_u8(0x3E);
    
    red = vandq_u8(vshlq_n_u8(lo, 1), rgb6Mask);
    green = vandq_u8(vshrn_high_n_u16(
        vshrn_n_u16(vreinterpretq_u16_u8(vzip1q_u8(lo, hi)), 4), 
        vreinterpretq_u16_u8(vzip2q_u8(lo, hi)), 4), rgb6Mask);
    blue = vandq_u8(vshrq_n_u8(hi, 1), rgb6Mask);
}


template <bool enableSpriteBlend, bool enable3DBlend, int secondSrcBlend>
void GPU2DNeon::ApplyColorEffect()
{
    PROFILER_SECTION(applyColorEffect)

    uint8x16_t blendTargets1 = vdupq_n_u8(BlendCnt);
    uint8x16_t blendTargets2 = vdupq_n_u8(BlendCnt >> 8);

    uint8x16_t cntBlendMode = vdupq_n_u8((BlendCnt >> 6) & 0x3);

    uint8x16_t vecEVY = vdupq_n_u8(EVY);
    uint8x16_t vecEVA = vdupq_n_u8(EVA);
    uint8x16_t vecEVB = vdupq_n_u8(EVB);

    for (int i = 0; i < 256; i += 32)
    {
        unroll2(j,
            uint8x16x4_t bgobjline = vld4q_u8((u8*)&BGOBJLine[8 + i + j * 16]);
            uint8x16x4_t bgobjlineBelow;
            if (secondSrcBlend > 0)
                bgobjlineBelow = vld4q_u8((u8*)&BGOBJLine[8 + 272 + i + j * 16]);

            uint8x16_t windowMask = vtstq_u8(vld1q_u8(&WindowMask[8 + i + j * 16]), vdupq_n_u8(0x20));

            uint8x16_t flag1 = bgobjline.val[3];
            uint8x16_t flag2 = bgobjlineBelow.val[3];

            uint8x16_t maskSpriteBlend1 = vtstq_u8(flag1, vdupq_n_u8(0x80));
            uint8x16_t mask3DBlend1 = vtstq_u8(flag1, vdupq_n_u8(0x40));
            if (enableSpriteBlend)
            {
                flag1 = vbslq_u8(maskSpriteBlend1, vdupq_n_u8(0x10), flag1);
                if (secondSrcBlend > 0)
                    flag2 = vbslq_u8(vtstq_u8(flag2, vdupq_n_u8(0x80)), vdupq_n_u8(0x10), flag2);
            }
            if (enable3DBlend)
            {
                flag1 = vbslq_u8(mask3DBlend1, vdupq_n_u8(0x01), flag1);
                if (secondSrcBlend > 0)
                    flag2 = vbslq_u8(vtstq_u8(flag2, vdupq_n_u8(0x40)), vdupq_n_u8(0x01), flag2);
            }

            uint8x16_t pixelsTarget1 = vtstq_u8(flag1, blendTargets1);
            uint8x16_t pixelsTarget2 = vtstq_u8(flag2, blendTargets2);

            uint8x16_t coloreffect;
            if (secondSrcBlend == 2)
                coloreffect = vbslq_u8(vandq_u8(vandq_u8(pixelsTarget1, pixelsTarget2), windowMask), cntBlendMode, vdupq_n_u8(0));
            else
                coloreffect = vbslq_u8(vandq_u8(pixelsTarget1, windowMask), cntBlendMode, vdupq_n_u8(0));

            if (enable3DBlend && secondSrcBlend > 0)
                coloreffect = vbslq_u8(vandq_u8(mask3DBlend1, pixelsTarget2), vdupq_n_u8(4), coloreffect);
            if (enableSpriteBlend && secondSrcBlend > 0)
                coloreffect = vbslq_u8(vandq_u8(maskSpriteBlend1, pixelsTarget2), vdupq_n_u8(5), coloreffect);

            uint8x16_t blendPixels = vceqq_u8(coloreffect, vdupq_n_u8(1));
            if (vmaxvq_u8(blendPixels) && secondSrcBlend == 2)
            {
                bgobjline.val[0] = vbslq_u8(blendPixels, ColorBlend4(bgobjline.val[0], bgobjlineBelow.val[0], vecEVA, vecEVB), bgobjline.val[0]);
                bgobjline.val[1] = vbslq_u8(blendPixels, ColorBlend4(bgobjline.val[1], bgobjlineBelow.val[1], vecEVA, vecEVB), bgobjline.val[1]);
                bgobjline.val[2] = vbslq_u8(blendPixels, ColorBlend4(bgobjline.val[2], bgobjlineBelow.val[2], vecEVA, vecEVB), bgobjline.val[2]);
            }
            uint8x16_t brightnessUpPixels = vceqq_u8(coloreffect, vdupq_n_u8(2));
            if (vmaxvq_u8(brightnessUpPixels))
            {
                bgobjline.val[0] = vbslq_u8(brightnessUpPixels, ColorBrightnessUp(bgobjline.val[0], vecEVY), bgobjline.val[0]);
                bgobjline.val[1] = vbslq_u8(brightnessUpPixels, ColorBrightnessUp(bgobjline.val[1], vecEVY), bgobjline.val[1]);
                bgobjline.val[2] = vbslq_u8(brightnessUpPixels, ColorBrightnessUp(bgobjline.val[2], vecEVY), bgobjline.val[2]);
            }
            uint8x16_t brightnessDownPixels = vceqq_u8(coloreffect, vdupq_n_u8(3));
            if (vmaxvq_u8(brightnessDownPixels))
            {
                bgobjline.val[0] = vbslq_u8(brightnessDownPixels, ColorBrightnessDown(bgobjline.val[0], vecEVY), bgobjline.val[0]);
                bgobjline.val[1] = vbslq_u8(brightnessDownPixels, ColorBrightnessDown(bgobjline.val[1], vecEVY), bgobjline.val[1]);
                bgobjline.val[2] = vbslq_u8(brightnessDownPixels, ColorBrightnessDown(bgobjline.val[2], vecEVY), bgobjline.val[2]);
            }
            uint8x16_t blend3DPixels = vceqq_u8(coloreffect, vdupq_n_u8(4));
            if (vmaxvq_u8(blend3DPixels) && enable3DBlend && secondSrcBlend > 0)
            {
                uint8x16_t alpha = vandq_u8(bgobjline.val[3], vdupq_n_u8(0x1F));
                bgobjline.val[0] = vbslq_u8(blend3DPixels, ColorBlend5(bgobjline.val[0], bgobjlineBelow.val[0], alpha), bgobjline.val[0]);
                bgobjline.val[1] = vbslq_u8(blend3DPixels, ColorBlend5(bgobjline.val[1], bgobjlineBelow.val[1], alpha), bgobjline.val[1]);
                bgobjline.val[2] = vbslq_u8(blend3DPixels, ColorBlend5(bgobjline.val[2], bgobjlineBelow.val[2], alpha), bgobjline.val[2]);
            }
            uint8x16_t spriteBlend = vceqq_u8(coloreffect, vdupq_n_u8(5));
            if (vmaxvq_u8(spriteBlend) && enableSpriteBlend && secondSrcBlend > 0)
            {
                uint8x16_t bitmapAlpha = vandq_u8(bgobjline.val[3], vdupq_n_u8(0x1F));
                uint8x16_t eva = vbslq_u8(mask3DBlend1, bitmapAlpha, vecEVA);
                uint8x16_t evb = vbslq_u8(mask3DBlend1, vsubq_u8(vdupq_n_u8(16), bitmapAlpha), vecEVB);

                bgobjline.val[0] = vbslq_u8(spriteBlend, ColorBlend4(bgobjline.val[0], bgobjlineBelow.val[0], eva, evb), bgobjline.val[0]);
                bgobjline.val[1] = vbslq_u8(spriteBlend, ColorBlend4(bgobjline.val[1], bgobjlineBelow.val[1], eva, evb), bgobjline.val[1]);
                bgobjline.val[2] = vbslq_u8(spriteBlend, ColorBlend4(bgobjline.val[2], bgobjlineBelow.val[2], eva, evb), bgobjline.val[2]);
            }

            vst4q_u8((u8*)&BGOBJLine[8 + j * 16 + i], bgobjline);
        )
    }

    PROFILER_END_SECTION
}

void GPU2DNeon::EnsurePaletteCoherent()
{
    {
        u64 paletteUpdates = BGExtPalUsed & ~BGExtPalStatus;
        BGExtPalStatus |= paletteUpdates;

        u8* base = GPU::Palette + (Num ? GPU::FastPalExtBOffset : GPU::FastPalExtAOffset) * 256 * 2;

        while (paletteUpdates != 0)
        {
            int idx = __builtin_ctzll(paletteUpdates);

            u16* dst = (u16*)(base + idx * 256 * 2);
            if (Num)
            {
                u32 mapping = GPU::VRAMMap_BBGExtPal[idx >> 4];
                if (mapping & (1<<7))
                    memcpy(dst, &GPU::VRAM_H[idx * 256 * 2], 256*2);
                else
                    memset(dst, 0, 256*2);
            }
            else
            {
                u32 mapping = GPU::VRAMMap_ABGExtPal[idx >> 4];
                memset(dst, 0, 256*2);
                if (mapping & (1<<4))
                    for (int i = 0; i < 256; i += 4)
                        *(u64*)&dst[i] |= *(u64*)&GPU::VRAM_E[idx * 256 * 2 + i * 2];
                if (mapping & (1<<5))
                    for (int i = 0; i < 256; i += 4)
                        *(u64*)&dst[i] |= *(u64*)&GPU::VRAM_F[(idx * 256 * 2 & 0x3FFF) + i * 2];
                if (mapping & (1<<6))
                    for (int i = 0; i < 256; i += 4)
                        *(u64*)&dst[i] |= *(u64*)&GPU::VRAM_G[(idx * 256 * 2 & 0x3FFF) + i * 2];
            }

            paletteUpdates &= ~((u64)1 << idx);
        }
    }
    {
        u16 paletteUpdates = OBJExtPalUsed & ~OBJExtPalStatus;
        OBJExtPalStatus |= paletteUpdates;

        u8* base = GPU::Palette + ((Num ? GPU::FastPalExtBOffset : GPU::FastPalExtAOffset)
            + GPU::FastPalExtBGSize) * 256 * 2;

        while (paletteUpdates != 0)
        {
            int idx = __builtin_ctz(paletteUpdates);

            u16* dst = (u16*)(base + idx * 256 * 2);
            if (Num)
            {
                if (GPU::VRAMMap_BOBJExtPal & (1<<8))
                    memcpy(dst, &GPU::VRAM_I[idx * 256 * 2], 256*2);
                else
                    memset(dst, 0, 256*2);
            }
            else
            {
                memset(dst, 0, 256*2);

                if (GPU::VRAMMap_AOBJExtPal & (1<<5))
                    for (int i = 0; i < 256; i += 4)
                        *(u64*)&dst[i] |= *(u64*)&GPU::VRAM_F[idx * 256 * 2 + i * 2];
                if (GPU::VRAMMap_AOBJExtPal & (1<<6))
                    for (int i = 0; i < 256; i += 4)
                        *(u64*)&dst[i] |= *(u64*)&GPU::VRAM_G[idx * 256 * 2 + i * 2];
            }

            paletteUpdates &= ~(1 << idx);
        }
    }
}

inline void DrawPixels(u32* bgobjline, uint8x16_t moveMask, 
    uint8x16_t newValA, uint8x16_t newValB, uint8x16_t newValC, uint8x16_t newValD)
{
    uint8x16x4_t curLayer = vld4q_u8((u8*)bgobjline);
    if (vminvq_u8(moveMask) == 0)
    {
        uint8x16x4_t prevLayer = vld4q_u8((u8*)(bgobjline + 272));

        prevLayer.val[0] = vbslq_u8(moveMask, curLayer.val[0], prevLayer.val[0]);
        prevLayer.val[1] = vbslq_u8(moveMask, curLayer.val[1], prevLayer.val[1]);
        prevLayer.val[2] = vbslq_u8(moveMask, curLayer.val[2], prevLayer.val[2]);
        prevLayer.val[3] = vbslq_u8(moveMask, curLayer.val[3], prevLayer.val[3]);

        curLayer.val[0] = vbslq_u8(moveMask, newValA, curLayer.val[0]);
        curLayer.val[1] = vbslq_u8(moveMask, newValB, curLayer.val[1]);
        curLayer.val[2] = vbslq_u8(moveMask, newValC, curLayer.val[2]);
        curLayer.val[3] = vbslq_u8(moveMask, newValD, curLayer.val[3]);

        vst4q_u8((u8*)bgobjline, curLayer);
        vst4q_u8((u8*)(bgobjline + 272), prevLayer);
    }
    else
    {
        uint8x16x4_t newVal = {newValA, newValB, newValC, newValD};
        vst4q_u8((u8*)bgobjline, newVal);
        vst4q_u8((u8*)(bgobjline + 272), curLayer);
    }
}

void GPU2DNeon::DrawScanline(u32 line)
{
    u32* dst = &Framebuffer[256 * line];

    int n3dline = line;
    line = GPU::VCount;

    bool forceblank = false;

    // scanlines that end up outside of the GPU drawing range
    // (as a result of writing to VCount) are filled white
    if (line > 192) forceblank = true;

    // GPU B can be completely disabled by POWCNT1
    // oddly that's not the case for GPU A
    if (Num && !Enabled) forceblank = true;

    if (forceblank)
    {
        memset(dst, 0xFF, 256*4);
        return;
    }

    PROFILER_SECTION(drawscanline)

    u32 dispmode = DispCnt >> 16;
    dispmode &= (Num ? 0x1 : 0x3);

    if (Num == 0)
        _3DLine = GPU3D::GetLine(n3dline);

    bool needBGOBJLine = dispmode == 1
        || ((Num == 0) && CaptureCnt & (1 << 31) && !(CaptureCnt & (1 << 24)));

    if (needBGOBJLine)
        DrawScanline_BGOBJ(line);
    UpdateMosaicCounters(line);

    switch (dispmode)
    {
    case 0:
        {
            u128 val = 0x003F3F3FL | (0x003F3F3FL << 32);
            val |= val << 64;
            for (int i = 0; i < 256; i += 4)
                *(u128*)&dst[i] = val;
        }
        break;
    case 1:
            memcpy(dst, &BGOBJLine[8], 256*4);
        break;
    case 2:
    case 3:
        {
            u16* colors = NULL;
            if (dispmode == 2)
            {
                u32 vrambank = (DispCnt >> 18) & 0x3;
                if (GPU::VRAMMap_LCDC & (1<<vrambank))
                {
                    u16* vram = (u16*)GPU::VRAM[vrambank];
                    colors = &vram[line * 256];
                }
                else
                    memset(dst, 0, 256*4);
            }
            else
                colors = DispFIFOBuffer;

            if (colors != NULL)
            {
                for (int i = 0; i < 256; i += 64)
                {
                    unroll4(j,
                        uint8x16x2_t color = vld2q_u8((u8*)&colors[i + j * 16]);

                        uint8x16x4_t result;
                        RGB5ToRGB6(color.val[0], color.val[1], result.val[0], result.val[1], result.val[2]);

                        vst4q_u8((u8*)&dst[i + j * 16], result);
                    )
                }
            }
        };
    }
    if ((Num == 0) && (CaptureCnt & (1<<31)))
    {
        u32 capwidth, capheight;
        switch ((CaptureCnt >> 20) & 0x3)
        {
        case 0: capwidth = 128; capheight = 128; break;
        case 1: capwidth = 256; capheight = 64;  break;
        case 2: capwidth = 256; capheight = 128; break;
        case 3: capwidth = 256; capheight = 192; break;
        }

        if (line < capheight)
            DoCapture(line, capwidth);
    }

    if (dispmode != 0 && (MasterBrightness >> 14) == 1)
    {
        // up
        u32 factor = MasterBrightness & 0x1F;
        if (factor > 16) factor = 16;
        uint8x16_t factorVec = vdupq_n_u8(factor);

        for (int i = 0; i < 256; i += 16)
        {
            uint8x16x4_t colors = vld4q_u8((u8*)&dst[i]);

            uint8x16x4_t result =
            {
                vshlq_n_u8(ColorBrightnessUp(colors.val[2], factorVec), 2),
                vshlq_n_u8(ColorBrightnessUp(colors.val[1], factorVec), 2),
                vshlq_n_u8(ColorBrightnessUp(colors.val[0], factorVec), 2),
                vdupq_n_u8(0xFF)
            };

            vst4q_u8((u8*)&dst[i], result);
        }
    }
    else if (dispmode != 0 && (MasterBrightness >> 14) == 2)
    {
        // down
        u32 factor = MasterBrightness & 0x1F;
        if (factor > 16) factor = 16;
        uint8x16_t factorVec = vdupq_n_u8(factor);

        for (int i = 0; i < 256; i += 16)
        {
            uint8x16x4_t colors = vld4q_u8((u8*)&dst[i]);

            uint8x16x4_t result =
            {
                vshlq_n_u8(ColorBrightnessDown(colors.val[2], factorVec), 2),
                vshlq_n_u8(ColorBrightnessDown(colors.val[1], factorVec), 2),
                vshlq_n_u8(ColorBrightnessDown(colors.val[0], factorVec), 2),
                vdupq_n_u8(0xFF)
            };

            vst4q_u8((u8*)&dst[i], result);
        }
    }
    else
    {
        for (int i = 0; i < 256; i += 16)
        {
            uint8x16x4_t colors = vld4q_u8((u8*)&dst[i]);
            uint8x16x4_t result =
            {
                vshlq_n_u8(colors.val[2], 2),
                vshlq_n_u8(colors.val[1], 2),
                vshlq_n_u8(colors.val[0], 2),
                vdupq_n_u8(0xFF)
            };

            vst4q_u8((u8*)&dst[i], result);
        }
    }

    PROFILER_END_SECTION
}

void GPU2DNeon::PalettiseRange(u32 start)
{
    PROFILER_SECTION(palettiseRange)

    uint8x16_t colorMask = vdupq_n_u8(0x1F << 1);
    uint8x16_t bitmapBit = vdupq_n_u8(1 << 7);
    for (int i = 0; i < 256; i += 16)
    {
        uint8x16x4_t pixels = vld4q_u8((u8*)&BGOBJLine[i + start]);

        uint8x16_t paletted = vtstq_u8(pixels.val[2], bitmapBit);

        uint16x8_t indices0 = vreinterpretq_u16_u8(
            vzip1q_u8(vbicq_u8(pixels.val[0], paletted), vbicq_u8(pixels.val[1], paletted)));
        uint16x8_t indices1 = vreinterpretq_u16_u8(
            vzip2q_u8(vbicq_u8(pixels.val[0], paletted), vbicq_u8(pixels.val[1], paletted)));

        uint16x8_t colorsLo;
        unroll8(i,
            colorsLo = vld1q_lane_u16((u16*)&GPU::Palette[indices0[i] * 2], colorsLo, i);)
        uint16x8_t colorsHi;
        unroll8(i,
            colorsHi = vld1q_lane_u16((u16*)&GPU::Palette[indices1[i] * 2], colorsHi, i);)

        uint8x16_t red = vandq_u8(vshlq_n_u8(vuzp1q_u8(vreinterpretq_u8_u16(colorsLo), vreinterpretq_u8_u16(colorsHi)), 1), colorMask);
        uint8x16_t green = vandq_u8(vshrn_high_n_u16(vshrn_n_u16(colorsLo, 4), colorsHi, 4), colorMask);

        uint8x16_t upper = vuzp2q_u8(vreinterpretq_u8_u16(colorsLo), vreinterpretq_u8_u16(colorsHi));
        uint8x16_t blue = vandq_u8(vshrq_n_u8(upper, 1), colorMask);

        uint8x16x4_t result = {
            vbslq_u8(paletted, pixels.val[0], red),
            vbslq_u8(paletted, pixels.val[1], green),
            vbicq_u8(vbslq_u8(paletted, pixels.val[2], blue), vdupq_n_u8(0xC0)),
            pixels.val[3]};

        vst4q_u8((u8*)&BGOBJLine[i + start], result);
    }

    PROFILER_END_SECTION
}

void GPU2DNeon::DrawScanline_BGOBJ(u32 line)
{
    BGExtPalUsed = 0;

    PROFILER_SECTION(BGOBJ)

    if (DispCnt & (1<<7))
    {
        u128 val = 0xFFBF3F3FL | (0xFFBF3F3FL << 32);
        val |= val << 64;
        for (int i = 0; i < 256; i += 4)
            *(u128*)&BGOBJLine[i] = val;
        return;
    }

    {
        u128 backdrop = Num ? 0x200 : 0;
        backdrop |= 0x20000000;
        backdrop |= (backdrop << 32);
        backdrop |= (backdrop << 64);

        for (int i = 0; i < 256; i+=4)
            *(u128*)&BGOBJLine[i + 8] = backdrop;
    }

    if (DispCnt & 0xE000)
        CalculateWindowMask(line, &WindowMask[8], &OBJWindow[8]);
    else
        memset(WindowMask + 8, 0xFF, 256);

    switch (DispCnt & 0x7)
    {
    case 0: DrawScanlineBGMode<0>(line); break;
    case 1: DrawScanlineBGMode<1>(line); break;
    case 2: DrawScanlineBGMode<2>(line); break;
    case 3: DrawScanlineBGMode<3>(line); break;
    case 4: DrawScanlineBGMode<4>(line); break;
    case 5: DrawScanlineBGMode<5>(line); break;
    case 6: DrawScanlineBGMode6(line); break;
    case 7: DrawScanlineBGMode7(line); break;
    }

    PROFILER_END_SECTION

    EnsurePaletteCoherent();

    u32 cntBlendMode = (BlendCnt >> 6) & 0x3;
    u32 threeDEnabled = !Num && (DispCnt & (1 << 3));
    u32 blendSrc2 = 0;
    if (cntBlendMode == 1)
        blendSrc2 = 2;
    else
        blendSrc2 = !!((BlendCnt >> 8) & 0x3F);

    PalettiseRange(8);

    if (blendSrc2 == 0)
    {
        if (cntBlendMode == 0)
            return;
        if (cntBlendMode == 1 && EVA == 16 && EVB == 0)
            return;
        if (cntBlendMode >= 2 && EVY == 0)
            return;
    }
    else
        PalettiseRange(272 + 8);

    switch ((u32)SemiTransSprites + threeDEnabled * 2 + blendSrc2 * 4)
    {
    case 0: ApplyColorEffect<false, false, 0>(); return;
    case 1: ApplyColorEffect<true, false, 0>(); return;
    case 2: ApplyColorEffect<false, true, 0>(); return;
    case 3: ApplyColorEffect<true, true, 0>(); return;
    case 4: ApplyColorEffect<false, false, 1>(); return;
    case 5: ApplyColorEffect<true, false, 1>(); return;
    case 6: ApplyColorEffect<false, true, 1>(); return;
    case 7: ApplyColorEffect<true, true, 1>(); return;
    case 8: ApplyColorEffect<false, false, 2>(); return;
    case 9: ApplyColorEffect<true, false, 2>(); return;
    case 10: ApplyColorEffect<false, true, 2>(); return;
    case 11: ApplyColorEffect<true, true, 2>(); return;
    }
}


#define DoDrawBG(type, line, num) \
    { if ((BGCnt[num] & 0x0040) && (BGMosaicSize[0] > 0)) DrawBG_##type<true>(line, num); else DrawBG_##type<false>(line, num); }

#define DoDrawBG_Large(line) \
    { if ((BGCnt[2] & 0x0040) && (BGMosaicSize[0] > 0)) DrawBG_Large<true>(line); else DrawBG_Large<false>(line); }

void GPU2DNeon::DrawScanlineBGMode6(u32 line)
{
    for (int i = 3; i >= 0; i--)
    {
        if ((BGCnt[2] & 0x3) == i)
        {
            if (DispCnt & 0x0400)
            {
                DoDrawBG_Large(line)
            }
        }
        if ((BGCnt[0] & 0x3) == i)
        {
            if (DispCnt & 0x0100)
            {
                if ((!Num) && (DispCnt & 0x8))
                    DrawBG_3D();
            }
        }
        if ((DispCnt & 0x1000) && NumSprites[i])
            InterleaveSprites(0x40000 | (i<<16));
    }
}

void GPU2DNeon::DrawScanlineBGMode7(u32 line)
{
    for (int i = 3; i >= 0; i--)
    {
        if ((BGCnt[1] & 0x3) == i)
        {
            if (DispCnt & 0x0200)
            {
                DoDrawBG(Text, line, 1)
            }
        }
        if ((BGCnt[0] & 0x3) == i)
        {
            if (DispCnt & 0x0100)
            {
                if ((!Num) && (DispCnt & 0x8))
                    DrawBG_3D();
                else
                    DoDrawBG(Text, line, 0)
            }
        }
        if ((DispCnt & 0x1000) && NumSprites[i])
            InterleaveSprites(0x40000 | (i<<16));
    }
}

#define DoDrawBG(type, line, num) \
    { if ((BGCnt[num] & 0x0040) && (BGMosaicSize[0] > 0)) DrawBG_##type<true>(line, num); else DrawBG_##type<false>(line, num); }

#define DoDrawBG_Large(line) \
    { if ((BGCnt[2] & 0x0040) && (BGMosaicSize[0] > 0)) DrawBG_Large<true>(line); else DrawBG_Large<false>(line); }

template<u32 bgmode>
void GPU2DNeon::DrawScanlineBGMode(u32 line)
{
    for (int i = 3; i >= 0; i--)
    {
        if ((BGCnt[3] & 0x3) == i)
        {
            if (DispCnt & 0x0800)
            {
                if (bgmode >= 3)
                    DoDrawBG(Extended, line, 3)
                else if (bgmode >= 1)
                    DoDrawBG(Affine, line, 3)
                else
                    DoDrawBG(Text, line, 3)
            }
        }
        if ((BGCnt[2] & 0x3) == i)
        {
            if (DispCnt & 0x0400)
            {
                if (bgmode == 5)
                    DoDrawBG(Extended, line, 2)
                else if (bgmode == 4 || bgmode == 2)
                    DoDrawBG(Affine, line, 2)
                else
                    DoDrawBG(Text, line, 2)
            }
        }
        if ((BGCnt[1] & 0x3) == i)
        {
            if (DispCnt & 0x0200)
            {
                DoDrawBG(Text, line, 1)
            }
        }
        if ((BGCnt[0] & 0x3) == i)
        {
            if (DispCnt & 0x0100)
            {
                if ((!Num) && (DispCnt & 0x8))
                    DrawBG_3D();
                else
                    DoDrawBG(Text, line, 0)
            }
        }
        if ((DispCnt & 0x1000) && NumSprites[i])
            InterleaveSprites(0x4 | i);
    }
}

void GPU2DNeon::InterleaveSprites(u32 prio)
{
    uint8x16_t vecPrio = vdupq_n_u8(prio);
    for (int i = 0; i < 256; i += 32)
    {
        unroll2(j,
            uint8x16x4_t pixels = vld4q_u8((u8*)&OBJLine[8 + i + j * 16]);
            
            uint8x16_t windowMask = vtstq_u8(vld1q_u8(&WindowMask[8 + i + j * 16]), vdupq_n_u8(0x10));
            uint8x16_t moveMask = vandq_u8(windowMask, vceqq_u8(vandq_u8(pixels.val[2], vdupq_n_u8(0x7)), vecPrio));

            if (vmaxvq_u8(moveMask) == 0xFF)
            {
                uint8x16_t bitmapMask = vtstq_u8(pixels.val[2], vdupq_n_u8(0x80));

                uint8x16_t red;
                uint8x16_t green;
                uint8x16_t blue;
                RGB5ToRGB6(pixels.val[0], pixels.val[1], red, green, blue);
                blue = vorrq_u8(blue, vandq_u8(pixels.val[2], vdupq_n_u8(0x80)));

                pixels.val[0] = vbslq_u8(bitmapMask, red, pixels.val[0]);
                pixels.val[1] = vbslq_u8(bitmapMask, green, pixels.val[1]);
                pixels.val[2] = vbslq_u8(bitmapMask, blue, pixels.val[2]);

                DrawPixels(&BGOBJLine[8 + j * 16 + i], moveMask,
                    pixels.val[0], pixels.val[1], pixels.val[2], pixels.val[3]);
            }
        )
    }
}

void GPU2DNeon::DrawBG_3D()
{
    u16 xoff = BGXPos[0];
    int i = 0;
    int iend = 256;

    if (xoff & 0x100)
    {
        i = (0x100 - (xoff & 0xFF));
        xoff += i;
    }
    if ((xoff - i + iend - 1) & 0x100)
    {
        iend -= (xoff & 0xFF);
    }

    // align i
    for (; (i & 0xF) != 0 && i < iend; i++)
    {
        u32 c = _3DLine[xoff];
        xoff++;

        if ((c >> 24) == 0) continue;
        if (!(WindowMask[i + 8] & 0x01)) continue;

        BGOBJLine[i + 8 + 272] = BGOBJLine[i + 8];
        BGOBJLine[i + 8] = c | 0x40800000;
    }

    for (; i < iend; i += 16)
    {
        uint8x16x4_t c = vld4q_u8((u8*)&_3DLine[xoff]);
        xoff += 16;

        uint8x16_t windowMask = vtstq_u8(vld1q_u8(&WindowMask[i + 8]), vdupq_n_u8(0x01));
        uint8x16_t moveMask = vbicq_u8(windowMask, vceqzq_u8(c.val[3]));

        if (vmaxvq_u8(moveMask) == 0xFF)
        {
            DrawPixels(&BGOBJLine[8 + i], moveMask, 
                c.val[0], 
                c.val[1], 
                vorrq_u8(c.val[2], vdupq_n_u8(0x80)), // add bitmap flag
                vorrq_u8(c.val[3], vdupq_n_u8(0x40)));
        }
    }
}

void GPU2DNeon::DoCapture(u32 line, u32 width)
{
    u32 dstvram = (CaptureCnt >> 16) & 0x3;

    // TODO: confirm this
    // it should work like VRAM display mode, which requires VRAM to be mapped to LCDC
    if (!(GPU::VRAMMap_LCDC & (1<<dstvram)))
        return;

    u16* dst = (u16*)GPU::VRAM[dstvram];
    u32 dstaddr = (((CaptureCnt >> 18) & 0x3) << 14) + (line * width);

    u32* srcA;
    if (CaptureCnt & (1<<24))
        srcA = _3DLine;
    else
        srcA = &BGOBJLine[8];

    u16* srcB = NULL;
    u32 srcBaddr = line * 256;

    if (CaptureCnt & (1<<25))
    {
        srcB = &DispFIFOBuffer[0];
        srcBaddr = 0;
    }
    else
    {
        u32 srcvram = (DispCnt >> 18) & 0x3;
        if (GPU::VRAMMap_LCDC & (1<<srcvram))
            srcB = (u16*)GPU::VRAM[srcvram];

        if (((DispCnt >> 16) & 0x3) != 2)
            srcBaddr += ((CaptureCnt >> 26) & 0x3) << 14;
    }

    dstaddr &= 0xFFFF;
    srcBaddr &= 0xFFFF;

    PROFILER_SECTION(doCapture)

    switch ((CaptureCnt >> 29) & 0x3)
    {
    case 0: // source A
        {
            uint8x16_t rgb5Mask = vdupq_n_u8(0x3E);
            for (u32 i = 0; i < width; i += 64)
            {
                unroll4(j,
                    uint8x16x4_t color = vld4q_u8((u8*)&srcA[i + j * 16]);

                    uint8x16_t alpha = vbslq_u8(vceqzq_u8(color.val[3]), vdupq_n_u8(0), vdupq_n_u8(0x80));

                    color.val[0] = vshrq_n_u8(color.val[0], 1);
                    color.val[1] = vandq_u8(color.val[1], rgb5Mask);
                    color.val[2] = vandq_u8(color.val[2], rgb5Mask);

                    uint16x8x2_t result;
                    result.val[0] = vorrq_u16(vorrq_u16(vorrq_u16(
                            vshll_n_u8(vget_low_u8(color.val[0]), 0),
                            vshll_n_u8(vget_low_u8(color.val[1]), 4)),
                            vshlq_n_u16(vshll_n_u8(vget_low_u8(color.val[2]), 8), 1)),
                            vshll_n_u8(vget_low_u8(alpha), 8));
                    result.val[1] = vorrq_u16(vorrq_u16(vorrq_u16(
                        vshll_high_n_u8(color.val[0], 0),
                        vshll_high_n_u8(color.val[1], 4)),
                        vshlq_n_u16(vshll_high_n_u8(color.val[2], 8), 1)),
                        vshll_high_n_u8(alpha, 8));

                    vst1q_u16_x2((u16*)&dst[dstaddr], result);
                    dstaddr = (dstaddr + 16) & 0xFFFF;
                )
            }
        }
        break;

    case 1: // source B
        {
            if (srcB)
            {
                for (u32 i = 0; i < width; )
                {
                    u32 pixelsLeft = width - i;
                    u32 srcLeft = 0x10000 - srcBaddr;
                    u32 dstLeft = 0x10000 - dstaddr;

                    u32 copyAmount = std::min(std::min(pixelsLeft, dstLeft), srcLeft);
                    memcpy(&dst[dstaddr], &srcB[srcBaddr], copyAmount*2);
                    
                    srcBaddr = (srcBaddr + copyAmount) & 0xFFFF;
                    dstaddr = (dstaddr + copyAmount) & 0xFFFF;
                    i += copyAmount;
                }
            }
            else
            {
                for (u32 i = 0; i < width; )
                {
                    u32 pixelsLeft = width - i;
                    u32 dstLeft = 0x10000 - dstaddr;

                    u32 setAmount = std::min(pixelsLeft, dstLeft);
                    memset(&dst[dstaddr], 0, setAmount*2);
                    i += setAmount;
                }
            }
        }
        break;

    case 2: // sources A+B
    case 3:
        {
            u32 eva = CaptureCnt & 0x1F;
            u32 evb = (CaptureCnt >> 8) & 0x1F;

            // checkme
            if (eva > 16) eva = 16;
            if (evb > 16) evb = 16;

            uint8x16_t vecEva = vdupq_n_u8(eva);
            uint8x16_t vecEvb = vdupq_n_u8(evb);
            uint8x16_t rgb5Mask = vdupq_n_u8(0x1F);
            uint16x8_t redMask = vdupq_n_u16(0x1F);
            uint16x8_t greenMask = vdupq_n_u16(0x3E0);
            uint16x8_t blueMask = vdupq_n_u16(0x7C00);
            uint16x8_t alphaBit = vdupq_n_u16(0x8000);
            uint8x16_t evaNull = vdupq_n_u8(eva == 0 ? 0xFF : 0);
            uint8x16_t evbNull = vdupq_n_u8(evb == 0 ? 0xFF : 0);
            uint16x8_t overflowVal = vdupq_n_u16(0x1FF);
            uint8x16_t vecNull = vdupq_n_u8(0);

            if (srcB)
            {
                if (eva == 0 && evb == 0)
                    memset(dst, 0, width*4);
                else
                {
                    for (u32 i = 0; i < width; i += 32)
                    {
                        unroll2(j,
                            uint8x16x4_t inA = vld4q_u8((u8*)&srcA[i + j * 16]);
                            uint16x8x2_t inB16 = vld1q_u16_x2(&srcB[srcBaddr]);

                            uint8x16_t rA = vandq_u8(vshrq_n_u8(inA.val[0], 1), rgb5Mask);
                            uint8x16_t gA = vandq_u8(vshrq_n_u8(inA.val[1], 1), rgb5Mask);
                            uint8x16_t bA = vandq_u8(vshrq_n_u8(inA.val[2], 1), rgb5Mask);
                            uint8x16_t aA = vceqzq_u8(inA.val[3]);

                            rA = vbslq_u8(aA, vecNull, rA);
                            gA = vbslq_u8(aA, vecNull, gA);
                            bA = vbslq_u8(aA, vecNull, bA);

                            uint8x16_t rB = vandq_u8(vmovn_high_u16(vmovn_u16(inB16.val[0]), inB16.val[1]), rgb5Mask);
                            uint8x16_t gB = vandq_u8(vshrn_high_n_u16(vshrn_n_u16(inB16.val[0], 5), inB16.val[1], 5), rgb5Mask);
                            uint8x16_t bB = vandq_u8(vshrq_n_u8(vshrn_high_n_u16(vshrn_n_u16(inB16.val[0], 8), inB16.val[1], 8), 2), rgb5Mask);
                            uint8x16_t aB = vmovn_high_u16(vmovn_u16(
                                vtstq_u16(inB16.val[0], alphaBit)),
                                vtstq_u16(inB16.val[1], alphaBit));

                            rB = vbslq_u8(aB, rB, vecNull);
                            gB = vbslq_u8(aB, gB, vecNull);
                            bB = vbslq_u8(aB, bB, vecNull);

                            uint8x16_t transparent = vmvnq_u8(vandq_u8(vorrq_u8(aA, evaNull), vornq_u8(evbNull, aB)));

                            uint16x8_t rD0 = vaddq_u16(
                                vmull_u8(vget_low_u8(rA), vget_low_u8(vecEva)),
                                vmull_u8(vget_low_u8(rB), vget_low_u8(vecEvb)));
                            uint16x8_t gD0 = vaddq_u16(
                                vmull_u8(vget_low_u8(gA), vget_low_u8(vecEva)),
                                vmull_u8(vget_low_u8(gB), vget_low_u8(vecEvb)));
                            uint16x8_t bD0 = vaddq_u16(
                                vmull_u8(vget_low_u8(bA), vget_low_u8(vecEva)),
                                vmull_u8(vget_low_u8(bB), vget_low_u8(vecEvb)));
                            uint16x8_t rD1 = vaddq_u16(
                                vmull_high_u8(rA, vecEva),
                                vmull_high_u8(rB, vecEvb));
                            uint16x8_t gD1 = vaddq_u16(
                                vmull_high_u8(gA, vecEva),
                                vmull_high_u8(gB, vecEvb));
                            uint16x8_t bD1 = vaddq_u16(
                                vmull_high_u8(bA, vecEva),
                                vmull_high_u8(bB, vecEvb));

                            rD0 = vminq_u16(rD0, overflowVal);
                            gD0 = vminq_u16(gD0, overflowVal);
                            bD0 = vminq_u16(bD0, overflowVal);
                            rD1 = vminq_u16(rD1, overflowVal);
                            gD1 = vminq_u16(gD1, overflowVal);
                            bD1 = vminq_u16(bD1, overflowVal);

                            uint16x8x2_t result;
                            result.val[0] = vorrq_u16(vorrq_u16(vorrq_u16(
                                vandq_u16(vshrq_n_u16(rD0, 4), redMask),
                                vandq_u16(vshlq_n_u16(gD0, 5-4), greenMask)),
                                vandq_u16(vshlq_n_u16(bD0, 10-4), blueMask)),
                                vandq_u16(vreinterpretq_u16_u8(vzip1q_u8(transparent, transparent)), alphaBit));
                            result.val[1] = vorrq_u16(vorrq_u16(vorrq_u16(
                                vandq_u16(vshrq_n_u16(rD1, 4), redMask),
                                vandq_u16(vshlq_n_u16(gD1, 5-4), greenMask)),
                                vandq_u16(vshlq_n_u16(bD1, 10-4), blueMask)),
                                vandq_u16(vreinterpretq_u16_u8(vzip2q_u8(transparent, transparent)), alphaBit));
                        
                            vst1q_u16_x2(&dst[dstaddr], result);
                            dstaddr = (dstaddr + 16) & 0xFFFF;
                            srcBaddr = (srcBaddr + 16) & 0xFFFF;
                        )
                    }
                }
            }
            else
            {
                if (eva == 0)
                {
                    memset(dst, 0, width*4);
                }
                else
                {
                    for (u32 i = 0; i < width; i += 32)
                    {
                        unroll2(j,
                            uint8x16x4_t colors = vld4q_u8((u8*)&srcA[i + j * 16]);

                            uint8x16_t rA = vshrq_n_u8(colors.val[0], 1);
                            uint8x16_t gA = vshrq_n_u8(colors.val[1], 1);
                            uint8x16_t bA = vshrq_n_u8(colors.val[2], 1);
                            uint8x16_t aA = vceqzq_u8(colors.val[3]);

                            uint16x8x2_t result;
                            result.val[0] = vorrq_u16(vorrq_u16(vorrq_u16(
                                vandq_u16(vshrq_n_u16(vmull_u8(vget_low_u8(rA), vget_low_u8(vecEva)), 4), redMask),
                                vandq_u16(vshlq_n_u16(vmull_u8(vget_low_u8(gA), vget_low_u8(vecEva)), 5-4), greenMask)),
                                vandq_u16(vshlq_n_u16(vmull_u8(vget_low_u8(bA), vget_low_u8(vecEva)), 10-4), blueMask)),
                                alphaBit);
                            result.val[1] = vorrq_u16(vorrq_u16(vorrq_u16(
                                vandq_u16(vshrq_n_u16(vmull_high_u8(rA, vecEva), 4), redMask),
                                vandq_u16(vshlq_n_u16(vmull_high_u8(gA, vecEva), 5-4), greenMask)),
                                vandq_u16(vshlq_n_u16(vmull_high_u8(bA, vecEva), 10-4), blueMask)),
                                alphaBit);

                            result.val[0] = vbslq_u16(vreinterpretq_u16_u8(vzip1q_u8(aA, aA)), vdupq_n_u16(0), result.val[0]);
                            result.val[1] = vbslq_u16(vreinterpretq_u16_u8(vzip2q_u8(aA, aA)), vdupq_n_u16(0), result.val[1]);

                            vst1q_u16_x2(&dst[dstaddr], result);
                            dstaddr = (dstaddr + 16) & 0xFFFF;
                        )
                    }
                }
            }
        }
        break;
    }

    PROFILER_END_SECTION
}

template<bool mosaic>
void GPU2DNeon::DrawBG_Text(u32 line, u32 bgnum)
{
    u16 bgcnt = BGCnt[bgnum];

    u32 tilesetaddr, tilemapaddr;
    u32 extpal, extpalslot;

    u16 xoff = BGXPos[bgnum];
    u16 yoff = BGYPos[bgnum] + line;

    if (bgcnt & 0x0040)
    {
        // vertical mosaic
        yoff -= BGMosaicY;
    }

    u32 widexmask = (bgcnt & 0x4000) ? 0x100 : 0;

    extpal = (DispCnt & 0x40000000);
    if (extpal) extpalslot = ((bgnum<2) && (bgcnt&0x2000)) ? (2+bgnum) : bgnum;

    if (Num)
    {
        tilesetaddr = ((bgcnt & 0x003C) << 12);
        tilemapaddr = ((bgcnt & 0x1F00) << 3);
    }
    else
    {
        tilesetaddr = ((DispCnt & 0x07000000) >> 8) + ((bgcnt & 0x003C) << 12);
        tilemapaddr = ((DispCnt & 0x38000000) >> 11) + ((bgcnt & 0x1F00) << 3);
    }

    u32 width = (bgcnt & 0x4000) ? 512 : 256;
    u32 height = (bgcnt & 0x8000) ? 512 : 256;
    u8* tilemapptr = GPU::GetBGCachePtr(Num, tilemapaddr, width * height * 2);
    u8* tilesetptr = GPU::GetBGCachePtr(Num, tilesetaddr, 1024 * (bgcnt & 0x80 ? 32 : 16));

    // adjust Y position in tilemap
    if (bgcnt & 0x8000)
    {
        tilemapptr += ((yoff & 0x1F8) << 3);
        if (bgcnt & 0x4000)
            tilemapptr += ((yoff & 0x100) << 3);
    }
    else
        tilemapptr += ((yoff & 0xF8) << 3);

    u32 localxoff = 8 - (xoff & 0x7);
    xoff &= ~0x7;

    u32* dst = &BGOBJLine[localxoff];
    u8* windowMask = &WindowMask[localxoff];
    u32 xofftarget = xoff + 256 + (localxoff == 8 ? 0 : 8);

    uint8x16_t compositorFlag = vdupq_n_u8(1 << bgnum);

    if (bgcnt & 0x80)
    {
        u32 palOffset;
        if (extpal)
            palOffset = extpalslot * 16 + (Num ? GPU::FastPalExtBOffset : GPU::FastPalExtAOffset);
        else
            palOffset = Num ? 2 : 0;
        u64 extpalsUsed = 0;

        uint8x16_t extpalMask = vdupq_n_u8(extpal ? 0xFF : 0);

        if (localxoff)
        {
            u16 curtile = *(u16*)(tilemapptr + ((xoff & 0xF8) >> 2) + ((xoff & widexmask) << 3));
            xoff += 8;
            uint8x8_t curtileNeon = vdup_n_u8((curtile >> 8) | curtile & 0xFF00);
            extpalsUsed |= (1 << (curtile >> 12));

            uint8x8_t hflip = vtst_u8(curtileNeon, vdup_n_u8(1 << 2));
            uint8x8_t extpal = vshr_n_u8(curtileNeon, 4);

            uint8x8_t windowmaskbits = vld1_u8(windowMask);
            windowMask += 8;

            u64 pixels0 = *(u64*)(tilesetptr + ((curtile & 0x03FF) << 6)
                    + (((curtile & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 3));
            if (pixels0)
            {
                uint8x8_t movemask = vtst_u8(windowmaskbits, vdup_n_u8(1 << bgnum));

                uint8x8_t pixels = vreinterpret_u8_u64(vdup_n_u64(pixels0));

                pixels = vbsl_u8(hflip, vrev64_u8(pixels), pixels);
                movemask = vbic_u8(movemask, vceqz_u8(pixels));

                uint8x8x4_t resLayer = vld4_u8((u8*)dst);
                uint8x8x4_t resLayerBelow = vld4_u8((u8*)(dst + 272));
                resLayerBelow.val[0] = vbsl_u8(movemask, resLayer.val[0], resLayerBelow.val[0]);
                resLayerBelow.val[1] = vbsl_u8(movemask, resLayer.val[1], resLayerBelow.val[1]);
                resLayerBelow.val[2] = vbsl_u8(movemask, resLayer.val[2], resLayerBelow.val[2]);
                resLayerBelow.val[3] = vbsl_u8(movemask, resLayer.val[3], resLayerBelow.val[3]);

                uint8x8_t palette = vadd_u8(vdup_n_u8(palOffset), vand_u8(extpal, vget_low_u8(extpalMask)));

                resLayer.val[0] = vbsl_u8(movemask, pixels, resLayer.val[0]);
                resLayer.val[1] = vbsl_u8(movemask, palette, resLayer.val[1]);
                resLayer.val[2] = vbsl_u8(movemask, vdup_n_u8(0), resLayer.val[2]);
                resLayer.val[3] = vbsl_u8(movemask, vget_low_u8(compositorFlag), resLayer.val[3]);

                vst4_u8((u8*)dst, resLayer);
                vst4_u8((u8*)(dst + 272), resLayerBelow);
            }
            dst += 8;
        }

        for (int i = 0; i < 256; i += 32)
        {
            u16 curtile0 = *(u16*)(tilemapptr + ((xoff & 0xF8) >> 2) + ((xoff & widexmask) << 3));
            xoff += 8;
            u16 curtile1 = *(u16*)(tilemapptr + ((xoff & 0xF8) >> 2) + ((xoff & widexmask) << 3));
            xoff += 8;
            u16 curtile2 = *(u16*)(tilemapptr + ((xoff & 0xF8) >> 2) + ((xoff & widexmask) << 3));
            xoff += 8;
            u16 curtile3 = *(u16*)(tilemapptr + ((xoff & 0xF8) >> 2) + ((xoff & widexmask) << 3));
            xoff += 8;

            extpalsUsed |= (1 << (curtile0 >> 12));
            extpalsUsed |= (1 << (curtile1 >> 12));
            extpalsUsed |= (1 << (curtile2 >> 12));
            extpalsUsed |= (1 << (curtile3 >> 12));

            u64 curtiles = (u64)curtile0 | ((u64)curtile1 << 16) | ((u64)curtile2 << 32) | ((u64)curtile3 << 48);
            curtiles &= 0xFF00FF00FF00FF00;
            curtiles |= curtiles >> 8;
            uint8x16_t curtilesNeon = vreinterpretq_u8_u64(vdupq_n_u64(curtiles));
            curtilesNeon = vzip1q_u8(curtilesNeon, curtilesNeon);

            uint8x16_t hflip = vtstq_u8(curtilesNeon, vdupq_n_u8(1 << 2));
            uint8x16_t extpal = vshrq_n_u8(curtilesNeon, 4);

            uint8x16x2_t windowmaskbits = vld1q_u8_x2(windowMask);
            windowMask += 32;

            u64 pixels0 = *(u64*)(tilesetptr + ((curtile0 & 0x03FF) << 6)
                                         + (((curtile0 & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 3));
            u64 pixels1 = *(u64*)(tilesetptr + ((curtile1 & 0x03FF) << 6)
                                         + (((curtile1 & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 3));
            u64 pixels2 = *(u64*)(tilesetptr + ((curtile2 & 0x03FF) << 6)
                                         + (((curtile2 & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 3));
            u64 pixels3 = *(u64*)(tilesetptr + ((curtile3 & 0x03FF) << 6)
                                         + (((curtile3 & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 3));

            if (pixels0 || pixels1)
            {
                uint8x16_t movemask = vtstq_u8(windowmaskbits.val[0], vdupq_n_u8(1 << bgnum));

                uint64x2_t pixels64 = {pixels0, pixels1};
                uint8x16_t pixels = vreinterpretq_u8_u64(pixels64);

                pixels = vbslq_u8(vzip1q_u8(hflip, hflip), vrev64q_u8(pixels), pixels);
                movemask = vbicq_u8(movemask, vceqzq_u8(pixels));

                uint8x16_t palette = vaddq_u8(vdupq_n_u8(palOffset), vandq_u8(vzip1q_u8(extpal, extpal), extpalMask));
                DrawPixels(dst, movemask, pixels, palette, vdupq_n_u8(0), compositorFlag);
            }
            dst += 16;
            if (pixels2 || pixels3)
            {
                uint8x16_t movemask = vtstq_u8(windowmaskbits.val[1], vdupq_n_u8(1 << bgnum));

                uint64x2_t pixels64 = {pixels2, pixels3};
                uint8x16_t pixels = vreinterpretq_u8_u64(pixels64);

                pixels = vbslq_u8(vzip2q_u8(hflip, hflip), vrev64q_u8(pixels), pixels);
                movemask = vbicq_u8(movemask, vceqzq_u8(pixels));

                uint8x16_t palette = vaddq_u8(vdupq_n_u8(palOffset), vandq_u8(vzip2q_u8(extpal, extpal), extpalMask));
                DrawPixels(dst, movemask, pixels, palette, vdupq_n_u8(0), compositorFlag);
            }
            dst += 16;
        }

        if (extpal)
            BGExtPalUsed |= extpalsUsed << (extpalslot * 16);
    }
    else
    {
        if (localxoff)
        {
            u16 curtile = *(u16*)(tilemapptr + ((xoff & 0xF8) >> 2) + ((xoff & widexmask) << 3));
            xoff += 8;
            uint8x8_t curtileNeon = vdup_n_u8((curtile >> 8) | curtile & 0xFF00);

            uint8x8_t hflip = vtst_u8(curtileNeon, vdup_n_u8(1 << 2));
            uint8x8_t extpal = vshl_n_u8(vshr_n_u8(curtileNeon, 4), 4);

            uint8x8_t windowmaskbits = vld1_u8(windowMask);
            windowMask += 8;

            u64 pixels0 = *(u32*)(tilesetptr + ((curtile & 0x03FF) << 5)
                    + (((curtile & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 2));
            if (pixels0)
            {
                uint8x8_t pixels = vreinterpret_u8_u64(vdup_n_u64(pixels0));
                pixels = vzip1_u8(vshr_n_u8(vshl_n_u8(pixels, 4), 4), vshr_n_u8(pixels, 4));

                uint8x8_t movemask = vtst_u8(windowmaskbits, vdup_n_u8(1 << bgnum));

                pixels = vbsl_u8(hflip, vrev64_u8(pixels), pixels);
                movemask = vbic_u8(movemask, vceqz_u8(pixels));

                uint8x8x4_t resLayer = vld4_u8((u8*)dst);
                uint8x8x4_t resLayerBelow = vld4_u8((u8*)(dst + 272));
                resLayerBelow.val[0] = vbsl_u8(movemask, resLayer.val[0], resLayerBelow.val[0]);
                resLayerBelow.val[1] = vbsl_u8(movemask, resLayer.val[1], resLayerBelow.val[1]);
                resLayerBelow.val[2] = vbsl_u8(movemask, resLayer.val[2], resLayerBelow.val[2]);
                resLayerBelow.val[3] = vbsl_u8(movemask, resLayer.val[3], resLayerBelow.val[3]);

                resLayer.val[0] = vbsl_u8(movemask, vadd_u8(pixels, extpal), resLayer.val[0]);
                resLayer.val[1] = vbsl_u8(movemask, vdup_n_u8(Num ? 2 : 0), resLayer.val[1]);
                resLayer.val[2] = vbsl_u8(movemask, vdup_n_u8(0), resLayer.val[2]);
                resLayer.val[3] = vbsl_u8(movemask, vget_low_u8(compositorFlag), resLayer.val[3]);

                vst4_u8((u8*)dst, resLayer);
                vst4_u8((u8*)(dst + 272), resLayerBelow);
            }
            dst += 8;
        }

        for (int i = 0; i < 256; i += 32)
        {
            u16 curtile0 = *(u16*)(tilemapptr + ((xoff & 0xF8) >> 2) + ((xoff & widexmask) << 3));
            xoff += 8;
            u16 curtile1 = *(u16*)(tilemapptr + ((xoff & 0xF8) >> 2) + ((xoff & widexmask) << 3));
            xoff += 8;
            u16 curtile2 = *(u16*)(tilemapptr + ((xoff & 0xF8) >> 2) + ((xoff & widexmask) << 3));
            xoff += 8;
            u16 curtile3 = *(u16*)(tilemapptr + ((xoff & 0xF8) >> 2) + ((xoff & widexmask) << 3));
            xoff += 8;

            u64 curtiles = (u64)curtile0 | ((u64)curtile1 << 16) | ((u64)curtile2 << 32) | ((u64)curtile3 << 48);
            curtiles &= 0xFF00FF00FF00FF00;
            curtiles |= curtiles >> 8;
            uint8x16_t curtilesNeon = vreinterpretq_u8_u64(vdupq_n_u64(curtiles));
            curtilesNeon = vzip1q_u8(curtilesNeon, curtilesNeon);

            uint8x16_t hflip = vtstq_u8(curtilesNeon, vdupq_n_u8(1 << 2));
            uint8x16_t pal = vshlq_n_u8(vshrq_n_u8(curtilesNeon, 4), 4);

            uint8x16x2_t windowmaskbits = vld1q_u8_x2(windowMask);
            windowMask += 32;

            u32 pixels0 = *(u32*)(tilesetptr + ((curtile0 & 0x03FF) << 5)
                                         + (((curtile0 & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 2));
            u32 pixels1 = *(u32*)(tilesetptr + ((curtile1 & 0x03FF) << 5)
                                         + (((curtile1 & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 2));
            u32 pixels2 = *(u32*)(tilesetptr + ((curtile2 & 0x03FF) << 5)
                                         + (((curtile2 & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 2));
            u32 pixels3 = *(u32*)(tilesetptr + ((curtile3 & 0x03FF) << 5)
                                         + (((curtile3 & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 2));

            uint32x4_t allPixels = {pixels0, pixels1, pixels2, pixels3};
            uint8x16_t pixelsHi = vshrq_n_u8(vreinterpretq_u8_u32(allPixels), 4);
            uint8x16_t pixelsLo = vshrq_n_u8(vshlq_n_u8(vreinterpretq_u8_u32(allPixels), 4), 4);

            if (pixels0 || pixels1)
            {
                uint8x16_t movemask = vtstq_u8(windowmaskbits.val[0], vdupq_n_u8(1 << bgnum));

                uint8x16_t pixels = vzip1q_u8(pixelsLo, pixelsHi);

                pixels = vbslq_u8(vzip1q_u8(hflip, hflip), vrev64q_u8(pixels), pixels);
                movemask = vbicq_u8(movemask, vceqzq_u8(pixels));

                pixels = vaddq_u8(pixels, vzip1q_u8(pal, pal));

                DrawPixels(dst, movemask, pixels, vdupq_n_u8(Num ? 2 : 0), vdupq_n_u8(0), compositorFlag);
            }
            dst += 16;
            if (pixels2 || pixels3)
            {
                uint8x16_t movemask = vtstq_u8(windowmaskbits.val[1], vdupq_n_u8(1 << bgnum));

                uint8x16_t pixels = vzip2q_u8(pixelsLo, pixelsHi);

                pixels = vbslq_u8(vzip2q_u8(hflip, hflip), vrev64q_u8(pixels), pixels);
                movemask = vbicq_u8(movemask, vceqzq_u8(pixels));

                pixels = vaddq_u8(pixels, vzip2q_u8(pal, pal));

                DrawPixels(dst, movemask, pixels, vdupq_n_u8(Num ? 2 : 0), vdupq_n_u8(0), compositorFlag);
            }
            dst += 16;
        }
    }
}

template<bool mosaic>
void GPU2DNeon::DrawBG_Affine(u32 line, u32 bgnum)
{
    u16 bgcnt = BGCnt[bgnum];

    u32 tilesetaddr, tilemapaddr;

    u32 coordmask;
    u32 yshift;
    u32 size;
    switch (bgcnt & 0xC000)
    {
    case 0x0000: coordmask = 0x07800; yshift = 7; size = 128*128; break;
    case 0x4000: coordmask = 0x0F800; yshift = 8; size = 256*256; break;
    case 0x8000: coordmask = 0x1F800; yshift = 9; size = 512*512; break;
    case 0xC000: coordmask = 0x3F800; yshift = 10; size = 1024*1024; break;
    }
    yshift -= 3;

    u32 overflowmask;
    if (bgcnt & 0x2000) overflowmask = 0;
    else                overflowmask = ~(coordmask | 0x7FF);

    s16 rotA = BGRotA[bgnum-2];
    s16 rotB = BGRotB[bgnum-2];
    s16 rotC = BGRotC[bgnum-2];
    s16 rotD = BGRotD[bgnum-2];

    s32 rotX = BGXRefInternal[bgnum-2];
    s32 rotY = BGYRefInternal[bgnum-2];

    if (bgcnt & 0x0040)
    {
        // vertical mosaic
        rotX -= (BGMosaicY * rotB);
        rotY -= (BGMosaicY * rotD);
    }

    if (Num)
    {
        tilesetaddr = ((bgcnt & 0x003C) << 12);
        tilemapaddr = ((bgcnt & 0x1F00) << 3);
    }
    else
    {
        tilesetaddr = ((DispCnt & 0x07000000) >> 8) + ((bgcnt & 0x003C) << 12);
        tilemapaddr = ((DispCnt & 0x38000000) >> 11) + ((bgcnt & 0x1F00) << 3);
    }
    u8* tilemapptr = GPU::GetBGCachePtr(Num, tilemapaddr, size);
    u8* tilesetptr = GPU::GetBGCachePtr(Num, tilesetaddr, 256 * 64);
    
    int32x4_t dx = vshlq_n_s32(vdupq_n_s32(rotA), 2);
    int32x4_t dy = vshlq_n_s32(vdupq_n_s32(rotC), 2);

    const int32x4_t factorDist = {0, 1, 2, 3};
    int32x4_t vecRotX = vaddq_s32(vdupq_n_s32(rotX), vmulq_s32(vdupq_n_s32(rotA), factorDist));
    int32x4_t vecRotY = vaddq_s32(vdupq_n_s32(rotY), vmulq_s32(vdupq_n_s32(rotC), factorDist));

    int32x4_t vecCoordmask = vdupq_n_s32(coordmask);
    int32x4_t vecYShift = vdupq_n_s32(yshift);
    int32x4_t vecOverflowMask = vdupq_n_s32(overflowmask);

    int32x4_t tileMask = vdupq_n_s32(0x7);

    for (int i = 0; i < 256; i += 16)
    {
        int32x4x4_t tileoff;

        uint8x16_t moveMask;

        unroll4(j,
            int32x4_t offset = vaddq_s32(
                vshlq_s32(vshrq_n_s32(vandq_s32(vecRotY, vecCoordmask), 11), vecYShift),
                vshrq_n_s32(vandq_s32(vecRotX, vecCoordmask), 11));

            // this is terrible
            uint8x16_t overflow = vreinterpretq_u8_u32(vtstq_u32(
                vreinterpretq_u32_s32(vorrq_s32(vecRotX, vecRotY)), 
                vreinterpretq_u32_s32(vecOverflowMask)));
            moveMask = vreinterpretq_u8_u32(vsetq_lane_u32(
                vreinterpretq_u32_u8(vuzp1q_u8(vuzp1q_u8(overflow, overflow), vuzp1q_u8(overflow, overflow)))[0],
                vreinterpretq_u32_u8(moveMask), j));

            u32 tiles = 0;
            for (int k = 0; k < 4; k++)
                tiles |= *(tilemapptr + offset[k]) << k * 8;

            tileoff.val[j] = vaddq_s32(vaddq_s32(
                vandq_s32(vshrq_n_s32(vecRotX, 8), tileMask),
                vshlq_n_s32(vandq_s32(vshrq_n_s32(vecRotY, 8), tileMask), 3)),
                vreinterpretq_s32_u32(
                    vshll_n_u16(vget_low_u16(vshll_n_u8(vreinterpret_u8_u64(vdup_n_u64(tiles)), 6)), 0)));

            vecRotX = vaddq_s32(vecRotX, dx);
            vecRotY = vaddq_s32(vecRotY, dy);
        )

        uint8x16_t pixels;
        unroll4(j,
            unroll4(k, pixels = vld1q_lane_u8(tilesetptr + tileoff.val[j][k], pixels, j * 4 + k);))

        uint8x16_t windowMask = vtstq_u8(vld1q_u8(&WindowMask[8 + i]), vdupq_n_u8(1 << bgnum));

        moveMask = vbicq_u8(vbicq_u8(windowMask, vceqzq_u8(pixels)), moveMask);

        DrawPixels(&BGOBJLine[8 + i], moveMask, pixels, vdupq_n_u8(Num ? 2 : 0), vdupq_n_u8(0), vdupq_n_u8(1 << bgnum));
    }

    BGXRefInternal[bgnum-2] += rotB;
    BGYRefInternal[bgnum-2] += rotD;
}

template<bool mosaic>
void GPU2DNeon::DrawBG_Extended(u32 line, u32 bgnum)
{
    u16 bgcnt = BGCnt[bgnum];

    u32 tilesetaddr, tilemapaddr;
    u32 extpal;

    extpal = (DispCnt & 0x40000000);

    s16 rotA = BGRotA[bgnum-2];
    s16 rotB = BGRotB[bgnum-2];
    s16 rotC = BGRotC[bgnum-2];
    s16 rotD = BGRotD[bgnum-2];

    s32 rotX = BGXRefInternal[bgnum-2];
    s32 rotY = BGYRefInternal[bgnum-2];

    if (bgcnt & 0x0040)
    {
        // vertical mosaic
        rotX -= (BGMosaicY * rotB);
        rotY -= (BGMosaicY * rotD);
    }

    int32x4_t dx = vshlq_n_s32(vdupq_n_s32(rotA), 2);
    int32x4_t dy = vshlq_n_s32(vdupq_n_s32(rotC), 2);

    const int32x4_t factorDist = {0, 1, 2, 3};
    int32x4_t vecRotX = vaddq_s32(vdupq_n_s32(rotX), vmulq_s32(vdupq_n_s32(rotA), factorDist));
    int32x4_t vecRotY = vaddq_s32(vdupq_n_s32(rotY), vmulq_s32(vdupq_n_s32(rotC), factorDist));

    if (bgcnt & 0x0080)
    {
        // bitmap modes

        u32 xmask, ymask;
        u32 yshift;
        u32 size;
        switch (bgcnt & 0xC000)
        {
        case 0x0000: xmask = 0x07FFF; ymask = 0x07FFF; yshift = 7; size = 128*128; break;
        case 0x4000: xmask = 0x0FFFF; ymask = 0x0FFFF; yshift = 8; size = 256*256; break;
        case 0x8000: xmask = 0x1FFFF; ymask = 0x0FFFF; yshift = 9; size = 512*256; break;
        case 0xC000: xmask = 0x1FFFF; ymask = 0x1FFFF; yshift = 9; size = 512*512; break;
        }

        u32 ofxmask, ofymask;
        if (bgcnt & 0x2000)
        {
            ofxmask = 0;
            ofymask = 0;
        }
        else
        {
            ofxmask = ~xmask;
            ofymask = ~ymask;
        }

        if (Num) tilemapaddr = ((bgcnt & 0x1F00) << 6);
        else     tilemapaddr = ((bgcnt & 0x1F00) << 6);

        int32x4_t vecXMask = vdupq_n_s32(xmask);
        int32x4_t vecYMask = vdupq_n_s32(ymask);
        int32x4_t vecOfxMask = vdupq_n_s32(ofxmask);
        int32x4_t vecOfyMask = vdupq_n_s32(ofymask);
        int32x4_t vecYShift = vdupq_n_s32(yshift);

        if (bgcnt & 0x0004)
        {
            // direct color bitmap
            u8* tilemapptr = GPU::GetBGCachePtr(Num, tilemapaddr, size * 2);

            for (int i = 0; i < 256; i += 16)
            {
                uint8x16x2_t colors;
                uint8x16_t moveMask;

                unroll4(j,
                    uint8x16_t overflow = vreinterpretq_u8_u32(vorrq_u32(
                        vtstq_u32(vreinterpretq_u32_s32(vecRotX), vreinterpretq_u32_s32(vecOfxMask)),
                        vtstq_u32(vreinterpretq_u32_s32(vecRotY), vreinterpretq_u32_s32(vecOfyMask))));
                    moveMask = vreinterpretq_u8_u32(vsetq_lane_u32(
                        vreinterpretq_u32_u8(vuzp1q_u8(vuzp1q_u8(overflow, overflow), vuzp1q_u8(overflow, overflow)))[0],
                        vreinterpretq_u32_u8(moveMask), j));

                    int32x4_t offset = vshlq_n_s32(
                        vaddq_s32(
                            vshlq_s32(vshrq_n_s32(vandq_s32(vecRotY, vecYMask), 8), vecYShift), 
                            vshrq_n_s32(vandq_s32(vecRotX, vecXMask), 8)),
                        1);

                    unroll4(k,
                        colors = vld2q_lane_u8(tilemapptr + offset[k], colors, j * 4 + k);)

                    vecRotX = vaddq_s32(vecRotX, dx);
                    vecRotY = vaddq_s32(vecRotY, dy);
                )

                uint8x16_t windowMask = vtstq_u8(vld1q_u8(&WindowMask[8 + i]), vdupq_n_u8(1 << bgnum));
                moveMask = vbicq_u8(vandq_u8(windowMask, vtstq_u8(colors.val[1], vdupq_n_u8(0x80))), moveMask);

                uint8x16_t red;
                uint8x16_t green;
                uint8x16_t blue;
                RGB5ToRGB6(colors.val[0], colors.val[1], red, green, blue);

                DrawPixels(&BGOBJLine[8 + i], moveMask, 
                    red,
                    green,
                    vorrq_u8(blue, vdupq_n_u8(1 << 7)),
                    vdupq_n_u8(1 << bgnum));
            }
        }
        else
        {
            // 256-color bitmap

            u8* tilemapptr = GPU::GetBGCachePtr(Num, tilemapaddr, size);

            for (int i = 0; i < 256; i += 16)
            {
                uint8x16_t pixels;
                uint8x16_t moveMask;

                unroll4(j,
                    uint8x16_t overflow = vreinterpretq_u8_u32(vorrq_u32(
                        vtstq_u32(vreinterpretq_u32_s32(vecRotX), vreinterpretq_u32_s32(vecOfxMask)), 
                        vtstq_u32(vreinterpretq_u32_s32(vecRotY), vreinterpretq_u32_s32(vecOfyMask))));
                    moveMask = vreinterpretq_u8_u32(vsetq_lane_u32(
                        vreinterpretq_u32_u8(vuzp1q_u8(vuzp1q_u8(overflow, overflow), vuzp1q_u8(overflow, overflow)))[0],
                        vreinterpretq_u32_u8(moveMask), j));

                    int32x4_t offset = vaddq_s32(
                            vshlq_s32(vshrq_n_s32(vandq_s32(vecRotY, vecYMask), 8), vecYShift), 
                            vshrq_n_s32(vandq_s32(vecRotX, vecXMask), 8));

                    unroll4(k,
                        pixels = vld1q_lane_u8(tilemapptr + offset[k], pixels, j * 4 + k);)

                    vecRotX = vaddq_s32(vecRotX, dx);
                    vecRotY = vaddq_s32(vecRotY, dy);
                )

                uint8x16_t windowMask = vtstq_u8(vld1q_u8(&WindowMask[8 + i]), vdupq_n_u8(1 << bgnum));
                moveMask = vbicq_u8(vbicq_u8(windowMask, vceqzq_u8(pixels)), moveMask);

                DrawPixels(&BGOBJLine[8 + i], moveMask, 
                    pixels, vdupq_n_u8(Num ? 2 : 0), vdupq_n_u8(0),
                    vdupq_n_u8(1 << bgnum));
            }
        }
    }
    else
    {
        // mixed affine/text mode

        u32 coordmask;
        u32 yshift;
        u32 size;
        switch (bgcnt & 0xC000)
        {
        case 0x0000: coordmask = 0xF; yshift = 7; size = 128*128; break;
        case 0x4000: coordmask = 0x1F; yshift = 8; size = 256*256; break;
        case 0x8000: coordmask = 0x3F; yshift = 9; size = 512*512; break;
        case 0xC000: coordmask = 0x7F; yshift = 10; size = 1024*1024; break;
        }
        yshift -= 3;

        u32 overflowmask;
        if (bgcnt & 0x2000) overflowmask = 0;
        else                overflowmask = ~coordmask;

        if (Num)
        {
            tilesetaddr = ((bgcnt & 0x003C) << 12);
            tilemapaddr = ((bgcnt & 0x1F00) << 3);
        }
        else
        {
            tilesetaddr = ((DispCnt & 0x07000000) >> 8) + ((bgcnt & 0x003C) << 12);
            tilemapaddr = ((DispCnt & 0x38000000) >> 11) + ((bgcnt & 0x1F00) << 3);
        }
        
        u8* tilemapptr = GPU::GetBGCachePtr(Num, tilemapaddr, size*2);
        u8* tilesetptr = GPU::GetBGCachePtr(Num, tilesetaddr, 1024*64);

        uint16x8_t tilenumMask = vdupq_n_u16(0x3FF);
        uint16x8_t tileMask = vdupq_n_u16(7);

        uint16x8_t hflipBit = vdupq_n_u16(0x400);
        uint16x8_t vflipBit = vdupq_n_u16(0x800);

        uint16x8_t vecOverflowMask = vdupq_n_u16(overflowmask);
        uint16x8_t vecCoordMask = vdupq_n_u16(coordmask);
        int16x8_t vecYShift = vdupq_n_s16(yshift);

        uint8x16_t extpalMask = vdupq_n_u8(extpal ? 0xFF : 0);

        u32 paletteOffset;
        if (extpal)
            paletteOffset = (Num ? GPU::FastPalExtBOffset : GPU::FastPalExtAOffset) + bgnum * 16;
        else
            paletteOffset = Num ? 2 : 0;
        uint8x16_t vecPaletteOffset = vdupq_n_u8(paletteOffset);

        uint16x8_t extpalUsed = vdupq_n_u16(0);

        for (int i = 0; i < 256; i += 32)
        {
            uint16x8x4_t tileoff;
            uint8x16_t extpalPal;

            uint8x16_t moveMask0;
            uint8x16_t moveMask1;

            uint8x16x2_t extpalIndex;

            unroll4(j,
                int32x4_t tweenRotX = vaddq_s32(vecRotX, dx);
                int32x4_t tweenRotY = vaddq_s32(vecRotY, dy);

                uint16x8_t rotXLo11 = vreinterpretq_u16_s16(vshrn_high_n_s32(
                        vshrn_n_s32(vecRotX, 11), tweenRotX, 11));
                uint16x8_t rotYLo11 = vreinterpretq_u16_s16(vshrn_high_n_s32(
                        vshrn_n_s32(vecRotY, 11), tweenRotY, 11));

                uint8x16_t overflow = vreinterpretq_u8_u16(vtstq_u16(vorrq_u16(rotXLo11, rotYLo11), vecOverflowMask));
                overflow = vuzp1q_u8(overflow, overflow);

                if (j == 0 || j == 1)
                    moveMask0 = vreinterpretq_u8_u64(vsetq_lane_u64(
                        vreinterpretq_u64_u8(overflow)[0],
                        vreinterpretq_u64_u8(moveMask0),
                        j & 1));
                else
                    moveMask1 = vreinterpretq_u8_u64(vsetq_lane_u64(
                        vreinterpretq_u64_u8(overflow)[0],
                        vreinterpretq_u64_u8(moveMask1),
                        j & 1));

                uint16x8_t offset = vaddq_u16(vandq_u16(rotXLo11, vecCoordMask),
                    vshlq_u16(vandq_u16(rotYLo11, vecCoordMask), vecYShift));

                uint16x8_t tiles;
                unroll8(k,
                    tiles = vld1q_lane_u16((u16*)(tilemapptr + offset[k] * 2), tiles, k);)

                extpalIndex.val[j >> 1] = vreinterpretq_u8_u64(vsetq_lane_u64(
                        // shrn doesn't work with 12-bit shifts (for some reason)
                        // so we have to do it in two steps
                        vreinterpret_u64_u8(vshr_n_u8(vshrn_n_u16(tiles, 8), 4))[0],
                        vreinterpretq_u64_u8(extpalIndex.val[j >> 1]),
                        j & 1));

                extpalUsed = vorrq_u16(extpalUsed, vshlq_u16(vdupq_n_u16(1), vreinterpretq_s16_u16(vshrq_n_u16(tiles, 12)))); 

                uint16x8_t localxoff = vandq_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotX, 8), tweenRotX, 8)), tileMask); 
                uint16x8_t localyoff = vandq_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotY, 8), tweenRotY, 8)), tileMask); 

                localxoff = vbslq_u16(vtstq_u16(tiles, hflipBit),
                    vsubq_u16(vdupq_n_u16(7), localxoff), localxoff);
                localyoff = vbslq_u16(vtstq_u16(tiles, vflipBit),
                    vsubq_u16(vdupq_n_u16(7), localyoff), localyoff);

                tileoff.val[j] = vaddq_u16(
                    vaddq_u16(vshlq_n_u16(vandq_u16(tiles, tilenumMask), 6), localxoff),
                    vshlq_n_u16(localyoff, 3));

                vecRotX = vaddq_s32(tweenRotX, dx);
                vecRotY = vaddq_s32(tweenRotY, dy);
            )

            uint8x16_t pixels0, pixels1;
            unroll2(j, unroll8(k,
                    pixels0 = vld1q_lane_u8(tilesetptr + tileoff.val[j][k], pixels0, j * 8 + k);
                    pixels1 = vld1q_lane_u8(tilesetptr + tileoff.val[j + 2][k], pixels1, j * 8 + k);
            ))

            uint8x16x2_t windowMask = vld1q_u8_x2(&WindowMask[i + 8]);

            moveMask0 = vbicq_u8(vbicq_u8(vtstq_u8(windowMask.val[0], vdupq_n_u8(1 << bgnum)), vceqzq_u8(pixels0)), moveMask0);
            moveMask1 = vbicq_u8(vbicq_u8(vtstq_u8(windowMask.val[1], vdupq_n_u8(1 << bgnum)), vceqzq_u8(pixels1)), moveMask1);

            DrawPixels(&BGOBJLine[8 + i], moveMask0, pixels0, vaddq_u8(vecPaletteOffset, vandq_u8(extpalIndex.val[0], extpalMask)), vdupq_n_u8(0), vdupq_n_u8(1 << bgnum));
            DrawPixels(&BGOBJLine[8 + 16 + i], moveMask1, pixels1, vaddq_u8(vecPaletteOffset, vandq_u8(extpalIndex.val[1], extpalMask)), vdupq_n_u8(0), vdupq_n_u8(1 << bgnum));
        }

        if (extpal)
        {
            u64 extpalUsed64 = 
                vreinterpretq_u64_u16(extpalUsed)[0] | vreinterpretq_u64_u16(extpalUsed)[1];
            extpalUsed64 |= extpalUsed64 >> 32;
            extpalUsed64 |= extpalUsed64 >> 16;
            extpalUsed64 &= 0xFFFF;

            BGExtPalUsed |= extpalUsed64 << (bgnum * 16);
        }
    }

    BGXRefInternal[bgnum-2] += rotB;
    BGYRefInternal[bgnum-2] += rotD;
}

template <bool mosaic>
void GPU2DNeon::DrawBG_Large(u32 line)
{
    u16 bgcnt = BGCnt[2];

    // large BG sizes:
    // 0: 512x1024
    // 1: 1024x512
    // 2: 512x256
    // 3: 512x512
    u32 xmask, ymask;
    u32 yshift;
    switch (bgcnt & 0xC000)
    {
    case 0x0000: xmask = 0x1FFFF; ymask = 0x3FFFF; yshift = 9; break;
    case 0x4000: xmask = 0x3FFFF; ymask = 0x1FFFF; yshift = 10; break;
    case 0x8000: xmask = 0x1FFFF; ymask = 0x0FFFF; yshift = 9; break;
    case 0xC000: xmask = 0x1FFFF; ymask = 0x1FFFF; yshift = 9; break;
    }

    u32 ofxmask, ofymask;
    if (bgcnt & 0x2000)
    {
        ofxmask = 0;
        ofymask = 0;
    }
    else
    {
        ofxmask = ~xmask;
        ofymask = ~ymask;
    }

    s16 rotA = BGRotA[0];
    s16 rotB = BGRotB[0];
    s16 rotC = BGRotC[0];
    s16 rotD = BGRotD[0];

    s32 rotX = BGXRefInternal[0];
    s32 rotY = BGYRefInternal[0];

    if (bgcnt & 0x0040)
    {
        // vertical mosaic
        rotX -= (BGMosaicY * rotB);
        rotY -= (BGMosaicY * rotD);
    }

    const int32x4_t factorDist = {0, 1, 2, 3};
    int32x4_t vecRotX = vaddq_s32(vdupq_n_s32(rotX), vmulq_s32(factorDist, vdupq_n_s32(rotA)));
    int32x4_t vecRotY = vaddq_s32(vdupq_n_s32(rotY), vmulq_s32(factorDist, vdupq_n_s32(rotC)));

    int32x4_t dx = vshlq_n_s32(vdupq_n_s32(rotA), 2);
    int32x4_t dy = vshlq_n_s32(vdupq_n_s32(rotC), 2);

    int32x4_t vecOfxMask = vdupq_n_s32(ofxmask);
    int32x4_t vecOfyMask = vdupq_n_s32(ofymask);
    int32x4_t vecYShift = vdupq_n_s32(yshift);

    int32x4_t vecXMask = vdupq_n_s32(xmask);
    int32x4_t vecYMask = vdupq_n_s32(ymask);

    u8* tilemapptr = GPU::GetBGCachePtr(Num, 0, 512*1024);

    // 256-color bitmap
    for (int i = 0; i < 256; i += 16)
    {
        uint8x16_t pixels;
        uint8x16_t moveMask;

        unroll4(j,
            uint8x16_t overflow = vreinterpretq_u8_u32(vorrq_u32(
                vtstq_u32(vreinterpretq_u32_s32(vecRotX), vreinterpretq_u32_s32(vecOfxMask)), 
                vtstq_u32(vreinterpretq_u32_s32(vecRotY), vreinterpretq_u32_s32(vecOfyMask))));
            moveMask = vreinterpretq_u8_u32(vsetq_lane_u32(
                vreinterpretq_u32_u8(vuzp1q_u8(vuzp1q_u8(overflow, overflow), vuzp1q_u8(overflow, overflow)))[0],
                vreinterpretq_u32_u8(moveMask), j));

            int32x4_t offset = vaddq_s32(
                    vshlq_s32(vshrq_n_s32(vandq_s32(vecRotY, vecYMask), 8), vecYShift), 
                    vshrq_n_s32(vandq_s32(vecRotX, vecXMask), 8));

            unroll4(k, pixels = vld1q_lane_u8(tilemapptr + offset[k], pixels, j * 4 + k);)

            vecRotX = vaddq_s32(vecRotX, dx);
            vecRotY = vaddq_s32(vecRotY, dy);
        )

        uint8x16_t windowMask = vtstq_u8(vld1q_u8(&WindowMask[8 + i]), vdupq_n_u8(0x4));
        moveMask = vbicq_u8(vbicq_u8(windowMask, vceqzq_u8(pixels)), moveMask);

        DrawPixels(&BGOBJLine[8 + i], moveMask, 
            pixels, vdupq_n_u8(Num ? 2 : 0), vdupq_n_u8(0),
            vdupq_n_u8(1 << 2));
    }

    BGXRefInternal[0] += rotB;
    BGYRefInternal[0] += rotD;
}

#define DoDrawSprite(type, ...) \
    if (iswin) \
    { \
        DrawSprite_##type<true>(__VA_ARGS__); \
    } \
    else \
    { \
        DrawSprite_##type<false>(__VA_ARGS__); \
    }
void GPU2DNeon::DrawSprites(u32 line)
{
    if (line == 0)
    {
        // reset those counters here
        // TODO: find out when those are supposed to be reset
        // it would make sense to reset them at the end of VBlank
        // however, sprites are rendered one scanline in advance
        // so they need to be reset a bit earlier

        OBJMosaicY = 0;
        OBJMosaicYCount = 0;
    }

    OBJExtPalUsed = 0;
    SemiTransSprites = false;

    NumSprites[0] = NumSprites[1] = NumSprites[2] = NumSprites[3] = 0;
    memset(OBJLine, 0, 272*4);
    memset(OBJWindow, 0, 272);

    if (!(DispCnt & 0x1000)) return;

    PROFILER_SECTION(drawsprites)

    memset(OBJIndex, 0xFF, 272);

    u16* oam = (u16*)&GPU::OAM[Num ? 0x400 : 0];

    if (!(GPU::OAMStatus & (1 << Num)))
    {
        NumSpritesPerLayer[0] = NumSpritesPerLayer[1] = NumSpritesPerLayer[2] = NumSpritesPerLayer[3] = 0;
        for (int i = 127; i >= 0; i--)
        {
            u16* attrib = &oam[i*4];

            if ((attrib[0] & 0x300) == 0x200)
                continue;

            u32 bgnum = 3 - ((attrib[2] & 0x0C00) >> 10);
            u32 index = NumSpritesPerLayer[bgnum]++;
            SpriteCache[bgnum][index] = i;
        }
        GPU::OAMStatus |= 1 << Num;
    }

    const s32 spritewidth[16] =
    {
        8, 16, 8, 8,
        16, 32, 8, 8,
        32, 32, 16, 8,
        64, 64, 32, 8
    };
    const s32 spriteheight[16] =
    {
        8, 8, 16, 8,
        16, 8, 32, 8,
        32, 16, 32, 8,
        64, 32, 64, 8
    };

    for (int bgnum = 0; bgnum < 4; bgnum++)
    {
        for (int i = 0; i < NumSpritesPerLayer[bgnum]; i++)
        {
            u32 sprnum = SpriteCache[bgnum][i];
            u16* attrib = &oam[sprnum*4];

            bool iswin = (((attrib[0] >> 10) & 0x3) == 2);

            u32 sprline;
            if ((attrib[0] & 0x1000) && !iswin)
            {
                // apply Y mosaic
                sprline = OBJMosaicY;
            }
            else
                sprline = line;

            if (attrib[0] & 0x0100)
            {
                u32 sizeparam = (attrib[0] >> 14) | ((attrib[1] & 0xC000) >> 12);
                s32 width = spritewidth[sizeparam];
                s32 height = spriteheight[sizeparam];
                s32 boundwidth = width;
                s32 boundheight = height;

                if (attrib[0] & 0x0200)
                {
                    boundwidth <<= 1;
                    boundheight <<= 1;
                }

                u32 ypos = attrib[0] & 0xFF;
                ypos = (sprline - ypos) & 0xFF;
                if (ypos >= (u32)boundheight)
                    continue;

                s32 xpos = (s32)(attrib[1] << 23) >> 23;
                if (xpos <= -boundwidth)
                    continue;

                u32 rotparamgroup = (attrib[1] >> 9) & 0x1F;

                DoDrawSprite(Rotscale, sprnum, boundwidth, boundheight, width, height, xpos, ypos);

                NumSprites[3 - bgnum]++;
            }
            else
            {
                u32 sizeparam = (attrib[0] >> 14) | ((attrib[1] & 0xC000) >> 12);
                s32 width = spritewidth[sizeparam];
                s32 height = spriteheight[sizeparam];

                u32 ypos = attrib[0] & 0xFF;
                ypos = (sprline - ypos) & 0xFF;
                if (ypos >= (u32)height)
                    continue;

                s32 xpos = (s32)(attrib[1] << 23) >> 23;
                if (xpos <= -width)
                    continue;

                DoDrawSprite(Normal, sprnum, width, height, xpos, ypos);

                NumSprites[3 - bgnum]++;
            }
        }
    }

    PROFILER_END_SECTION
}

template <bool invertMoveMask>
inline void DrawSpritePixels(u32* objlinePtr, u8* objindicesPtr, uint8x16_t moveMask, uint8x16_t primary,
    uint8x16_t secondary, uint8x16_t tertiary, uint8x16_t quaternary, uint8x16_t tertiaryTrans,
    uint8x16_t index)
{
    uint8x16x4_t objline = vld4q_u8((u8*)objlinePtr);
    uint8x16_t indices = vld1q_u8(objindicesPtr);

    uint8x16_t objlineEmpty = vceqzq_u8(objline.val[2]);

    if (invertMoveMask)
    {
        objline.val[0] = vbslq_u8(moveMask, objline.val[0], primary);
        objline.val[1] = vbslq_u8(moveMask, objline.val[1], secondary);
        objline.val[2] = vbslq_u8(moveMask, objline.val[2], tertiary);
        objline.val[3] = vbslq_u8(moveMask, objline.val[3], quaternary);

        objline.val[2] = vbslq_u8(vandq_u8(objlineEmpty, moveMask), tertiaryTrans, objline.val[2]);
        indices = vbslq_u8(vornq_u8(objlineEmpty, moveMask), index, indices);
    }
    else
    {
        objline.val[0] = vbslq_u8(moveMask, primary, objline.val[0]);
        objline.val[1] = vbslq_u8(moveMask, secondary, objline.val[1]);
        objline.val[2] = vbslq_u8(moveMask, tertiary, objline.val[2]);
        objline.val[3] = vbslq_u8(moveMask, quaternary, objline.val[3]);

        objline.val[2] = vbslq_u8(vbicq_u8(objlineEmpty, moveMask), tertiaryTrans, objline.val[2]);
        indices = vbslq_u8(vorrq_u8(objlineEmpty, moveMask), index, indices);
    }

    vst4q_u8((u8*)objlinePtr, objline);
    vst1q_u8(objindicesPtr, indices);
}
template <bool invertMoveMask>
inline void DrawSpritePixelsHalf(u32* objlinePtr, u8* objindicesPtr, uint8x8_t moveMask, uint8x8_t primary,
    uint8x8_t secondary, uint8x8_t tertiary, uint8x8_t quaternary, uint8x8_t tertiaryTrans,
    uint8x8_t index)
{
    uint8x8x4_t objline = vld4_u8((u8*)objlinePtr);
    uint8x8_t indices = vld1_u8(objindicesPtr);

    uint8x8_t objlineEmpty = vceqz_u8(objline.val[3]);

    if (invertMoveMask)
    {
        objline.val[0] = vbsl_u8(moveMask, objline.val[0], primary);
        objline.val[1] = vbsl_u8(moveMask, objline.val[1], secondary);
        objline.val[2] = vbsl_u8(moveMask, objline.val[2], tertiary);
        objline.val[3] = vbsl_u8(moveMask, objline.val[3], quaternary);

        objline.val[2] = vbsl_u8(vand_u8(objlineEmpty, moveMask), tertiaryTrans, objline.val[2]);
        indices = vbsl_u8(vorn_u8(objlineEmpty, moveMask), index, indices);
    }
    else
    {
        objline.val[0] = vbsl_u8(moveMask, primary, objline.val[0]);
        objline.val[1] = vbsl_u8(moveMask, secondary, objline.val[1]);
        objline.val[2] = vbsl_u8(moveMask, tertiary, objline.val[2]);
        objline.val[3] = vbsl_u8(moveMask, quaternary, objline.val[3]);

        objline.val[2] = vbsl_u8(vbic_u8(objlineEmpty, moveMask), objline.val[2], tertiaryTrans);
        indices = vbsl_u8(vorr_u8(objlineEmpty, moveMask), index, indices);
    }

    vst4_u8((u8*)objlinePtr, objline);
    vst1_u8(objindicesPtr, indices);
}
template <bool invertMoveMask>
inline void DrawSpritePixelsWindow(u8* windowPtr, uint8x16_t moveMask)
{
    uint8x16_t window = vld1q_u8(windowPtr);
    if (invertMoveMask)
        window = vbslq_u8(moveMask, window, vdupq_n_u8(1));
    else
        window = vbslq_u8(moveMask, vdupq_n_u8(1), window);
    vst1q_u8(windowPtr, window);
}
template <bool invertMoveMask>
inline void DrawSpritePixelsWindowHalf(u8* windowPtr, uint8x8_t moveMask)
{
    uint8x8_t window = vld1_u8(windowPtr);
    if (invertMoveMask)
        window = vbsl_u8(moveMask, window, vdup_n_u8(1));
    else
        window = vbsl_u8(moveMask, vdup_n_u8(1), window);
    vst1_u8(windowPtr, window);
}

template<bool window>
void GPU2DNeon::DrawSprite_Normal(u32 num, u32 width, u32 height, s32 xpos, s32 ypos)
{
    u16* oam = (u16*)&GPU::OAM[Num ? 0x400 : 0];
    u16* attrib = &oam[num * 4];

    u8 compositorFlag = 0;
    u8 spriteFlags = ((attrib[2] & 0x0C00) >> 10) | 0x14;
    u32 tilenum = attrib[2] & 0x03FF;
    u32 spritemode = window ? 0 : ((attrib[0] >> 10) & 0x3);

    u32 wmask = width - 8; // really ((width - 1) & ~0x7)

    if ((attrib[0] & 0x1000) && !window)
        spriteFlags |= 0x8;

    // yflip
    if (attrib[1] & 0x2000)
        ypos = height-1 - ypos;

    // adjust to be a multiple of 8
    // xpos includes the 8px padding
    u32 xoff;
    u32 xend = width;
    if (xpos >= 0)
    {
        xoff = 0;
        if ((xpos+xend) > 256)
            xend = 256 + 8 + (xpos & 0x7) - xpos;

        xpos += 8;
    }
    else
    {
        u32 tileoff = xpos & 0x7;
        if (tileoff == 0)
            tileoff = 8;

        xoff = tileoff - 8 - xpos;
        xpos = tileoff;
    }

    uint8x16_t hflipMask = vdupq_n_u8(attrib[1] & 0x1000 ? 0xFF : 0);
    u32 xleft = xend - xoff;
    uint8x16_t vecIndex = vdupq_n_u8(num);

    if (spritemode == 3)
    {
        // bitmap sprite

        u32 alpha = attrib[2] >> 12;
        if (!alpha) return;
        alpha++;

        SemiTransSprites = true;

        compositorFlag |= alpha | 0xC0;
        spriteFlags |= 0x80;

        uint8x16_t vecCompositorFlags = vdupq_n_u8(compositorFlag);
        uint8x16_t vecSpriteFlags = vdupq_n_u8(spriteFlags);
        uint8x16_t vecSpriteFlagsTrans = vdupq_n_u8(spriteFlags & 0x18);

        if (DispCnt & 0x40)
        {
            if (DispCnt & 0x20)
            {
                // 'reserved'
                // draws nothing
                return;
            }
            else
            {
                tilenum <<= (7 + ((DispCnt >> 22) & 0x1));
                tilenum += (ypos * width * 2);
            }
        }
        else
        {
            if (DispCnt & 0x20)
            {
                tilenum = ((tilenum & 0x01F) << 4) + ((tilenum & 0x3E0) << 7);
                tilenum += (ypos * 256 * 2);
            }
            else
            {
                tilenum = ((tilenum & 0x00F) << 4) + ((tilenum & 0x3F0) << 7);
                tilenum += (ypos * 128 * 2);
            }
        }

        u8* pixelsptr = GPU::GetOBJCachePtr(Num, tilenum, xleft / 8 * 128);
        s32 pixelstride;
        if (attrib[1] & 0x1000) // xflip
        {
            pixelsptr += (width << 1);
            pixelsptr -= (xoff << 1);
            pixelsptr -= 16;
            pixelstride = -16;
        }
        else
        {
            pixelsptr += (xoff << 1);
            pixelstride = 16;
        }

        for (; xleft >= 16; xleft -= 16)
        {
            uint8x8x2_t pixels0 = vld2_u8(pixelsptr);
            pixelsptr += pixelstride;
            uint8x8x2_t pixels1 = vld2_u8(pixelsptr);
            pixelsptr += pixelstride;

            uint8x16_t pixelsLo = vcombine_u8(pixels0.val[0], pixels1.val[0]);
            uint8x16_t pixelsHi = vcombine_u8(pixels0.val[1], pixels1.val[1]);

            pixelsLo = vbslq_u8(hflipMask, vrev64q_u8(pixelsLo), pixelsLo);
            pixelsHi = vbslq_u8(hflipMask, vrev64q_u8(pixelsHi), pixelsHi);

            uint8x16_t moveMask = vtstq_u8(pixelsHi, vdupq_n_u8(0x80));

            if (window)
                DrawSpritePixelsWindow<false>(&OBJWindow[xpos], moveMask);
            else
                DrawSpritePixels<false>(&OBJLine[xpos], &OBJIndex[xpos], 
                    moveMask, pixelsLo, pixelsHi, vecSpriteFlags, vecCompositorFlags, vecSpriteFlagsTrans, 
                    vecIndex);

            xpos += 16;
        }
        if (xleft == 8)
        {
            uint8x8x2_t pixels = vld2_u8(pixelsptr);
            pixelsptr += pixelstride;

            pixels.val[0] = vbsl_u8(vget_low_u8(hflipMask), vrev64_u8(pixels.val[0]), pixels.val[0]);
            pixels.val[1] = vbsl_u8(vget_low_u8(hflipMask), vrev64_u8(pixels.val[1]), pixels.val[1]);

            uint8x8_t moveMask = vtst_u8(pixels.val[1], vdup_n_u8(0x80));

            if (window)
                DrawSpritePixelsWindowHalf<false>(&OBJWindow[xpos], moveMask);
            else
                DrawSpritePixelsHalf<false>(&OBJLine[xpos], &OBJIndex[xpos], 
                    moveMask, pixels.val[0], pixels.val[1], vget_low_u8(vecSpriteFlags), 
                    vget_low_u8(vecCompositorFlags), vget_low_u8(vecSpriteFlagsTrans), vget_low_u8(vecIndex));
            xpos += 8;
            xleft -= 8;
        }
    }
    else
    {
        if (DispCnt & 0x10)
        {
            tilenum <<= ((DispCnt >> 20) & 0x3);
            tilenum += ((ypos >> 3) * (width >> 3)) << ((attrib[0] & 0x2000) ? 1:0);
        }
        else
        {
            tilenum += ((ypos >> 3) * 0x20);
        }

        // compositor flag (semi transparent or not)
        if (spritemode == 1)
        {
            SemiTransSprites = true;
            compositorFlag |= 0x80;
        }
        else
            compositorFlag |= 0x10;

        uint8x16_t vecCompositorFlags = vdupq_n_u8(compositorFlag);
        uint8x16_t vecSpriteFlags = vdupq_n_u8(spriteFlags);
        uint8x16_t vecSpriteFlagsTrans = vdupq_n_u8(spriteFlags & 0x18);
        if (attrib[0] & 0x2000)
        {
            // 256-color
            u8* pixelsptr = GPU::GetOBJCachePtr(Num, tilenum << 5, xleft / 8 * 64);
            pixelsptr += ((ypos & 0x7) << 3);

            s32 pixelstride;
            if (attrib[1] & 0x1000) // xflip
            {
                pixelsptr += (((width-1) & wmask) << 3);
                pixelsptr -= ((xoff & wmask) << 3);
                pixelstride = -64;
            }
            else
            {
                pixelsptr += ((xoff & wmask) << 3);
                pixelstride = 64;
            }

            u8 paletteIndex = 0;
            if (!window)
            {
                if (DispCnt & 0x80000000)
                {
                    u32 extPalSlot = (attrib[2] & 0xF000) >> 12;

                    paletteIndex = (Num ? GPU::FastPalExtBOffset : GPU::FastPalExtAOffset)
                        + GPU::FastPalExtBGSize + extPalSlot;
                    OBJExtPalUsed |= 1 << extPalSlot;
                }
                else
                    paletteIndex = Num ? 3 : 1;
            }
            uint8x16_t vecPalIndex = vdupq_n_u8(paletteIndex);
            for (; xleft >= 16; xleft -= 16)
            {
                uint8x16x4_t objline = vld4q_u8((u8*)&OBJLine[xpos]);
                uint8x16_t indices = vld1q_u8((u8*)&OBJIndex[xpos]);

                uint8x16_t pixels;
                pixels = vreinterpretq_u8_u64(vld1q_lane_u64((uint64_t*)pixelsptr, vreinterpretq_u64_u8(pixels), 0));
                pixelsptr += pixelstride;
                pixels = vreinterpretq_u8_u64(vld1q_lane_u64((uint64_t*)pixelsptr, vreinterpretq_u64_u8(pixels), 1));
                pixelsptr += pixelstride;

                pixels = vbslq_u8(hflipMask, vrev64q_u8(pixels), pixels);

                uint8x16_t moveMask = vceqzq_u8(pixels);

                if (window)
                    DrawSpritePixelsWindow<true>(&OBJWindow[xpos], moveMask);
                else
                    DrawSpritePixels<true>(&OBJLine[xpos], &OBJIndex[xpos], moveMask,
                        pixels, vecPalIndex, vecSpriteFlags, vecSpriteFlagsTrans, vecCompositorFlags, vecIndex);

                xpos += 16;
            }
            if (xleft == 8)
            {
                uint8x8_t pixels = vld1_u8(pixelsptr);
                pixelsptr += pixelstride;

                pixels = vbsl_u8(vget_low_u8(hflipMask), vrev64_u8(pixels), pixels);

                uint8x8_t moveMask = vceqz_u8(pixels);
                if (window)
                    DrawSpritePixelsWindowHalf<true>(&OBJWindow[xpos], moveMask);
                else
                    DrawSpritePixelsHalf<true>(&OBJLine[xpos], &OBJIndex[xpos], moveMask, 
                        pixels, vget_low_u8(vecPalIndex), vget_low_u8(vecSpriteFlags), 
                        vget_low_u8(vecCompositorFlags), vget_low_u8(vecSpriteFlagsTrans),
                        vget_low_u8(vecIndex));

                // only for the sake of completeness
                xpos += 8;
                xleft -= 8;
            }
        }
        else
        {
            // 16-color
            u8* pixelsptr = GPU::GetOBJCachePtr(Num, tilenum << 5, xleft / 8 * 32);
            pixelsptr += ((ypos & 0x7) << 2);

            s32 pixelstride;
            if (attrib[1] & 0x1000) // xflip
            {
                pixelsptr += (((width-1) & wmask) << 2);
                pixelsptr -= ((xoff & wmask) << 2);
                pixelstride = -32;
            }
            else
            {
                pixelsptr += ((xoff & wmask) << 2);
                pixelstride = 32;
            }

            u8 paletteIndex = Num ? 3 : 1;
            uint8x16_t paletteOffset = vdupq_n_u8((attrib[2] & 0xF000) >> 8);
            uint8x16_t paletteIndexVec = vdupq_n_u8(paletteIndex);
            for (; xleft >= 16; xleft -= 16)
            {
                uint8x16x4_t objline = vld4q_u8((u8*)&OBJLine[xpos]);
                uint8x16_t indices = vld1q_u8((u8*)&OBJIndex[xpos]);

                uint8x8_t pixels4Bit;
                pixels4Bit = vreinterpret_u8_u32(vld1_lane_u32((u32*)pixelsptr, vreinterpret_u32_u8(pixels4Bit), 0));
                pixelsptr += pixelstride;
                pixels4Bit = vreinterpret_u8_u32(vld1_lane_u32((u32*)pixelsptr, vreinterpret_u32_u8(pixels4Bit), 1));
                pixelsptr += pixelstride;

                uint8x16_t pixels = vzip1q_u8(
                    vcombine_u8(vshr_n_u8(vshl_n_u8(pixels4Bit, 4), 4), vdup_n_u8(0)), 
                    vcombine_u8(vshr_n_u8(pixels4Bit, 4), vdup_n_u8(0)));
                pixels = vbslq_u8(hflipMask, vrev64q_u8(pixels), pixels);

                uint8x16_t moveMask = vceqzq_u8(pixels);

                if (window)
                    DrawSpritePixelsWindow<true>(&OBJWindow[xpos], moveMask);
                else
                    DrawSpritePixels<true>(&OBJLine[xpos], &OBJIndex[xpos],
                        moveMask, vaddq_u8(pixels, paletteOffset), 
                        paletteIndexVec, vecSpriteFlags, vecCompositorFlags, vecSpriteFlagsTrans, vecIndex);
                xpos += 16;
            }
            if (xleft == 8)
            {
                uint8x8x4_t objline = vld4_u8((u8*)&OBJLine[xpos]);
                uint8x8_t indices = vld1_u8((u8*)&OBJIndex[xpos]);

                uint8x8_t pixels;
                pixels = vreinterpret_u8_u32(vld1_lane_u32((u32*)pixelsptr, vreinterpret_u32_u8(pixels), 0));
                pixelsptr += pixelstride;

                pixels = vzip1_u8(vshr_n_u8(vshl_n_u8(pixels, 4), 4), vshr_n_u8(pixels, 4));
                pixels = vbsl_u8(vget_low_u8(hflipMask), vrev64_u8(pixels), pixels);

                uint8x8_t moveMask = vceqz_u8(pixels);
                if (window)
                    DrawSpritePixelsWindowHalf<true>(&OBJWindow[xpos], moveMask);
                else
                    DrawSpritePixelsHalf<true>(&OBJLine[xpos], &OBJIndex[xpos],
                        moveMask, vadd_u8(pixels, vget_low_u8(paletteOffset)), 
                        vget_low_u8(paletteIndexVec), vget_low_u8(vecSpriteFlags), 
                        vget_low_u8(vecCompositorFlags), vget_low_u8(vecSpriteFlagsTrans),
                        vget_low_u8(vecIndex));

                // only for the sake of completeness
                xpos += 8;
                xleft -= 8;
            }
        }
    }
}


template<bool window>
void GPU2DNeon::DrawSprite_Rotscale(u32 num, u32 boundwidth, u32 boundheight, u32 width, u32 height, s32 xpos, s32 ypos)
{
    u16* oam = (u16*)&GPU::OAM[Num ? 0x400 : 0];
    u16* attrib = &oam[num * 4];
    u16* rotparams = &oam[(((attrib[1] >> 9) & 0x1F) * 16) + 3];

    u8 compositorFlags = 0;
    u8 spriteFlags = ((attrib[2] & 0x0C00) >> 10) | 0xC;
    u32 tilenum = attrib[2] & 0x03FF;
    u32 spritemode = window ? 0 : ((attrib[0] >> 10) & 0x3);

    u32 ytilefactor;

    s32 centerX = boundwidth >> 1;
    s32 centerY = boundheight >> 1;

    if ((attrib[0] & 0x1000) && !window)
        spriteFlags |= 0x8;

    u32 xoff;
    if (xpos >= 0)
    {
        xoff = 0;
        if ((xpos+boundwidth) > 256)
            boundwidth = 256 + 8 + (xpos & 0x7) - xpos;

        xpos += 8;
    }
    else
    {
        u32 tileoff = xpos & 0x7;
        if (tileoff == 0)
            tileoff = 8;

        xoff = tileoff - 8 - xpos;
        xpos = tileoff;
    }

    s16 rotA = (s16)rotparams[0];
    s16 rotB = (s16)rotparams[4];
    s16 rotC = (s16)rotparams[8];
    s16 rotD = (s16)rotparams[12];

    s32 rotX = ((xoff-centerX) * rotA) + ((ypos-centerY) * rotB) + (width << 7);
    s32 rotY = ((xoff-centerX) * rotC) + ((ypos-centerY) * rotD) + (height << 7);

    int32x4_t dx = vshlq_n_s32(vdupq_n_s32(rotA), 2);
    int32x4_t dy = vshlq_n_s32(vdupq_n_s32(rotC), 2);

    const int32x4_t factorDist = {0, 1, 2, 3};
    int32x4_t vecRotX = vaddq_s32(vdupq_n_s32(rotX), vmulq_s32(vdupq_n_s32(rotA), factorDist));
    int32x4_t vecRotY = vaddq_s32(vdupq_n_s32(rotY), vmulq_s32(vdupq_n_s32(rotC), factorDist));

    u32 xleft = boundwidth - xoff;

    uint16x8_t vecWidth = vdupq_n_u16(width);
    uint16x8_t vecHeight = vdupq_n_u16(height);

    uint8x16_t vecIndex = vdupq_n_u8(num);

    if (spritemode == 3)
    {
        u32 alpha = attrib[2] >> 12;
        if (!alpha) return;
        alpha++;

        compositorFlags |= 0xC0 | alpha;
        spriteFlags |= 0x80;

        SemiTransSprites = true;

        uint8x16_t vecCompositorFlags = vdupq_n_u8(compositorFlags);
        uint8x16_t vecSpriteFlags = vdupq_n_u8(spriteFlags);
        uint8x16_t vecSpriteFlagsTrans = vdupq_n_u8(spriteFlags & 0x18);

        if (DispCnt & 0x40)
        {
            if (DispCnt & 0x20)
            {
                // 'reserved'
                // draws nothing

                return;
            }
            else
            {
                tilenum <<= (7 + ((DispCnt >> 22) & 0x1));
                ytilefactor = ((width >> 8) * 2);
            }
        }
        else
        {
            if (DispCnt & 0x20)
            {
                tilenum = ((tilenum & 0x01F) << 4) + ((tilenum & 0x3E0) << 7);
                ytilefactor = (256 * 2);
            }
            else
            {
                tilenum = ((tilenum & 0x00F) << 4) + ((tilenum & 0x3F0) << 7);
                ytilefactor = (128 * 2);
            }
        }

        uint16x8_t vecYTileFactor = vdupq_n_u16(ytilefactor);
        u8* pixelsptr = GPU::GetOBJCachePtr(Num, tilenum, (xleft / 8 + 1) * 8 * 2);

        for (; xleft >= 16; xleft -= 16)
        {
            uint16x8x2_t offsets;
            uint8x16_t moveMask;

            unroll2(j,
                int32x4_t tweenRotX = vaddq_s32(vecRotX, dx);
                int32x4_t tweenRotY = vaddq_s32(vecRotY, dy);

                uint16x8_t widthCheck = vcgeq_u16(vshrn_high_n_u32(vshrn_n_u32(
                    vreinterpretq_u32_s32(vecRotX), 8), 
                    vreinterpretq_u32_s32(tweenRotX), 8), vecWidth);
                uint16x8_t heightCheck = vcgeq_u16(vshrn_high_n_u32(vshrn_n_u32(
                    vreinterpretq_u32_s32(vecRotY), 8), 
                    vreinterpretq_u32_s32(tweenRotY), 8), vecHeight);

                uint16x8_t outsideBounds = vorrq_u16(widthCheck, heightCheck);

                moveMask = vreinterpretq_u8_u64(vsetq_lane_u64(
                    vreinterpret_u64_u8(vmovn_u16(outsideBounds))[0], 
                    vreinterpretq_u64_u8(moveMask),
                    j));

                offsets.val[j] = vaddq_u16(
                    vmulq_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotY, 8), tweenRotY, 8)), vecYTileFactor),
                    vshlq_n_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotX, 8), tweenRotX, 8)), 1));

                offsets.val[j] = vbslq_u16(outsideBounds, vdupq_n_u16(0), offsets.val[j]);

                vecRotX = vaddq_s32(tweenRotX, dx);
                vecRotY = vaddq_s32(tweenRotY, dy);
            )

            uint8x16x2_t pixels;
            unroll2(j, unroll8(k,
                    pixels = vld2q_lane_u8(pixelsptr + offsets.val[j][k], pixels, j * 8 + k);))

            moveMask = vbicq_u8(vtstq_u8(pixels.val[1], vdupq_n_u8(0x80)), moveMask);

            if (window)
                DrawSpritePixelsWindow<false>(&OBJWindow[xpos], moveMask);
            else
                DrawSpritePixels<false>(&OBJLine[xpos], &OBJIndex[xpos], moveMask, pixels.val[0], 
                    pixels.val[1], vecSpriteFlags, vecCompositorFlags, vecSpriteFlagsTrans, vecIndex);

            xpos += 16;
        }
        if (xleft == 8)
        {
            int32x4_t tweenRotX = vaddq_s32(vecRotX, dx);
            int32x4_t tweenRotY = vaddq_s32(vecRotY, dy);

            uint16x8_t widthCheck = vcgeq_u16(vshrn_high_n_u32(vshrn_n_u32(
                vreinterpretq_u32_s32(vecRotX), 8), 
                vreinterpretq_u32_s32(tweenRotX), 8), vecWidth);
            uint16x8_t heightCheck = vcgeq_u16(vshrn_high_n_u32(vshrn_n_u32(
                vreinterpretq_u32_s32(vecRotY), 8), 
                vreinterpretq_u32_s32(tweenRotY), 8), vecHeight);

            uint16x8_t outsideBounds = vorrq_u16(widthCheck, heightCheck);

            uint16x8_t offsets = vaddq_u16(
                vmulq_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotY, 8), tweenRotY, 8)), vecYTileFactor),
                vshlq_n_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotX, 8), tweenRotX, 8)), 1));
            offsets = vbslq_u16(outsideBounds, vdupq_n_u16(0), offsets);

            vecRotX = vaddq_s32(tweenRotX, dx);
            vecRotY = vaddq_s32(tweenRotY, dy);

            uint8x8x2_t pixels;
            unroll8(j, pixels = vld2_lane_u8(pixelsptr + offsets[j], pixels, j);)

            uint8x8_t moveMask = vbic_u8(vtst_u8(pixels.val[1], vdup_n_u8(0x80)), moveMask);

            if (window)
                DrawSpritePixelsWindowHalf<false>(&OBJWindow[xpos], moveMask);
            else
                DrawSpritePixelsHalf<false>(&OBJLine[xpos], &OBJIndex[xpos], moveMask, pixels.val[0], 
                    pixels.val[1], vget_low_u8(vecSpriteFlags), vget_low_u8(vecCompositorFlags), 
                    vget_low_u8(vecSpriteFlagsTrans), vget_low_u8(vecIndex));

            xpos += 8;
        }
    }
    else
    {
        if (DispCnt & 0x10)
        {
            tilenum <<= ((DispCnt >> 20) & 0x3);
            ytilefactor = (width >> 3) << ((attrib[0] & 0x2000) ? 1:0);
        }
        else
        {
            ytilefactor = 0x20;
        }

        if (spritemode == 1)
        {
            SemiTransSprites = true;
            compositorFlags |= 0x80;
        }
        else
            compositorFlags |= 0x10;

        uint8x16_t vecCompositorFlags = vdupq_n_u8(compositorFlags);
        uint8x16_t vecSpriteFlags = vdupq_n_u8(spriteFlags);
        uint8x16_t vecSpriteFlagsTrans = vdupq_n_u8(spriteFlags & 0x18);

        if (attrib[0] & 0x2000)
        {
            // 256-color
            tilenum <<= 5;
            ytilefactor <<= 5;
            u8* pixelsptr = GPU::GetOBJCachePtr(Num, tilenum, (xleft / 8 + 1) * 64);

            u32 paletteIndex = 0;
            if (!window)
            {
                if (DispCnt & 0x80000000)
                {
                    u32 extPalSlot = (attrib[2] & 0xF000) >> 12;

                    paletteIndex = (Num ? GPU::FastPalExtBOffset : GPU::FastPalExtAOffset)
                        + GPU::FastPalExtBGSize + extPalSlot;
                    OBJExtPalUsed |= 1 << extPalSlot;
                }
                else
                    paletteIndex = Num ? 3 : 1;
            }

            uint8x16_t vecPaletteIndex = vdupq_n_u8(paletteIndex);
            uint16x8_t vecYTileFactor = vdupq_n_u16(ytilefactor);

            uint16x8_t tileMaskX = vdupq_n_u16(0x7);
            uint16x8_t tileMaskY = vdupq_n_u16(0x38);

            for (; xleft >= 16; xleft -= 16)
            {
                uint16x8x2_t offsets;
                uint8x16_t moveMask;

                unroll2(j,
                    int32x4_t tweenRotX = vaddq_s32(vecRotX, dx);
                    int32x4_t tweenRotY = vaddq_s32(vecRotY, dy);

                    uint16x8_t widthCheck = vcgeq_u16(vshrn_high_n_u32(vshrn_n_u32(
                        vreinterpretq_u32_s32(vecRotX), 8), 
                        vreinterpretq_u32_s32(tweenRotX), 8), vecWidth);
                    uint16x8_t heightCheck = vcgeq_u16(vshrn_high_n_u32(vshrn_n_u32(
                        vreinterpretq_u32_s32(vecRotY), 8), 
                        vreinterpretq_u32_s32(tweenRotY), 8), vecHeight);

                    uint16x8_t outsideBounds = vorrq_u16(widthCheck, heightCheck);

                    moveMask = vreinterpretq_u8_u64(vsetq_lane_u64(
                        vreinterpret_u64_u8(vmovn_u16(outsideBounds))[0], 
                        vreinterpretq_u64_u8(moveMask),
                        j));

                    offsets.val[j] = vaddq_u16(vaddq_u16(vaddq_u16(
                        vmulq_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotY, 11), tweenRotY, 11)), vecYTileFactor),
                        vandq_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotY, 5), tweenRotY, 5)), tileMaskY)),
                        vshlq_n_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotX, 11), tweenRotX, 11)), 6)),
                        vandq_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotX, 8), tweenRotX, 8)), tileMaskX));

                    offsets.val[j] = vbslq_u16(outsideBounds, vdupq_n_u16(0), offsets.val[j]);

                    vecRotX = vaddq_s32(tweenRotX, dx);
                    vecRotY = vaddq_s32(tweenRotY, dy);
                )

                uint8x16_t pixels;
                unroll2(j, unroll8(k,
                        pixels = vld1q_lane_u8(pixelsptr + offsets.val[j][k], pixels, j * 8 + k);
                ))

                moveMask = vorrq_u8(vceqzq_u8(pixels), moveMask);

                if (window)
                    DrawSpritePixelsWindow<true>(&OBJWindow[xpos], moveMask);
                else
                    DrawSpritePixels<true>(&OBJLine[xpos], &OBJIndex[xpos], moveMask, pixels, 
                        vecPaletteIndex, vecSpriteFlags, vecCompositorFlags, vecSpriteFlagsTrans, vecIndex);

                xpos += 16;
            }
            if (xleft == 8)
            {
                int32x4_t tweenRotX = vaddq_s32(vecRotX, dx);
                int32x4_t tweenRotY = vaddq_s32(vecRotY, dy);

                uint16x8_t widthCheck = vcgeq_u16(vshrn_high_n_u32(
                    vshrn_n_u32(vreinterpretq_u32_s32(vecRotX), 8), 
                    vreinterpretq_u32_s32(tweenRotX), 8), vecWidth);
                uint16x8_t heightCheck = vcgeq_u16(vshrn_high_n_u32(
                    vshrn_n_u32(vreinterpretq_u32_s32(vecRotY), 8), 
                    vreinterpretq_u32_s32(tweenRotY), 8), vecHeight);
                uint16x8_t outsideBounds = vorrq_u16(widthCheck, heightCheck);

                uint16x8_t offsets = vaddq_u16(vaddq_u16(vaddq_u16(
                    vmulq_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotY, 11), tweenRotY, 11)), vecYTileFactor),
                    vandq_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotY, 5), tweenRotY, 5)), tileMaskY)),
                    vshlq_n_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotX, 11), tweenRotX, 11)), 6)),
                    vandq_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotX, 8), tweenRotX, 8)), tileMaskX));

                offsets = vbslq_u16(outsideBounds, vdupq_n_u16(0), offsets);

                vecRotX = vaddq_s32(tweenRotX, dx);
                vecRotY = vaddq_s32(tweenRotY, dy);

                uint8x8_t pixels;
                unroll8(j,
                    pixels = vld1_lane_u8(pixelsptr + offsets[j], pixels, j);)

                uint8x8_t moveMask = vorr_u8(vceqz_u8(pixels), vmovn_u16(outsideBounds));

                if (window)
                    DrawSpritePixelsWindowHalf<true>(&OBJWindow[xpos], moveMask);
                else
                    DrawSpritePixelsHalf<true>(&OBJLine[xpos], &OBJIndex[xpos], moveMask, pixels,
                        vget_low_u8(vecPaletteIndex), vget_low_u8(vecSpriteFlags), vget_low_u8(vecCompositorFlags),
                        vget_low_u8(vecSpriteFlagsTrans), vget_low_u8(vecIndex));

                xpos += 8;
            }
        }
        else
        {
            // 16-color
            tilenum <<= 5;
            ytilefactor <<= 5;
            u8* pixelsptr = GPU::GetOBJCachePtr(Num, tilenum, (xleft / 8 + 1) * 32);

            uint16x8_t vecYTileFactor = vdupq_n_u16(ytilefactor);

            uint16x8_t tileMaskX = vdupq_n_u16(0x3);
            uint16x8_t tileMaskY = vdupq_n_u16(0x1C);

            uint8x16_t vecPaletteIndex = vdupq_n_u8(Num ? 3 : 1);
            uint8x16_t colorOffset = vdupq_n_u8((attrib[2] & 0xF000) >> 8);

            for (; xleft >= 16; xleft -= 16)
            {
                uint16x8x2_t offsets;
                uint8x16_t moveMask;
                uint8x16_t evenPixel;

                unroll2(j,
                    int32x4_t tweenRotX = vaddq_s32(vecRotX, dx);
                    int32x4_t tweenRotY = vaddq_s32(vecRotY, dy);

                    uint16x8_t flooredX = vshrn_high_n_u32(
                        vshrn_n_u32(vreinterpretq_u32_s32(vecRotX), 8), 
                        vreinterpretq_u32_s32(tweenRotX), 8);
                    uint16x8_t flooredY = vshrn_high_n_u32(
                        vshrn_n_u32(vreinterpretq_u32_s32(vecRotY), 8), 
                        vreinterpretq_u32_s32(tweenRotY), 8);

                    uint16x8_t widthCheck = vcgeq_u16(flooredX, vecWidth);
                    uint16x8_t heightCheck = vcgeq_u16(flooredY, vecHeight);

                    uint16x8_t outsideBounds = vorrq_u16(widthCheck, heightCheck);

                    moveMask = vreinterpretq_u8_u64(vsetq_lane_u64(
                        vreinterpret_u64_u8(vmovn_u16(outsideBounds))[0], 
                        vreinterpretq_u64_u8(moveMask),
                        j));
                    evenPixel = vreinterpretq_u8_u64(vsetq_lane_u64(
                        vreinterpret_u64_u8(vmovn_u16(vceqzq_u16(vandq_u16(flooredX, vdupq_n_u16(1)))))[0], 
                        vreinterpretq_u64_u8(evenPixel),
                        j));

                    offsets.val[j] = vaddq_u16(vaddq_u16(vaddq_u16(
                        vmulq_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotY, 11), tweenRotY, 11)), vecYTileFactor),
                        vandq_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotY, 6), tweenRotY, 6)), tileMaskY)),
                        vshlq_n_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotX, 11), tweenRotX, 11)), 5)),
                        vandq_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotX, 9), tweenRotX, 9)), tileMaskX));

                    offsets.val[j] = vbslq_u16(outsideBounds, vdupq_n_u16(0), offsets.val[j]);

                    vecRotX = vaddq_s32(tweenRotX, dx);
                    vecRotY = vaddq_s32(tweenRotY, dy);
                )

                uint8x16_t pixels;
                unroll2(j,
                    unroll8(k,
                        pixels = vld1q_lane_u8(pixelsptr + offsets.val[j][k], pixels, j * 8 + k);
                ))
                pixels = vbslq_u8(evenPixel, vshrq_n_u8(vshlq_n_u8(pixels, 4), 4), vshrq_n_u8(pixels, 4));

                moveMask = vorrq_u8(vceqzq_u8(pixels), moveMask);

                if (window)
                    DrawSpritePixelsWindow<true>(&OBJWindow[xpos], moveMask);
                else
                    DrawSpritePixels<true>(&OBJLine[xpos], &OBJIndex[xpos], moveMask, vaddq_u8(pixels, colorOffset), 
                        vecPaletteIndex, vecSpriteFlags, vecCompositorFlags, vecSpriteFlagsTrans, vecIndex);
                xpos += 16;
            }
            if (xleft == 8)
            {
                int32x4_t tweenRotX = vaddq_s32(vecRotX, dx);
                int32x4_t tweenRotY = vaddq_s32(vecRotY, dy);

                uint16x8_t flooredX = vshrn_high_n_u32(
                    vshrn_n_u32(vreinterpretq_u32_s32(vecRotX), 8), 
                    vreinterpretq_u32_s32(tweenRotX), 8);
                uint16x8_t flooredY = vshrn_high_n_u32(
                    vshrn_n_u32(vreinterpretq_u32_s32(vecRotY), 8), 
                    vreinterpretq_u32_s32(tweenRotY), 8);

                uint16x8_t widthCheck = vcgeq_u16(flooredX, vecWidth);
                uint16x8_t heightCheck = vcgeq_u16(flooredY, vecHeight);

                uint16x8_t outsideBounds = vorrq_u16(widthCheck, heightCheck);
                uint8x8_t evenPixel = vmovn_u16(vceqzq_u16(vandq_u16(flooredX, vdupq_n_u16(1))));

                uint16x8_t offsets = vaddq_u16(vaddq_u16(vaddq_u16(
                    vmulq_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotY, 11), tweenRotY, 11)), vecYTileFactor),
                    vandq_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotY, 6), tweenRotY, 6)), tileMaskY)),
                    vshlq_n_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotX, 11), tweenRotX, 11)), 5)),
                    vandq_u16(vreinterpretq_u16_s16(vshrn_high_n_s32(vshrn_n_s32(vecRotX, 9), tweenRotX, 9)), tileMaskX));

                offsets = vbslq_u16(outsideBounds, vdupq_n_u16(0), offsets);

                vecRotX = vaddq_s32(tweenRotX, dx);
                vecRotY = vaddq_s32(tweenRotY, dy);

                uint8x8_t pixels;
                unroll8(j, pixels = vld1_lane_u8(pixelsptr + offsets[j], pixels, j);)
                pixels = vbsl_u8(evenPixel, vshr_n_u8(vshl_n_u8(pixels, 4), 4), vshr_n_u8(pixels, 4));

                uint8x8_t moveMask = vorr_u8(vceqz_u8(pixels), vmovn_u16(outsideBounds));

                if (window)
                    DrawSpritePixelsWindowHalf<true>(&OBJWindow[xpos], moveMask);
                else
                    DrawSpritePixelsHalf<true>(&OBJLine[xpos], &OBJIndex[xpos], moveMask, vadd_u8(pixels, vget_low_u8(colorOffset)), 
                        vget_low_u8(vecPaletteIndex), vget_low_u8(vecSpriteFlags), vget_low_u8(vecCompositorFlags), 
                        vget_low_u8(vecSpriteFlagsTrans), vget_low_u8(vecIndex));
                xpos += 8;
            }
        }
    }
}