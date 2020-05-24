#include "ARMJIT_Memory.h"

#include "ARMJIT_Internal.h"
#include "ARMJIT_Compiler.h"

#include "GPU.h"
#include "GPU3D.h"
#include "Wifi.h"
#include "NDSCart.h"
#include "SPU.h"

#include <malloc.h>

/*
	We're handling fastmem here.

	Basically we're repurposing a big piece of virtual memory
	and map the memory regions as they're structured on the DS
	in it.

	On most systems you have a single piece of main ram, 
	maybe some video ram and faster cache RAM and that's about it.
	Here we have a lot more different memory regions. Not only
	that but they all have mirrors (the worst case is 16kb SWRAM 
	which is mirrored 1024x).

	We handle this by only mapping those regions which are actually
	used and by praying the games don't go wild.

	Beware, this file is full of platform specific code.

*/

struct FaultDescription;

namespace ARMJIT_Memory
{
void FaultHandler(FaultDescription* faultDesc);
}

#ifdef __SWITCH__
#include "switch/compat_switch.h"

extern "C" void ARM_RestoreContext(u64* registers) __attribute__((noreturn));

struct FaultDescription
{
	ThreadExceptionDump* ExceptionDesc;

	u32 GetEmulatedAddr()
	{
		// now this is podracing
		return ExceptionDesc->cpu_gprs[0].w;
	}
	u64 RealAddr()
	{
		return ExceptionDesc->far.x;
	}

	// I'll laugh at myself when 128-bit processors become standard
	u64 GetPC()
	{
		return ExceptionDesc->pc.x;
	}

	void RestoreAndRepeat()
	{
		ExceptionDesc->pc.x -= 4;
		// I know this is hacky
		ARM_RestoreContext(&ExceptionDesc->cpu_gprs[0].x);
	}
};

extern char __start__;
extern char __rodata_start;

extern "C"
{

alignas(16) u8 __nx_exception_stack[0x8000];
u64 __nx_exception_stack_size = sizeof(__nx_exception_stack);

void __libnx_exception_handler(ThreadExceptionDump* ctx)
{
	FaultDescription desc;
	desc.ExceptionDesc = ctx;

	ARMJIT_Memory::FaultHandler(&desc);

	if (ctx->pc.x >= (u64)&__start__ && ctx->pc.x < (u64)&__rodata_start)
	{
		printf("non JIT fault in .text at 0x%x (type %d)\n", ctx->pc.x - (u64)&__start__, ctx->error_desc);
	}
	else
	{
		printf("non JIT fault somewhere in deep (address) space at %x (type %d)\n", ctx->pc.x, ctx->error_desc);
	}
	svcExitProcess();
}

}

#endif

namespace ARMJIT_Memory
{

struct Mapping
{
	u64 Addr;
	u32 Size, Offset;
	u32 Num;
};

ARMJIT::TinyVector<Mapping> Mappings[memregions_Count];

void* FastMem9Start, *FastMem7Start;

const u32 MemoryTotalSize =
	NDS::MainRAMSize
	+ NDS::SharedWRAMSize
	+ NDS::ARM7WRAMSize
	+ DTCMPhysicalSize;

const u32 MemBlockMainRAMOffset = 0;
const u32 MemBlockSWRAMOffset = NDS::MainRAMSize;
const u32 MemBlockARM7WRAMOffset = NDS::MainRAMSize + NDS::SharedWRAMSize;
const u32 MemBlockDTCMOffset = NDS::MainRAMSize + NDS::SharedWRAMSize + NDS::ARM7WRAMSize;


#ifdef __SWITCH__

u8* MemoryBase;
u8* MemoryBaseCodeMem;

#endif

u8* AllMemory;

bool MapIntoRange(void* dst, u32 offset, u32 size)
{
#ifdef __SWITCH__
	return R_SUCCEEDED(svcMapProcessMemory(dst, envGetOwnProcessHandle(), 
		(u64)(MemoryBaseCodeMem + offset), size));
#endif
}

bool MapIntoRangeRegister(u32 num, int region, u32 mirrorStart, u32 offset, u32 size)
{
	Mapping mapping = {mirrorStart, size, offset, num};
	Mappings[region].Add(mapping);

	if (num == 0)
		return MapIntoRange((u8*)FastMem9Start + mirrorStart, offset, size);
	else
		return MapIntoRange((u8*)FastMem7Start + mirrorStart, offset, size);
}

void UnmapFromRange(void* dst, u32 offset, u32 size)
{
#ifdef __SWITCH__
	svcUnmapProcessMemory(dst, envGetOwnProcessHandle(),
		(u64)(MemoryBaseCodeMem + offset), size);
#endif
}

bool MapAtAddress(u32 addr)
{
	u32 num = NDS::CurCPU;

	printf("map at address %d %08x\n", num, addr);

	int region = num == 0
		? ClassifyAddress9(addr)
		: ClassifyAddress7(addr);

	u32 mirrorStart;
	u32 mirrorSize;
	u32 memOffset;

	switch (region)
	{
	case memregion_MainRAM:
		mirrorStart = addr & ~(NDS::MainRAMSize - 1);
		mirrorSize = NDS::MainRAMSize;
		memOffset = MemBlockMainRAMOffset;
		break;
	//case memregion_DTCM:
	//	break;
	case memregion_WRAM7:
		mirrorStart = addr & ~(NDS::ARM7WRAMSize - 1);
		mirrorSize = NDS::ARM7WRAMSize;
		memOffset = MemBlockARM7WRAMOffset;
	default:
		return false;
	}

	if (num == 0)
	{
		// look whether the stupid DTCM breaks the mapping apart

		u32 dtcmBase = NDS::ARM9->DTCMBase;
		u32 dtcmEnd = NDS::ARM9->DTCMBase + NDS::ARM9->DTCMSize;

		// this is not an off by one error
		// but to prevent making a mapping with length 0
		// when the boundary of DTCM lies at the boundary
		// of the underlying region
		bool startInside = dtcmBase > mirrorStart && dtcmBase < (mirrorStart + mirrorSize);
		bool endInside = dtcmEnd > mirrorStart && dtcmEnd < (mirrorStart + mirrorSize);

		if (startInside && endInside)
		{
			printf("arm9 mapped region split into two by DTCM(%08x-%08x) %d %08x size: %x\n", 
				dtcmBase, dtcmEnd, region, mirrorStart, mirrorSize);

			assert(MapIntoRangeRegister(0, region, mirrorStart, memOffset, dtcmBase - mirrorStart));
			assert(MapIntoRangeRegister(0, region, dtcmEnd, memOffset, mirrorStart + mirrorSize - dtcmEnd));
		}
		else if (startInside)
		{
			printf("arm9 mapped region shortened on the right by DTCM(%08x-%08x) %d %08x size: %x",
				dtcmBase, dtcmEnd, region, mirrorStart, mirrorSize);
			assert(MapIntoRangeRegister(0, region, mirrorStart, memOffset, dtcmBase - mirrorStart));
		}
		else if (endInside)
		{
			printf("arm9 mapped region shortened on the left by DTCM(%08x-%08x) %d %08x size: %x",
				dtcmBase, dtcmEnd, region, mirrorStart, mirrorSize);
			assert(MapIntoRangeRegister(0, region, dtcmEnd, memOffset, mirrorStart + mirrorSize - dtcmEnd));
		}
		else
		{
			printf("arm9 mapped region %d %08x size: %x\n", region, mirrorStart, mirrorSize);
			assert(MapIntoRangeRegister(0, region, mirrorStart, memOffset, mirrorSize));
		}
	}
	else
	{
		printf("arm7, mapped region %d %08x size: %x\n", region, mirrorStart, mirrorSize);
		assert(MapIntoRangeRegister(1, region, mirrorStart, memOffset, mirrorSize));
	}

	return true;
}

void FaultHandler(FaultDescription* faultDesc)
{
	printf("fault!\n");
	if (ARMJIT::JITCompiler->IsJITFault(faultDesc->GetPC()))
	{
		bool protectedMemory = false;
		bool rewriteToSlowPath = true;
/*
#ifdef __SWITCH__
		u32 pageInfo;
		MemoryInfo memInfo;

		svcQueryMemory(&memInfo, &pageInfo, faultDesc->RealAddr());

		protectedMemory = memInfo.type != MemType_Unmapped;
#endif
		if (!protectedMemory)
		{
			// reading/writing unmapped memory, let's see if we can map something here

			rewriteToSlowPath = !MapAtAddress(faultDesc->GetEmulatedAddr());
		}*/

		if (rewriteToSlowPath)
		{
			ARMJIT::JITCompiler->RewriteMemAccess(faultDesc->GetPC());
		}
		faultDesc->RestoreAndRepeat();
	}
}

void Init()
{
#ifdef __SWITCH__
    MemoryBase = (u8*)memalign(0x1000, MemoryTotalSize);
	MemoryBaseCodeMem = (u8*)virtmemReserve(MemoryTotalSize);

    bool succeded = R_SUCCEEDED(svcMapProcessCodeMemory(envGetOwnProcessHandle(), (u64)MemoryBaseCodeMem, 
        (u64)MemoryBase, MemoryTotalSize));
    assert(succeded);
	succeded = R_SUCCEEDED(svcSetProcessMemoryPermission(envGetOwnProcessHandle(), (u64)MemoryBaseCodeMem, 
        MemoryTotalSize, Perm_Rx));
	assert(succeded);

	FastMem9Start = virtmemReserve(0x10000000);
	FastMem7Start = virtmemReserve(0x10000000);

	AllMemory = (u8*)virtmemReserve(MemoryTotalSize);
#endif

	assert(MapIntoRange(AllMemory, 0, MemoryTotalSize));

	NDS::MainRAM = AllMemory + MemBlockMainRAMOffset;
	NDS::SharedWRAM = AllMemory + MemBlockSWRAMOffset;
	NDS::ARM7WRAM = AllMemory + MemBlockARM7WRAMOffset;
	NDS::ARM9->DTCM = AllMemory + MemBlockDTCMOffset;
}

void DeInit()
{
	virtmemFree(FastMem9Start, 0x10000000);
	virtmemFree(FastMem7Start, 0x10000000);

	UnmapFromRange(AllMemory, 0, MemoryTotalSize);

	virtmemFree(AllMemory, MemoryTotalSize);
    svcUnmapProcessCodeMemory(envGetOwnProcessHandle(), (u64)MemoryBaseCodeMem, (u64)MemoryBase, MemoryTotalSize);
	virtmemFree(MemoryBaseCodeMem, MemoryTotalSize);
    free(MemoryBase);
}

void Reset()
{
	
}

bool IsMappable(int region)
{
    switch (region)
    {
    case memregion_MainRAM:
    case memregion_DTCM:
    case memregion_WRAM7:
    case memregion_SWRAM9:
    case memregion_SWRAM7:
        return true;
    default:
        return false;
    }
}

int ClassifyAddress9(u32 addr)
{
	if (addr < NDS::ARM9->ITCMSize)
		return memregion_ITCM;
	else if (addr >= NDS::ARM9->DTCMBase && addr < (NDS::ARM9->DTCMBase + NDS::ARM9->DTCMSize))
		return memregion_DTCM;
	else if ((addr & 0xFFFFF000) == 0xFFFF0000)
		return memregion_BIOS9;
	else
	{
		switch (addr & 0xFF000000)
		{
		case 0x02000000:
			return memregion_MainRAM;
		case 0x03000000:
			return memregion_SWRAM9;
		case 0x04000000:
			return memregion_IO9;
		case 0x06000000:
			return memregion_VRAM;
		}
	}
	return memregion_Other;
}

int ClassifyAddress7(u32 addr)
{
	if (addr < 0x00004000)
		return memregion_BIOS7;
	else
	{
		switch (addr & 0xFF800000)
		{
		case 0x02000000:
		case 0x02800000:
			return memregion_MainRAM;
		case 0x03000000:
			if (NDS::SWRAM_ARM7.Mem)
				return memregion_SWRAM7;
			else
				return memregion_WRAM7;
		case 0x03800000:
			return memregion_WRAM7;
		case 0x04000000:
			return memregion_IO7;
		case 0x04800000:
			return memregion_Wifi;
		case 0x06000000:
		case 0x06800000:
			return memregion_VWRAM;
		}
	}
	return memregion_Other;
}


void WifiWrite32(u32 addr, u32 val)
{
	Wifi::Write(addr, val & 0xFFFF);
	Wifi::Write(addr + 2, val >> 16);
}

u32 WifiRead32(u32 addr)
{
	return Wifi::Read(addr) | (Wifi::Read(addr + 2) << 16);
}

template <typename T>
void VRAMWrite(u32 addr, T val)
{
	switch (addr & 0x00E00000)
	{
	case 0x00000000: GPU::WriteVRAM_ABG<T>(addr, val); return;
	case 0x00200000: GPU::WriteVRAM_BBG<T>(addr, val); return;
	case 0x00400000: GPU::WriteVRAM_AOBJ<T>(addr, val); return;
	case 0x00600000: GPU::WriteVRAM_BOBJ<T>(addr, val); return;
	default: GPU::WriteVRAM_LCDC<T>(addr, val); return;
	}
}
template <typename T>
T VRAMRead(u32 addr)
{
	switch (addr & 0x00E00000)
	{
	case 0x00000000: return GPU::ReadVRAM_ABG<T>(addr);
	case 0x00200000: return GPU::ReadVRAM_BBG<T>(addr);
	case 0x00400000: return GPU::ReadVRAM_AOBJ<T>(addr);
	case 0x00600000: return GPU::ReadVRAM_BOBJ<T>(addr);
	default: return GPU::ReadVRAM_LCDC<T>(addr);
	}
}

void* GetFuncForAddr(ARM* cpu, u32 addr, bool store, int size)
{
	if (cpu->Num == 0)
	{
		switch (addr & 0xFF000000)
		{
		case 0x04000000:
			if (!store && size == 32 && addr == 0x04100010 && NDS::ExMemCnt[0] & (1<<11))
				return (void*)NDSCart::ReadROMData;

			/*
				unfortunately we can't map GPU2D this way
				since it's hidden inside an object

				though GPU3D registers are accessed much more intensive
			*/
			if (addr >= 0x04000320 && addr < 0x040006A4)
			{
				switch (size | store)
				{
				case 8: return (void*)GPU3D::Read8;		
				case 9: return (void*)GPU3D::Write8;		
				case 16: return (void*)GPU3D::Read16;
				case 17: return (void*)GPU3D::Write16;
				case 32: return (void*)GPU3D::Read32;
				case 33: return (void*)GPU3D::Write32;
				}
			}

			switch (size | store)
			{
			case 8: return (void*)NDS::ARM9IORead8;
			case 9: return (void*)NDS::ARM9IOWrite8;
			case 16: return (void*)NDS::ARM9IORead16;
			case 17: return (void*)NDS::ARM9IOWrite16;
			case 32: return (void*)NDS::ARM9IORead32;
			case 33: return (void*)NDS::ARM9IOWrite32;
			}
			break;
		case 0x06000000:
			switch (size | store)
			{
			case 8: return (void*)VRAMRead<u8>;		
			case 9: return NULL;
			case 16: return (void*)VRAMRead<u16>;
			case 17: return (void*)VRAMWrite<u16>;
			case 32: return (void*)VRAMRead<u32>;
			case 33: return (void*)VRAMWrite<u32>;
			}
			break;
		}
	}
	else
	{
		switch (addr & 0xFF800000)
		{
		case 0x04000000:
			if (addr >= 0x04000400 && addr < 0x04000520)
			{
				switch (size | store)
				{
				case 8: return (void*)SPU::Read8;		
				case 9: return (void*)SPU::Write8;		
				case 16: return (void*)SPU::Read16;
				case 17: return (void*)SPU::Write16;
				case 32: return (void*)SPU::Read32;
				case 33: return (void*)SPU::Write32;
				}
			}

			switch (size | store)
			{
			case 8: return (void*)NDS::ARM7IORead8;
			case 9: return (void*)NDS::ARM7IOWrite8;		
			case 16: return (void*)NDS::ARM7IORead16;
			case 17: return (void*)NDS::ARM7IOWrite16;
			case 32: return (void*)NDS::ARM7IORead32;
			case 33: return (void*)NDS::ARM7IOWrite32;
			}
			break;
		case 0x04800000:
			if (addr < 0x04810000 && size >= 16)
			{
				switch (size | store)
				{
				case 16: return (void*)Wifi::Read;
				case 17: return (void*)Wifi::Write;
				case 32: return (void*)WifiRead32;
				case 33: return (void*)WifiWrite32;
				}
			}
			break;
		case 0x06000000:
		case 0x06800000:
			switch (size | store)
			{
			case 8: return (void*)GPU::ReadVRAM_ARM7<u8>;
			case 9: return (void*)GPU::WriteVRAM_ARM7<u8>;
			case 16: return (void*)GPU::ReadVRAM_ARM7<u16>;
			case 17: return (void*)GPU::WriteVRAM_ARM7<u16>;
			case 32: return (void*)GPU::ReadVRAM_ARM7<u32>;
			case 33: return (void*)GPU::WriteVRAM_ARM7<u32>;
			}
		}
	}
	return NULL;
}

}