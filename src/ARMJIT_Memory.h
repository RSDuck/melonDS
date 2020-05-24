#ifndef ARMJIT_MEMORY
#define ARMJIT_MEMORY

#include "types.h"

namespace ARMJIT_Memory
{

extern void* FastMem9Start;
extern void* FastMem7Start;

void Init();
void DeInit();

void Reset();

enum
{
	memregion_Other = 0,
	memregion_ITCM,
	memregion_DTCM,
	memregion_BIOS9,
	memregion_MainRAM,
	memregion_SWRAM9,
	memregion_SWRAM7,
	memregion_IO9,
	memregion_VRAM,
	memregion_BIOS7,
	memregion_WRAM7,
	memregion_IO7,
	memregion_Wifi,
	memregion_VWRAM,
	memregions_Count
};

int ClassifyAddress9(u32 addr);
int ClassifyAddress7(u32 addr);

bool IsMappable(int region);

}

#endif