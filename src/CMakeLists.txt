project(core)

set (CMAKE_CXX_STANDARD 14)


add_library(core STATIC
	ARM.cpp
	ARMInterpreter.cpp
	ARMInterpreter_ALU.cpp
	ARMInterpreter_Branch.cpp
	ARMInterpreter_LoadStore.cpp
	ARM_InstrInfo.cpp
	Config.cpp
	CP15.cpp
	CRC32.cpp
	DMA.cpp
	GBACart.cpp
	GPU.cpp
	GPU2D.cpp
	GPU3D.cpp
	GPU3D_OpenGL.cpp
	GPU3D_Soft.cpp
	NDS.cpp
	NDSCart.cpp
	RTC.cpp
	Savestate.cpp
	SPI.cpp
	SPU.cpp
	Wifi.cpp
	WifiAP.cpp
)

if (ENABLE_OGLRENDERER)
	target_sources(core PRIVATE
		OpenGLSupport.cpp
	)

	add_definitions(-DOGLRENDERER_ENABLED)
endif()

if (ARCHITECTURE STREQUAL ARM64)
	option(ENABLE_NEONGPU "Enable GPU implementation utilising NEON instructions" ON)
endif()
if (ENABLE_NEONGPU)
	target_sources(core PRIVATE
		GPU2D_Neon.cpp
	)

	add_definitions(-DNEONGPU_ENABLED)
endif()

if (ENABLE_JIT)
	target_sources(core PRIVATE
		ARMJIT.cpp

		dolphin/CommonFuncs.cpp
		
	)

	if (ARCHITECTURE STREQUAL x86_64)
		target_sources(core PRIVATE
			dolphin/x64ABI.cpp
			dolphin/x64CPUDetect.cpp
			dolphin/x64Emitter.cpp

			ARMJIT_x64/ARMJIT_Compiler.cpp
			ARMJIT_x64/ARMJIT_ALU.cpp
			ARMJIT_x64/ARMJIT_LoadStore.cpp
			ARMJIT_x64/ARMJIT_Branch.cpp
		)
	endif()
	if (ARCHITECTURE STREQUAL ARM64)
		target_sources(core PRIVATE
			dolphin/Arm64Emitter.cpp
			dolphin/MathUtil.cpp

			ARMJIT_A64/ARMJIT_Compiler.cpp
			ARMJIT_A64/ARMJIT_ALU.cpp
			ARMJIT_A64/ARMJIT_LoadStore.cpp
			ARMJIT_A64/ARMJIT_Branch.cpp
		)
	endif()
endif()

if (WIN32)
	target_link_libraries(core ole32 comctl32 ws2_32 opengl32)
else()
	if (ENABLE_OGLRENDERER)
		target_link_libraries(core GL EGL)
	else()
		target_link_libraries(core)
	endif()
endif()
