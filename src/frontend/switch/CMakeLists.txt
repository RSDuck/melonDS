project(switch)

SET(SOURCES_SWITCH
    main.cpp
    Platform.cpp
    PlatformConfig.cpp
    Gfx.cpp
    BoxGui.cpp
    BackButton.cpp
    Filebrowser.cpp
    KeyExplanations.cpp
    ROMMetaDatabase.cpp
    SettingsDialog.cpp
    StartMenu.cpp
    FaultHandler.cpp
    ErrorDialog.cpp
    GpuMemHeap.cpp
    UploadBuffer.cpp

    ../Util_ROM.cpp
    ../Util_Audio.cpp
    ../Util_Video.cpp

    mm_vec/mm_vec.c
    stb_truetype/stb_truetype.c
    stb_image/stb_image.c
)

add_executable(melonDS.elf ${SOURCES_SWITCH})
if (${CMAKE_BUILD_TYPE} STREQUAL Debug)
    if (ENABLE_LTO)
        target_link_libraries(melonDS.elf nxd deko3dd core)
    else()
        target_link_libraries(melonDS.elf core deko3dd nxd)
    endif()
else()
    if (ENABLE_LTO)
        target_link_libraries(melonDS.elf core nx deko3d)
    else()
        target_link_libraries(melonDS.elf core deko3d nx)
    endif()
endif()

target_include_directories(melonDS.elf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(melonDS.elf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_include_directories(melonDS.elf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/control.nacp
                   COMMAND ${NACPTOOL} --create "melonDS switch fast alpha" "Arisotura, RSDuck, Hydr8gon" "release 6" ${CMAKE_BINARY_DIR}/control.nacp)

make_directory(${CMAKE_BINARY_DIR}/romfs/shaders)

configure_file(${CMAKE_SOURCE_DIR}/icon/melon_128x128.png ${CMAKE_BINARY_DIR}/romfs/melon_128x128.png COPYONLY)

compile_shader(vert Default_vsh)
compile_shader(frag Default_fsh)

compile_shader(vert ../../shaders/FullscreenQuad_vsh)
compile_shader_instance(frag ../../shaders/ComposeBGOBJ_fsh ComposeBGOBJ_fsh "ComposeBGOBJ")
compile_shader_instance(frag ../../shaders/ComposeBGOBJ_fsh ComposeBGOBJDirectBitmapOnly_fsh "ShowBitmap")
compile_shader_instance(frag ../../shaders/ComposeBGOBJ_fsh ComposeBGOBJShowBitmap_fsh "ShowBitmap;ComposeBGOBJ")
compile_shader_instance(frag ../../shaders/BGText_fsh BGText4bpp_fsh "Text4bpp")
compile_shader_instance(frag ../../shaders/BGText_fsh BGText4bppMosaic_fsh "Text4bpp;Mosaic")
compile_shader_instance(frag ../../shaders/BGText_fsh BGText8bpp_fsh "Text8bpp")
compile_shader_instance(frag ../../shaders/BGText_fsh BGText8bppMosaic_fsh "Text8bpp;Mosaic")
compile_shader_instance(frag ../../shaders/BGAffine_fsh BGAffine_fsh "Affine")
compile_shader_instance(frag ../../shaders/BGAffine_fsh BGAffineMosaic_fsh "Affine;Mosaic")
compile_shader_instance(frag ../../shaders/BGAffine_fsh BGExtendedBitmap8bpp_fsh "ExtendedBitmap8bpp")
compile_shader_instance(frag ../../shaders/BGAffine_fsh BGExtendedBitmap8bppMosaic_fsh "ExtendedBitmap8bpp;Mosaic")
compile_shader_instance(frag ../../shaders/BGAffine_fsh BGExtendedBitmapDirect_fsh "ExtendedBitmapDirectColor")
compile_shader_instance(frag ../../shaders/BGAffine_fsh BGExtendedBitmapDirectMosaic_fsh "ExtendedBitmapDirectColor;Mosaic")
compile_shader_instance(frag ../../shaders/BGAffine_fsh BGExtendedMixed_fsh "ExtendedMixed")
compile_shader_instance(frag ../../shaders/BGAffine_fsh BGExtendedMixedMosaic_fsh "ExtendedMixed;Mosaic")
compile_shader_instance(vert ../../shaders/OBJ_vsh OBJRegular_vsh "OBJRegular")
compile_shader_instance(vert ../../shaders/OBJ_vsh OBJAffine_vsh "OBJAffine")
compile_shader_instance(frag ../../shaders/OBJ_fsh OBJ4bpp_fsh "OBJ4bpp")
compile_shader_instance(frag ../../shaders/OBJ_fsh OBJ8bpp_fsh "OBJ8bpp")
compile_shader_instance(frag ../../shaders/OBJ_fsh OBJBitmap_fsh "OBJBitmap")
compile_shader_instance(frag ../../shaders/OBJ_fsh OBJWindow4bpp_fsh "OBJ4bpp;OBJWindow")
compile_shader_instance(frag ../../shaders/OBJ_fsh OBJWindow8bpp_fsh "OBJ8bpp;OBJWindow")
compile_shader_instance(comp ../../shaders/GPU3D_Comp InterpXSpansZBuffer "InterpSpans;ZBuffer")
compile_shader_instance(comp ../../shaders/GPU3D_Comp InterpXSpansWBuffer "InterpSpans;WBuffer")
compile_shader_instance(comp ../../shaders/GPU3D_Comp BinCombined "BinCombined")
compile_shader_instance(comp ../../shaders/GPU3D_Comp DepthBlendWBuffer "DepthBlend;WBuffer")
compile_shader_instance(comp ../../shaders/GPU3D_Comp DepthBlendZBuffer "DepthBlend;ZBuffer")
compile_shader_instance(comp ../../shaders/GPU3D_Comp RasteriseNoTextureWBuffer "Rasterise;WBuffer;NoTexture")
compile_shader_instance(comp ../../shaders/GPU3D_Comp RasteriseNoTextureWBufferToon "Rasterise;WBuffer;NoTexture;Toon")
compile_shader_instance(comp ../../shaders/GPU3D_Comp RasteriseNoTextureWBufferHighlight "Rasterise;WBuffer;NoTexture;Highlight")
compile_shader_instance(comp ../../shaders/GPU3D_Comp RasteriseUseTextureDecalWBuffer "Rasterise;WBuffer;UseTexture;Decal")
compile_shader_instance(comp ../../shaders/GPU3D_Comp RasteriseUseTextureModulateWBuffer "Rasterise;WBuffer;UseTexture;Modulate")
compile_shader_instance(comp ../../shaders/GPU3D_Comp RasteriseUseTextureToonWBuffer "Rasterise;WBuffer;UseTexture;Toon")
compile_shader_instance(comp ../../shaders/GPU3D_Comp RasteriseUseTextureHighlightWBuffer "Rasterise;WBuffer;UseTexture;Highlight")
compile_shader_instance(comp ../../shaders/GPU3D_Comp RasteriseShadowMaskWBuffer "Rasterise;WBuffer;ShadowMask")
compile_shader_instance(comp ../../shaders/GPU3D_Comp RasteriseNoTextureZBuffer "Rasterise;ZBuffer;NoTexture")
compile_shader_instance(comp ../../shaders/GPU3D_Comp RasteriseNoTextureZBufferToon "Rasterise;ZBuffer;NoTexture;Toon")
compile_shader_instance(comp ../../shaders/GPU3D_Comp RasteriseNoTextureZBufferHighlight "Rasterise;ZBuffer;NoTexture;Highlight")
compile_shader_instance(comp ../../shaders/GPU3D_Comp RasteriseUseTextureDecalZBuffer "Rasterise;ZBuffer;UseTexture;Decal")
compile_shader_instance(comp ../../shaders/GPU3D_Comp RasteriseUseTextureModulateZBuffer "Rasterise;ZBuffer;UseTexture;Modulate")
compile_shader_instance(comp ../../shaders/GPU3D_Comp RasteriseUseTextureToonZBuffer "Rasterise;ZBuffer;UseTexture;Toon")
compile_shader_instance(comp ../../shaders/GPU3D_Comp RasteriseUseTextureHighlightZBuffer "Rasterise;ZBuffer;UseTexture;Highlight")
compile_shader_instance(comp ../../shaders/GPU3D_Comp RasteriseShadowMaskZBuffer "Rasterise;ZBuffer;ShadowMask")
compile_shader_instance(comp ../../shaders/GPU3D_Comp FinalPass "FinalPass")
compile_shader_instance(comp ../../shaders/GPU3D_Comp FinalPassEdge "FinalPass;EdgeMarking")
compile_shader_instance(comp ../../shaders/GPU3D_Comp FinalPassFog "FinalPass;Fog")
compile_shader_instance(comp ../../shaders/GPU3D_Comp FinalPassEdgeFog "FinalPass;EdgeMarking;Fog")
compile_shader_instance(comp ../../shaders/GPU3D_Comp FinalPassAA "FinalPass;AntiAliasing")
compile_shader_instance(comp ../../shaders/GPU3D_Comp FinalPassEdgeAA "FinalPass;EdgeMarking;AntiAliasing")
compile_shader_instance(comp ../../shaders/GPU3D_Comp FinalPassFogAA "FinalPass;Fog;AntiAliasing")
compile_shader_instance(comp ../../shaders/GPU3D_Comp FinalPassEdgeFogAA "FinalPass;EdgeMarking;Fog;AntiAliasing")
compile_shader_instance(comp ../../shaders/GPU3D_Comp ClearCoarseBinMask "ClearCoarseBinMask")
compile_shader_instance(comp ../../shaders/GPU3D_Comp ClearIndirectWorkCount "ClearIndirectWorkCount")
compile_shader_instance(comp ../../shaders/GPU3D_Comp CalculateWorkOffsets "CalculateWorkOffsets")
compile_shader_instance(comp ../../shaders/GPU3D_Comp SortWork "SortWork")

message("${shader_paths}")
add_custom_target(melonDS.nro ALL
    ${ELF2NRO} ${CMAKE_BINARY_DIR}/melonDS.elf ${CMAKE_BINARY_DIR}/melonDS.nro --icon=${CMAKE_SOURCE_DIR}/icon/melon_switch.jpg --nacp=${CMAKE_BINARY_DIR}/control.nacp --romfsdir=${CMAKE_BINARY_DIR}/romfs
    DEPENDS melonDS.elf
    ${CMAKE_BINARY_DIR}/control.nacp
    "${shader_paths}")