find_program(ELF2NRO elf2nro)
find_program(NACPTOOL nacptool)
find_program(UAM uam)

project(switch)

SET(SOURCES_SWITCH
    main.cpp
    Platform.cpp
    profiler.cpp

    dr_wav.cpp

    imgui/imgui_impl_deko3d.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
)

add_executable(melonDS.elf ${SOURCES_SWITCH})
target_link_libraries(melonDS.elf core deko3dd nx)

add_custom_command(OUTPUT control.nacp
                   COMMAND ${NACPTOOL} --create "melonDS switch fast alpha" "Arisotura, RSDuck, Hydr8gon" "0.1" ${CMAKE_BINARY_DIR}/control.nacp)

make_directory(${CMAKE_BINARY_DIR}/romfs/shaders)

add_custom_command(OUTPUT romfs/shaders/imgui_vsh.dksh
    COMMAND ${UAM} --stage=vert --out ${CMAKE_BINARY_DIR}/romfs/shaders/imgui_vsh.dksh ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_vsh.glsl
    DEPENDS imgui/imgui_vsh.glsl
)
add_custom_command(OUTPUT romfs/shaders/imgui_fsh.dksh
    COMMAND ${UAM} --stage=frag --out ${CMAKE_BINARY_DIR}/romfs/shaders/imgui_fsh.dksh ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_fsh.glsl
    DEPENDS imgui/imgui_fsh.glsl
)

add_custom_target(melonDS.nro ALL
    ${ELF2NRO} ${CMAKE_BINARY_DIR}/melonDS.elf ${CMAKE_BINARY_DIR}/melonDS.nro --icon=${CMAKE_BINARY_DIR}/../icon/melon.jpg --nacp=${CMAKE_BINARY_DIR}/control.nacp --romfsdir=${CMAKE_BINARY_DIR}/romfs
    DEPENDS melonDS.elf control.nacp romfs/shaders/imgui_vsh.dksh romfs/shaders/imgui_fsh.dksh)