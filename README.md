This is the source of my melonDS switch version.

It's a mess and not finished. Over the next time, I want to integrate the ARM Neon code(of course not as it's the current state!) into melonDS, maybe the Android port benefits from this too. The ARM64 JIT has already been included in the JIT branch of the melonDS main repository.

Credits:
- Arisotura, obviously for melonDS
- Dolphin people, of whom I've taken the JIT code emitter and who helped me on IRC!
- Hydr8gon, who did the original melonDS switch port (from which this ports borrows quite a lot of code)
- [dear imgui by Omar Cornut](https://github.com/ocornut/imgui), [vec developed by Michael Mettke](https://github.com/vurtun/mmx) and [dr_wav by David Reid](https://github.com/mackron/dr_libs) (see the included files for the appropriate licenses)
- endrift, the cmake switch buildscript is based of the of mgba
- libnx and devkitpro people!
- see also [original credits](https://github.com/Arisotura/melonDS#credits)
