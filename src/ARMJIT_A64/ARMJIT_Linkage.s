#include "../ARMJIT_x64/ARMJIT_Offsets.h"

.text

#define RCPSR W27
#define RCycles W28
#define RCPU X29

.p2align 4,,15

.global ARM_Dispatch
ARM_Dispatch:
    stp x19, x20, [sp, #-96]!
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    stp x27, x28, [sp, #64]
    stp x29, x30, [sp, #80]

    mov RCPU, x0
    ldr RCycles, [RCPU, ARM_Cycles_offset]
    ldr RCPSR, [RCPU, ARM_CPSR_offset]

    br x1

.p2align 4,,15

.global ARM_Ret
ARM_Ret:
    str RCycles, [RCPU, ARM_Cycles_offset]
    str RCPSR, [RCPU, ARM_CPSR_offset]

    ldp x29, x30, [sp, #80]
    ldp x27, x28, [sp, #64]
    ldp x25, x26, [sp, #48]
    ldp x23, x24, [sp, #32]
    ldp x21, x22, [sp, #16]
    ldp x19, x20, [sp], #96

    ret