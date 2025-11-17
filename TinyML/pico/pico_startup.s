.syntax unified
.cpu cortex-m0plus
.thumb

.global _start
.global Reset_Handler

.section .text._start
_start:
    b Reset_Handler

.section .text.Reset_Handler
Reset_Handler:
    bl main
    b .
