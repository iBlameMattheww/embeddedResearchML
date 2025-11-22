#ifndef RP2040_REGS_H
#define RP2040_REGS_H

#include <stdint.h>

/*
 * ---------------------------
 *  RESETS Register Block
 * ---------------------------
 */

typedef struct
{
    volatile uint32_t RESET;        // 0x00
    volatile uint32_t WDSEL;        // 0x04
    volatile uint32_t RESET_DONE;   // 0x08
} rp2040_resets_hw_t;

#define RESETS_BASE 0x04000C000
#define RESETS ((rp2040_resets_hw_t *)RESETS_BASE)

#define RESETS_RESET_UART0_BITS (1u << 22)
#define RESETS_RESET_DONE_UART0_BITS (1u << 22)

/*
 * ---------------------------
 *  IO_BANK0 (GPIO MUX)
 * ---------------------------
 */

typedef struct
{
    volatile uint32_t STATUS;  
    volatile uint32_t CTRL;
} rp2040_iobank_gpio_t;

typedef struct
{
    rp2040_iobank_gpio_t GPIO[30];
} rp2040_iobank0_hw_t;

#define IO_BANK0_BASE 0x40014000
#define IO_BANK0 ((rp2040_iobank0_hw_t *)IO_BANK0_BASE)
#define GPIO_FUNC_UART 2

/*
 * ---------------------------
 *  UART0 Registers
 * ---------------------------
 */

typedef struct {
    volatile uint32_t DR;        // 0x00 Data register
    volatile uint32_t RSR;       // 0x04 Status register
    uint32_t _pad0[4];
    volatile uint32_t FR;        // 0x18 Flag register
    uint32_t _pad1;
    volatile uint32_t ILPR;
    volatile uint32_t IBRD;      // 0x24 Integer baud rate divisor
    volatile uint32_t FBRD;      // 0x28 Fractional baud rate divisor
    volatile uint32_t LCR_H;     // 0x2C Line control register
    volatile uint32_t CR;        // 0x30 Control register
    volatile uint32_t IFLS;
    volatile uint32_t IMSC;
    volatile uint32_t RIS;
    volatile uint32_t MIS;
    volatile uint32_t ICR;
    volatile uint32_t DMACR;
} rp2040_uart_hw_t;

#define UART0_BASE 0x40034000
#define UART0 ((rp2040_uart_hw_t *)UART0_BASE)

#endif // RP2040_REGS_H