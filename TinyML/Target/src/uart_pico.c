#include "uart.h"
#include "rp2040_regs.h"

void Uart_Write_Byte(uint8_t byte)
{
    while (UART0->FR & (1 << 5));
    UART0->DR = byte;
}

void Uart_Write_Buffer(const uint8_t* buffer, size_t length)
{
    for (size_t i = 0; i < length; i++)
    {
        Uart_Write_Byte(buffer[i]);
    }
}

int Uart_Read_Byte()
{
    while (UART0->FR & (1 << 4));
    return UART0->DR & 0xFF;
}

int Uart_Read_Buffer(uint8_t* buffer, size_t length)
{
    for (size_t i = 0; i < length; i++)
    {
        buffer[i] = (uint8_t)Uart_Read_Byte();
    }
    return length;
}

void Uart_Flush_Rx_Buffer()
{
    while (!(UART0->FR & (1 << 4)))
    {
        volatile uint32_t dummy = UART0->DR;
        (void)dummy;
    }
}

void Uart_Flush_Tx_Buffer()
{
    while (!(UART0->FR & (1 << 3)));
}


void Uart_Init()
{ 
    RESETS->RESET &= ~RESETS_RESET_UART0_BITS;
    while (!(RESETS->RESET_DONE & RESETS_RESET_DONE_UART0_BITS));

    IO_BANK0->GPIO[0].CTRL = GPIO_FUNC_UART;
    IO_BANK0->GPIO[1].CTRL = GPIO_FUNC_UART;

    UART0->IBRD = 26;  
    UART0->FBRD = 3;
    UART0->LCR_H = (3 << 5);

    UART0->CR = (1 << 0) | (1 << 8) | (1 << 9);
}