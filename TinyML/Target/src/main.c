#include "inference.h"
#include "uart.h"
#define GPIO2_OUT_REG (*(volatile uint32_t *)(0x40014000 + 0x008))
#define GPIO2_OE_REG (*(volatile uint32_t *)(0x40014000 + 0x024))

int main(void)
{
    Uart_Init();
    // Set GPIO2 as output
    GPIO2_OE_REG |= (1 << 2);
    while (1)
    {
        Uart_Write_Buffer((const uint8_t *)"TinyML Inference Started\r\n", 26);
        // Toggle GPIO2
        GPIO2_OUT_REG ^= (1 << 2);
        for (volatile int i = 0; i < 1000000; ++i); // Simple delay
    }
    return 0;
}