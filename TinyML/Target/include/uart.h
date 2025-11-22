#ifndef UART_H
#define UART_H

#include <stdint.h>
#include <stddef.h>

void Uart_Write_Byte(uint8_t byte);
void Uart_Write_Buffer(const uint8_t* buffer, size_t length);

int Uart_Read_Byte();
int Uart_Read_Buffer(uint8_t* buffer, size_t length);

void Uart_Flush_Rx_Buffer();
void Uart_Flush_Tx_Buffer();

void Uart_Init();

#endif