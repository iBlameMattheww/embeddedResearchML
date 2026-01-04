#ifndef SERIAL_H
#define SERIAL_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#define CDC_ITF 0
#define PENDING_PAYLOAD_SIZE 32
#define RX_BUFFER_SIZE 64

typedef enum 
{
    Cmd_None = 0x00,
    Cmd_Run = 0x01,
    Cmd_Ping = 0x02,
} serialCmd_t;

typedef struct 
{
    struct 
    {
        serialCmd_t command;
        uint8_t rx_Buffer[RX_BUFFER_SIZE];
        uint8_t rx_Length;
        uint8_t pendingPayload[PENDING_PAYLOAD_SIZE];
        uint8_t pendingPayloadLength;
        uint8_t sequenceNumber;
        bool acknowledged;
    } _private; 
} serial_t;

void SerialSendDone(serial_t *serial);
void ClearSerialCommand(serial_t *serial);
void SerialSendPhasePacket(serial_t *serial, int32_t p, int32_t q);
void SerialCopyPayload(serial_t *serial, void *destination);
serialCmd_t GetSerialCommand(serial_t *serial);
bool IsSerialCommandAvailable(serial_t *serial);
void SerialTask(serial_t *serial);
void Serial_Init(serial_t *serial);

#endif