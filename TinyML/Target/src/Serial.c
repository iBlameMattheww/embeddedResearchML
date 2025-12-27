#include "Serial.h"
#include "tusb.h"
#include "pico/stdlib.h"

enum
{
    PayloadBeginByte = 3,
    PhaseCoordinateSize = 4,
    PacketSize = 8,
    StartByte = 0xAA,
    DoneByte = 0xFF,
};

void SerialSendDone()
{
    uint8_t doneByte = DoneByte;
    tud_cdc_write(&doneByte, 1);
    tud_cdc_write_flush();
}

void SerialSendPhasePacket(int32_t p, int32_t q)
{
    uint8_t packet[PacketSize];
    memcpy(packet, &p, PhaseCoordinateSize);
    memcpy(packet + PhaseCoordinateSize, &q, PhaseCoordinateSize);
    tud_cdc_write(packet, PacketSize);
    tud_cdc_write_flush();
}

serialCmd_t GetSerialCommand(serial_t *serial)
{
    serialCmd_t cmd = serial->_private.command;
    if (cmd != Cmd_None)
    {
        memcpy(serial->_private.rx_Buffer, serial->_private.pendingPayload, serial->_private.pendingPayloadLength);
        serial->_private.command = Cmd_None;
    }
    return cmd;
}

bool IsSerialCommandAvailable(serial_t *serial)
{
    return serial->_private.command != Cmd_None;
}

void SerialTask(serial_t *serial)
{
    if (!tud_cdc_available())
    {
        return;
    }

    while(tud_cdc_available())
    {
        uint8_t byte = tud_cdc_read_char();

        if (serial->_private.rx_Length == 0)
        {
            if (byte == StartByte)
            {
                serial->_private.rx_Buffer[serial->_private.rx_Length++] = byte;
            }
        }
        else
        {
            serial->_private.rx_Buffer[serial->_private.rx_Length++] = byte;

            if (serial->_private.rx_Length == PayloadBeginByte)
            {
                // wait for payload
            }
            else if (serial->_private.rx_Length >= PayloadBeginByte)
            {
                uint8_t payloadLen = serial->_private.rx_Buffer[2];
                if (payloadLen > PENDING_PAYLOAD_SIZE)
                {
                    serial->_private.rx_Length = 0;
                    return;
                }
                if (serial->_private.rx_Length == PayloadBeginByte + payloadLen)
                {
                    serial->_private.command = (serialCmd_t)serial->_private.rx_Buffer[1];
                    serial->_private.pendingPayloadLength = payloadLen;
                    memcpy(serial->_private.pendingPayload, &serial->_private.rx_Buffer[PayloadBeginByte], payloadLen);
                    serial->_private.rx_Length = 0;
                }
            }

            if (serial->_private.rx_Length >= RX_BUFFER_SIZE)
            {
                serial->_private.rx_Length = 0;
            }
        }
    }
}

void Serial_Init(serial_t *serial) 
{
    stdio_init_all();
    memset(serial, 0, sizeof(serial_t));
}