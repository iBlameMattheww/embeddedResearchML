#include "Serial.h"
#include "tusb.h"
#include "pico/stdlib.h"

enum
{
    // RX command protocol (PC → Pico)
    StartByte = 0xAA,
    CommandByte = 1,
    PayloadLengthByte = 2,
    PayloadBeginByte = 3,

    // TX streaming protocol (Pico → PC)
    PacketStart = 0xA5,
    PacketPhase = 0x01,
    PacketDone  = 0xFF,

    PhaseCoordinateSize = 4,
    PhasePacketSize = 10, 
};


void SerialSendDone(void)
{
    uint8_t pkt[2] = { PacketStart, PacketDone };
    tud_cdc_n_write(CDC_ITF, pkt, sizeof(pkt));
}


void ClearSerialCommand(serial_t *serial)
{
    serial->_private.command = Cmd_None;
    serial->_private.pendingPayloadLength = 0;
}

bool SerialSendPhasePacket(int32_t p, int32_t q)
{
    uint8_t packet[10];

    packet[0] = 0xA5;
    packet[1] = 0x01;
    memcpy(&packet[2], &p, 4);
    memcpy(&packet[6], &q, 4);

    uint32_t written = tud_cdc_n_write(CDC_ITF, packet, 10);
    if (written == 10)
    {
        return true;
    }
    tud_cdc_n_write_flush(CDC_ITF);
    return false;
}

void SerialCopyPayload(serial_t *serial, void *destination)
{
    if (!serial->_private.pendingPayloadLength)
    {
        return;
    }
    memcpy(destination, serial->_private.pendingPayload, serial->_private.pendingPayloadLength);
}

serialCmd_t GetSerialCommand(serial_t *serial)
{
    return serial->_private.command;
}

bool IsSerialCommandAvailable(serial_t *serial)
{
    return serial->_private.command != Cmd_None;
}


// void SerialTask(serial_t *serial)
// {
//     (void)serial;

//     while (tud_cdc_n_available(0))
//     {
//         uint8_t byte = tud_cdc_n_read_char(0);
//         tud_cdc_n_write(0, &byte, 1);
//     }

//     // Flush once per loop, NOT per byte
//     tud_cdc_n_write_flush(0);
// }


void SerialTask(serial_t *serial)
{
    while (tud_cdc_n_available(0))
    {
        uint8_t byte = tud_cdc_n_read_char(0);

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

            if (serial->_private.rx_Length >= PayloadBeginByte)
            {
                uint8_t payloadLen = serial->_private.rx_Buffer[PayloadLengthByte];

                if (payloadLen > PENDING_PAYLOAD_SIZE)
                {
                    serial->_private.rx_Length = 0;
                    return;
                }

                if (serial->_private.rx_Length == PayloadBeginByte + payloadLen)
                {
                    serial->_private.command =
                        (serialCmd_t)serial->_private.rx_Buffer[CommandByte];

                    serial->_private.pendingPayloadLength = payloadLen;

                    memcpy(
                        serial->_private.pendingPayload,
                        &serial->_private.rx_Buffer[PayloadBeginByte],
                        payloadLen
                    );

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
    memset(serial, 0, sizeof(serial_t));
}