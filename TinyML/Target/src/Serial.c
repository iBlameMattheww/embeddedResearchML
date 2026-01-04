#include "Serial.h"
#include "tusb.h"
#include "pico/stdlib.h"
#include "utils.h"

enum
{
    // RX command protocol (PC → Pico)
    StartByte = 0xAA,
    CommandByte = 1,
    PayloadLengthByte = 2,
    PayloadBeginByte = 3,
    AckByte = 0x06,
    NackByte = 0x15,

    // TX streaming protocol (Pico → PC)
    PacketStart = 0xA5,
    PacketPhase = 0x01,
    PacketPayloadLength = 0x08,
    PacketDone  = 0xFF,

    PhaseCoordinateSize = 4,
    PhasePacketSize = 10, 
};


void SerialSendDone(serial_t *serial)
{
    uint8_t donePacket[5];
    donePacket[0] = PacketStart;
    donePacket[1] = PacketDone;
    donePacket[2] = 1;
    donePacket[3] = serial->_private.sequenceNumber;
    donePacket[4] = CRC_8(&donePacket[1], 3);

    tud_cdc_n_write(CDC_ITF, donePacket, sizeof(donePacket));
}


void ClearSerialCommand(serial_t *serial)
{
    serial->_private.command = Cmd_None;
    serial->_private.pendingPayloadLength = 0;
}

void SerialSendPhasePacket(serial_t *serial, int32_t p, int32_t q)
{
    uint8_t packet[13];

    packet[0] = PacketStart;
    packet[1] = PacketPhase;
    packet[2] = PacketPayloadLength;
    packet[3] = serial->_private.sequenceNumber;
    memcpy(&packet[4], &p, 4);
    memcpy(&packet[8], &q, 4);
    packet[12] = CRC_8(&packet[1], 11);

    tud_cdc_n_write(CDC_ITF, packet, sizeof(packet));
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

void SerialTask(serial_t *serial)
{
    while (tud_cdc_n_available(0))
    {
        uint8_t byte = tud_cdc_n_read_char(0);

        if (serial->_private.rx_Length == 0)
        {
            if (byte == AckByte)
            {
                serial->_private.acknowledged = true;
                continue;
            }
            else if (byte == NackByte)
            {
                serial->_private.acknowledged = false;
                continue;
            }

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