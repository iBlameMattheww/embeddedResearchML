import serial
import struct
import time
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# SERIAL CONFIG
# -------------------------------
PORT = "/dev/tty.usbmodem1201"
BAUD = 115200
TIMEOUT = 1.0

# -------------------------------
# PROTOCOL CONSTANTS
# -------------------------------
PKT_START  = 0xA5
PKT_PHASE  = 0x01
PKT_DONE   = 0xFF

PKT_ACK   = 0x06
# PKT_NACK  = 0x15

PHASE_PACKET_SIZE = 13
DONE_PACKET_SIZE  = 5

# -------------------------------
# CRC-8 (matches MCU)
# -------------------------------
def crc8(data: bytes) -> int:
    crc = 0xFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x31) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc

# -------------------------------
# SEQUENCE CHECK
# -------------------------------
def CheckSequence(expected, actual):
    if expected != actual:
        return False
    return True

def SendAck(ser):
    ser.write(bytes([PKT_ACK]))
    ser.flush()

# -------------------------------
# SEND RUN COMMAND
# -------------------------------
p0_float  = 0
q0_float  =  1
stepSize  = 3276        # Q16.16
numSteps  = 500

p0_q16 = int(p0_float * 65536)
q0_q16 = int(q0_float * 65536)

payload = struct.pack("<i I i i", stepSize, numSteps, p0_q16, q0_q16)
packet  = struct.pack("<BBB", 0xAA, 0x01, len(payload)) + payload

ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
time.sleep(2)
ser.reset_input_buffer()
ser.write(packet)
timeStart = time.time()

print("RUN sent")

# -------------------------------
# RECEIVE DATA
# -------------------------------
buffer = bytearray()
records = []
done = False
expectedSequence = 0

while not done:
    buffer.extend(ser.read(256))

    while True:
        if len(buffer) < 2:
            break

        if buffer[0] != PKT_START:
            buffer.pop(0)
            continue

        pkt_type = buffer[1]

        # ---------- PHASE ----------
        if pkt_type == PKT_PHASE:
            if len(buffer) < PHASE_PACKET_SIZE:
                break

            pkt = buffer[:PHASE_PACKET_SIZE]
            buffer = buffer[PHASE_PACKET_SIZE:]

            if pkt[2] != 8:
                continue

            if crc8(pkt[1:12]) != pkt[12]:
                print("PHASE CRC error")
                continue

            if not CheckSequence( expectedSequence, pkt[3]):
                print(f"PHASE sequence error. Expected {expectedSequence}, got {pkt[3]}")
                continue

            print(f"PHASE seq {pkt[3]} received")
            SendAck(ser)
            expectedSequence += 1
            if expectedSequence > 255:
                expectedSequence = 0

            seq = pkt[3]
            p_raw = struct.unpack("<i", pkt[4:8])[0]
            q_raw = struct.unpack("<i", pkt[8:12])[0]

            records.append({
                "seq": seq,
                "p_raw": p_raw,
                "q_raw": q_raw,
                "p": p_raw / 65536.0,
                "q": q_raw / 65536.0,
            })
            continue

        # ---------- DONE ----------
        if pkt_type == PKT_DONE:
            if len(buffer) < DONE_PACKET_SIZE:
                break

            pkt = buffer[:DONE_PACKET_SIZE]
            buffer = buffer[DONE_PACKET_SIZE:]

            if crc8(pkt[1:4]) != pkt[4]:
                print("DONE CRC error")
                continue

            if not CheckSequence(expectedSequence, pkt[3]):
                print(f"DONE sequence error. Expected {expectedSequence}, got {pkt[3]}")
                continue

            SendAck(ser)

            print("DONE received")
            done = True
            endTime = time.time()
            timeElapsed = endTime - timeStart
            print(f"Time elapsed: {timeElapsed:.2f} seconds")
            print(f"Time per step: {timeElapsed / 500:.6f} seconds")
            break

        buffer.pop(0)

ser.close()

# -------------------------------
# SAVE + PLOT
# -------------------------------
df = pd.DataFrame(records)
df.to_csv("/Users/matthewobrien/Documents/sympnet_debug.csv", index=False)
print(f"Saved {len(df)} points")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(df["q"], df["p"])
plt.axis("equal")
plt.grid(True)
plt.title("Embedded Symplectic Inference Phase Space")
plt.xlabel("q")
plt.ylabel("p")

plt.subplot(1,2,2)
plt.plot(df["p"], label="p")
plt.plot(df["q"], label="q")
plt.legend()
plt.grid(True)
plt.title("Time Series")
plt.xlabel("Step")
plt.ylabel("Value")

plt.tight_layout()
plt.savefig("/Users/matthewobrien/Documents/sympnet.svg")
plt.show()
