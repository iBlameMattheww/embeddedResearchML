import serial
import struct
import time
import numpy as np
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
START_BYTE = 0xAA
CMD_RUN    = 0x01

PKT_START  = 0xA5
PKT_PHASE  = 0x01
PKT_DONE   = 0xFF

PHASE_PACKET_SIZE = 13  # [A5][01][seq][p:int32][q:int32][crc]
DONE_PACKET_SIZE  = 5   # [A5][FF][1][seq]

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
# TEST PARAMETERS
# -------------------------------
p0_float  = -1.0
q0_float  =  0.5
stepSize  = 3276        # 0.05 in Q16.16
numSteps  = 500

p0_q16 = int(p0_float * 65536)
q0_q16 = int(q0_float * 65536)


# -------------------------------
# BUILD RUN PACKET
# -------------------------------
payload = struct.pack(
    "<i I i i",
    stepSize,
    numSteps,
    p0_q16,
    q0_q16
)

packet = struct.pack(
    "<BBB",
    START_BYTE,
    CMD_RUN,
    len(payload)
) + payload

print("Sending RUN command")
print(f"p0={p0_float}, q0={q0_float}, h={stepSize}, steps={numSteps}")

# -------------------------------
# OPEN SERIAL
# -------------------------------
ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
time.sleep(2)
ser.reset_input_buffer()
ser.write(packet)

# -------------------------------
# RECEIVE DATA
# -------------------------------
buffer = bytearray()
records = []
expected_seq = None
done = False
step = 0

print("Receiving data...")

while not done:
    data = ser.read(256)
    if not data:
        continue

    buffer.extend(data)

    while True:
        if len(buffer) < 5:
            break

        if buffer[0] != PKT_START:
            buffer.pop(0)
            continue

        pkt_type = buffer[1]

        # ---------------- PHASE ----------------
        if pkt_type == PKT_PHASE:
            if len(buffer) < PHASE_PACKET_SIZE:
                break

            pkt = buffer[:PHASE_PACKET_SIZE]
            buffer = buffer[PHASE_PACKET_SIZE:]

            if pkt[2] != 0x0C:
                print("Invalid payload length, dropping packet")
                continue

            length = pkt[2]
            seq    = pkt[3]
            crc_rx = pkt[12]
            crc_ok = crc8(pkt[1:12])

            if crc_rx != crc_ok:
                print("CRC error (PHASE), dropping packet")
                continue

            if expected_seq is None:
                expected_seq = seq
            elif seq != expected_seq:
                print(f"Sequence jump: got {seq}, expected {expected_seq}")
                expected_seq = seq

            expected_seq = (seq + 1) & 0xFF

            p_raw = struct.unpack("<i", pkt[4:8])[0]
            q_raw = struct.unpack("<i", pkt[8:12])[0]

            records.append({
                "step": step,
                "seq": seq,
                "p_raw": p_raw,
                "q_raw": q_raw,
                "p_q16": p_raw / 65536.0,
                "q_q16": q_raw / 65536.0,
            })

            step += 1
            continue

        # ---------------- DONE ----------------
        if pkt_type == PKT_DONE:
            if len(buffer) < DONE_PACKET_SIZE:
                break

            pkt = buffer[:DONE_PACKET_SIZE]
            buffer = buffer[DONE_PACKET_SIZE:]

            crc_rx = pkt[4]
            crc_ok = crc8(pkt[1:4])

            if crc_rx != crc_ok:
                print("CRC error (DONE)")
                continue

            done = True
            break

        # Unknown → resync
        buffer.pop(0)


ser.close()

# -------------------------------
# DATAFRAME + CSV
# -------------------------------
df = pd.DataFrame(records)
df.to_csv("/Users/matthewobrien/Documents/sympnet_debug.csv", index=False)

print(f"Saved {len(df)} points to /Users/matthewobrien/Documents/sympnet_debug.csv")

# -------------------------------
# PLOTS
# -------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(df["q_q16"], df["p_q16"])
plt.xlabel("q")
plt.ylabel("p")
plt.title("Phase Space (Q16)")
plt.axis("equal")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(df["p_q16"], label="p")
plt.plot(df["q_q16"], label="q")
plt.legend()
plt.title("Time Series (Q16)")
plt.grid(True)

plt.tight_layout()
plt.show()
