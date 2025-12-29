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

PHASE_PACKET_SIZE = 10  # [A5][01][p:int32][q:int32]

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

done = False
step = 0

print("Receiving data...")

while not done:
    data = ser.read(256)
    if not data:
        continue

    buffer.extend(data)

    while len(buffer) >= 2:
        if buffer[0] != PKT_START:
            buffer.pop(0)
            continue

        pkt_type = buffer[1]

        if pkt_type == PKT_PHASE:
            if len(buffer) < PHASE_PACKET_SIZE:
                break

            _, _, p_raw, q_raw = struct.unpack(
                "<BBii", buffer[:PHASE_PACKET_SIZE]
            )
            buffer = buffer[PHASE_PACKET_SIZE:]

            records.append({
                "step": step,
                "p_raw": p_raw,
                "q_raw": q_raw,
                "p_q16": p_raw / 65536.0,
                "q_q16": q_raw / 65536.0,
                "p_float127": p_raw / 127.0,
                "q_float127": q_raw / 127.0,
            })

            step += 1
            continue

        if pkt_type == PKT_DONE:
            buffer = buffer[2:]
            done = True
            break

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
