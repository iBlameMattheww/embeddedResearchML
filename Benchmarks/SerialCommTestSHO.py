import serial
import struct
import time
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# SERIAL CONFIG
# -------------------------------

PORT = "/dev/tty.usbmodem1201"   # adjust if needed
BAUD = 115200
TIMEOUT = 1.0

CDC_ITF = 0

# -------------------------------
# PROTOCOL CONSTANTS (MATCH FW)
# -------------------------------

START_BYTE   = 0xAA
CMD_RUN      = 0x01

PKT_START    = 0xA5
PKT_PHASE    = 0x01
PKT_DONE     = 0xFF

PHASE_PACKET_SIZE = 10   # [A5][01][p:int32][q:int32]
DONE_PACKET_SIZE  = 2    # [A5][FF]

# -------------------------------
# TEST PARAMETERS
# -------------------------------

p0_float  = -1.0
q0_float  =  0.5
stepSize  = int(0.05 * 65536)         # Q16.16 (~0.05) = 3277
numSteps  = 500

# Quantize initial conditions (int8 → FW shifts to Q16)
p0 = int(round(p0_float * 127))
q0 = int(round(q0_float * 127))

# -------------------------------
# BUILD RUN PACKET
# -------------------------------

payload = struct.pack(
    "<i I b b",
    stepSize,
    numSteps,
    p0,
    q0
)


packet = struct.pack(
    "<BBB",
    START_BYTE,
    CMD_RUN,
    len(payload)
) + payload

print("Sending RUN command:")
print(f"  p0={p0_float}, q0={q0_float}")
print(f"  stepSize={stepSize}, numSteps={numSteps}")

# -------------------------------
# OPEN SERIAL
# -------------------------------

ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
time.sleep(2)
ser.reset_input_buffer()

ser.write(packet)

# -------------------------------
# RECEIVE BUFFERED DATA
# -------------------------------

buffer = bytearray()
p_vals = []
q_vals = []

done_received = False

print("\nReceiving buffered trajectory...\n")

while not done_received:
    data = ser.read(256)
    if not data:
        continue

    buffer.extend(data)

    while True:
        if len(buffer) < 2:
            break

        # Search for packet start
        if buffer[0] != PKT_START:
            buffer.pop(0)
            continue

        pkt_type = buffer[1]

        # Phase packet
        if pkt_type == PKT_PHASE:
            if len(buffer) < PHASE_PACKET_SIZE:
                break  # wait for more data

            _, _, p_raw, q_raw = struct.unpack(
                "<BBii", buffer[:PHASE_PACKET_SIZE]
            )

            buffer = buffer[PHASE_PACKET_SIZE:]

            p_vals.append(p_raw / 65536.0)
            q_vals.append(q_raw / 65536.0)
            continue

        # Done packet
        if pkt_type == PKT_DONE:
            buffer = buffer[2:]
            done_received = True
            break

        # Unknown packet → resync
        buffer.pop(0)


ser.close()

# -------------------------------
# RESULTS
# -------------------------------

p_vals = np.array(p_vals)
q_vals = np.array(q_vals)

print(f"\nReceived {len(p_vals)} points")

if len(p_vals) != numSteps:
    print("⚠️ WARNING: Step count mismatch")
else:
    print("✅ Step count matches")

# -------------------------------
# PLOTTING
# -------------------------------

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(q_vals, p_vals)
axs[0].set_xlabel("q")
axs[0].set_ylabel("p")
axs[0].set_title("Phase Space")
axs[0].grid(True)

axs[1].plot(p_vals, label="p")
axs[1].plot(q_vals, label="q")
axs[1].set_xlabel("Step")
axs[1].set_ylabel("Value")
axs[1].set_title("Time Series")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
