import serial
import struct
import time
import os
import pandas as pd
import numpy as np

from tqdm import tqdm

# -------------------------------
# Serial Config and Protocol Constants
# -------------------------------
PORT = "/dev/tty.usbmodem1201"
BAUD = 115200
TIMEOUT = 1.0

PKT_START  = 0xA5
PKT_PHASE  = 0x01
PKT_DONE   = 0xFF
PKT_ACK   = 0x06
PHASE_PACKET_SIZE = 13
DONE_PACKET_SIZE  = 5

# -------------------------------
# Training File Constants
# -------------------------------
TRAINING_DATA_FILE = "SimpleHarmonicOscillator/data/sho_trajectories.npy"
RESULTS_DIR = "Benchmarks/SHO_Results"

def QuantizeToQ16_16(value: float) -> int:
    return int(value * (1 << 16))

def CRC8(data: bytes) -> int:
    crc = 0xFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x31) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc

def CheckSequence(expected, actual):
    if expected != actual:
        return False
    return True

def SetupSerialConnection():
    ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    time.sleep(2)  # Wait for connection to establish
    return ser

def SendAck(ser):
    ser.write(bytes([PKT_ACK]))
    ser.flush()

def DataLoader(path):
    data = np.load(path)
    return data

def CollectFirstFiftyInitialValuesFromDataset(data): 
    initialConditions = []
    for trajectory in data[:50]:
        q0, p0 = trajectory[0]
        initialConditions.append((q0, p0))
    return initialConditions

def RunInferenceTrajectory(index, ser, records, q0, p0, stepSize, numSteps):
    p0_q16_16 = QuantizeToQ16_16(p0)
    q0_q16_16 = QuantizeToQ16_16(q0)

    payload = struct.pack("<i I i i", stepSize, numSteps, p0_q16_16, q0_q16_16)
    packet = struct.pack("<BBB", 0xAA, 0x01, len(payload)) + payload
    ser.write(packet)
    timeStart = time.time()

    buffer = bytearray()
    done = False
    expectedSequence = 0
    stepIndex = 0

    while not done:
        buffer.extend(ser.read(256))

        while True:
            if len(buffer) < 2:
                break

            if buffer[0] != PKT_START:
                buffer.pop(0)
                continue

            pkt_Type = buffer[1]

            if pkt_Type == PKT_PHASE:
                if len(buffer) < PHASE_PACKET_SIZE:
                    break

                pkt = buffer[:PHASE_PACKET_SIZE]
                buffer = buffer[PHASE_PACKET_SIZE:]

                if pkt[2] != 8:
                    continue

                if CRC8(pkt[1:12]) != pkt[12]:
                    continue

                if not CheckSequence(expectedSequence, pkt[3]):
                    continue

                SendAck(ser)
                expectedSequence = (expectedSequence + 1) % 256

                seq = pkt[3]
                p_raw = struct.unpack("<i", pkt[4:8])[0]
                q_raw = struct.unpack("<i", pkt[8:12])[0]

                endTime = time.time()
                timeElapsed = endTime - timeStart

                records.append({
                    "trajectory_index": index,
                    "step_index": stepIndex,
                    "q": q_raw / (1 << 16),
                    "p": p_raw / (1 << 16),
                    "inference_time_sec": timeElapsed
                })
                stepIndex += 1
                continue

            if pkt_Type == PKT_DONE:
                print(f"Receiving DONE packet for trajectory {index}")
                if len(buffer) < DONE_PACKET_SIZE:
                    break

                pkt = buffer[:DONE_PACKET_SIZE]
                buffer = buffer[DONE_PACKET_SIZE:]

                # if CRC8(pkt[1:4]) != pkt[4]:
                #     continue

                # if not CheckSequence(expectedSequence, pkt[3]):
                #     continue

                print(f"DONE packet received for trajectory {index}")
                SendAck(ser)
                done = True
                break

            buffer.pop(0)

def CreateOutputDirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def DataframeToCSV(df):
    outputFile = os.path.join(RESULTS_DIR, "sho_inference_results.csv")
    df.to_csv(outputFile, index=False)
    print(f"Data saved to {outputFile}")

def main():
    data = DataLoader(TRAINING_DATA_FILE)
    initialConditions = CollectFirstFiftyInitialValuesFromDataset(data)

    ser = SetupSerialConnection()
    records = []

    stepSize = QuantizeToQ16_16(0.5)  # dt for 500 steps over one period
    numSteps = 500

    for index, (q0, p0) in enumerate(tqdm(initialConditions, desc="Running Inference Trajectories")):
        RunInferenceTrajectory(index, ser, records, q0, p0, stepSize, numSteps)
        time.sleep(1)  # Small delay between trajectories

    ser.close()

    df = pd.DataFrame.from_records(records)
    print("Inference complete. Sample of results:")
    print(df.head())

    CreateOutputDirectory(RESULTS_DIR)
    DataframeToCSV(df)

if __name__ == "__main__":
    main()