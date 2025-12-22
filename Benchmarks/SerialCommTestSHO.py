import serial
import struct
import time

DISPLACEMENT_MAX = 1.5
DISPLACEMENT_MIN = -1.5

serialPort = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

x0 = float(input("Enter initial displacement x value within [-1.5, 1.5]: "))
if x0 < DISPLACEMENT_MIN or x0 > DISPLACEMENT_MAX:
    raise ValueError("x0 must be within the range [-1.5, 1.5]")

x0_scale = 127 / DISPLACEMENT_MAX
x0_int8 = int(round(x0 * x0_scale))

dataBytes = bytearray(struct.pack('b', x0_int8))
serialPort.write(dataBytes)
timeStart = time.time()

response = serialPort.read(2)
timeEnd = time.time()

