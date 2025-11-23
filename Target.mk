# ----------------------------------------------------
# Usage: make PLATFORM=<rpi|nano33|pico> [target]
# ----------------------------------------------------

PLATFORM ?= rpi

# ----------------------------------------------------
# Common
# ----------------------------------------------------
SRC_COMMON = TinyML/Target/src/*.c TinyML/Target/src/models/*.c

ifeq ($(PLATFORM),pico)
SRC = $(SRC_COMMON) TinyML/Target/src/platform/pico/uart_pico.c
else ifeq ($(PLATFORM),nano33)
SRC = $(SRC_COMMON) TinyML/Target/src/platform/nano33/uart_nano33.c
else
SRC = $(SRC_COMMON)
endif

INCLUDES = \
    -ITinyML/Target/include \
    -ITinyML/Target/include/exported_weights \
    -ITinyML/Target/src \
    -ITinyML/Target/src/models \
    -ITinyML/Target/tools

BUILD = build

# ====================================================
# PLATFORM: Raspberry Pi (Linux host)
# ====================================================
ifeq ($(PLATFORM),rpi)
CC = gcc
CFLAGS = -Wall
BIN = $(BUILD)/main
FLASH = @echo "No flashing for RPi"

# Simple compile
$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) -o $(BIN)


# ====================================================
# PLATFORM: Nano 33 BLE (nRF52840)
# ====================================================
else ifeq ($(PLATFORM),nano33)

CC = arm-none-eabi-gcc
CFLAGS = -mcpu=cortex-m4 -mthumb -Wall -ffreestanding -nostdlib
BIN = $(BUILD)/nano33.elf

$(BIN):
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) -lgcc -o $(BIN)

FLASH = @echo "Use nrfjprog here"


# ====================================================
# PLATFORM: Raspberry Pi Pico (RP2040)
# ====================================================
else ifeq ($(PLATFORM),pico)

CC = arm-none-eabi-gcc
CFLAGS = -mcpu=cortex-m0plus -mthumb -Wall -ffreestanding -nostdlib
LDFLAGS = -T TinyML/pico/pico_linker.ld -nostdlib -lgcc


BIN = $(BUILD)/pico.elf
UF2 = $(BUILD)/pico.uf2

# Simple Pico build (SDK boot/startup assumed)
$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) $(LDFLAGS) -o $(BIN)

# Convert to UF2
$(UF2): $(BIN)
	python tools/uf2conv.py -o $(UF2) -f RP2040 -b 0x10000000 $(BIN)

FLASH = @echo "Pico ready — drag $(UF2) to RPI-RP2 drive"

endif

# ====================================================
# Build Rules
# ====================================================
all: $(BUILD) $(BIN) $(UF2)

$(BUILD):
	mkdir $(BUILD)

# ====================================================
# Flash
# ====================================================
flash: all
	$(FLASH)

# ====================================================
# Cleanup
# ====================================================
clean:
	@if exist $(BUILD) rmdir /s /q $(BUILD)

.PHONY: all clean flash
