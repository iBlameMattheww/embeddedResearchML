# ----------------------------------------------------
# Usage: make PLATFORM=<rpi|nano33|pico> [target]
# ----------------------------------------------------

PLATFORM ?= rpi

# ----------------------------------------------------
# Common
# ----------------------------------------------------
SRC = TinyML/Target/src/*.c TinyML/Target/src/models/*.c

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


# ====================================================
# PLATFORM: Nano 33 BLE (nRF52840)
# ====================================================
else ifeq ($(PLATFORM),nano33)
CC = arm-none-eabi-gcc
CFLAGS = -mcpu=cortex-m4 -mthumb -Wall -ffreestanding -nostdlib
LDFLAGS =
BIN = $(BUILD)/nano33.elf

$(BUILD)/nano33.hex: $(BIN)
	arm-none-eabi-objcopy -O ihex $(BIN) $(BUILD)/nano33.hex

FLASH = nrfjprog --program $(BUILD)/nano33.hex --chiperase --reset


# ====================================================
# PLATFORM: Raspberry Pi Pico (RP2040)
# ====================================================
else ifeq ($(PLATFORM),pico)
CC = arm-none-eabi-gcc
CFLAGS = -mcpu=cortex-m0plus -mthumb -Wall -nostdlib -ffreestanding
LDFLAGS = -T TinyML/pico/pico_linker.ld -nostdlib
STARTUP = TinyML/pico/pico_startup.S

BIN = $(BUILD)/pico.elf
UF2 = $(BUILD)/pico.uf2

$(UF2): $(BIN)
	python tools/uf2conv.py -o $(UF2) -f RP2040 -b 0x10000000 $(BIN)


FLASH = @echo "Pico: Drag $(UF2) to the RPI-RP2 USB volume"

else
$(error Unknown PLATFORM: $(PLATFORM))
endif


# ====================================================
# Build Rules
# ====================================================
all: $(BUILD) $(BIN)

$(BUILD):
	mkdir -p $(BUILD)

$(BIN): $(SRC)
ifeq ($(PLATFORM),pico)
	$(CC) $(CFLAGS) $(INCLUDES) $(STARTUP) $(SRC) $(LDFLAGS) -lgcc -o $(BIN)
else ifeq ($(PLATFORM),nano33)
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) $(LDFLAGS) -lgcc -o $(BIN)
else
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) -o $(BIN)
endif


# ====================================================
# Flash
# ====================================================
flash: all
ifeq ($(PLATFORM),pico)
	$(MAKE) $(UF2)
endif
ifeq ($(PLATFORM),nano33)
	$(MAKE) $(BUILD)/nano33.hex)
endif
	$(FLASH)


# ====================================================
# Cleanup
# ====================================================
clean:
	rm -rf $(BUILD)

.PHONY: all clean flash
