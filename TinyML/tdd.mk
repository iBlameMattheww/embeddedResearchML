# ---------- Compiler and Flags ----------
CC       := g++
AR       := ar
CFLAGS   := -std=c++17 -Wall -Wextra -O0 -g

# ---------- Include Paths ----------
INCLUDES := \
    -Iunity/src \
    -ITarget/src \
    -ITarget/include \
    -ITarget/include/exported_weights \
    -ITarget/src/models \
    -ITarget/tools

# ---------- Directories ----------
UNITY_DIR  := unity/src
SRC_DIR    := Target/src
MODEL_DIR  := Target/src/models
TEST_DIR   := tests
BUILD_DIR  := build

# ---------- Sources ----------
UNITY_SRC  := $(wildcard $(UNITY_DIR)/*.c)

# Explicitly exclude main.c (Unity has its own main)
SRC_FILES  := \
    $(SRC_DIR)/Activations.c \
    $(SRC_DIR)/SymplecticInference.c \
    $(MODEL_DIR)/Sympnet.c \

TEST_FILES := $(wildcard $(TEST_DIR)/*.cpp)

# ---------- Outputs ----------
UNITY_LIB  := $(BUILD_DIR)/libunity.a
OUT        := $(BUILD_DIR)/tests

# ---------- Targets ----------
.PHONY: all run clean unity_lib

all: $(OUT)

# --- Build Unity as a static library ---
$(UNITY_LIB): $(UNITY_SRC)
	@mkdir -p $(BUILD_DIR)
	gcc -c $(UNITY_SRC) $(INCLUDES) -o $(BUILD_DIR)/unity.o
	$(AR) rcs $@ $(BUILD_DIR)/unity.o

# --- Build and link test executable ---
$(OUT): $(SRC_FILES) $(TEST_FILES) $(UNITY_LIB)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(SRC_FILES) $(TEST_FILES) $(UNITY_LIB)

run: all
	./$(OUT)

clean:
	rm -rf $(BUILD_DIR)
