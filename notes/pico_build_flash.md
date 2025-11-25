# Building & Flashing for Raspberry Pi Pico

## Prerequisites

- **Install the Pico SDK** and set the `PICO_SDK_PATH` environment variable.
- **Install CMake** and a suitable ARM toolchain (e.g., `arm-none-eabi-gcc`).
- **Connect your Pico in bootloader mode** (hold BOOTSEL while plugging in).

## Build Steps

1. Change to project root:
   ```sh
   cd embeddedResearchML
   ```
2. Remove previous build:
   ```sh
   rm -rf build
   ```
3. Create build directory:
   ```sh
   mkdir build
   cd build
   ```
4. Configure with CMake:
   ```sh
   cmake ..
   ```
5. Build:
   ```sh
   make -j4
   ```

## Flashing to Pico

- Drag the `.elf` file in the `build` folder to the Pico drive (bootloader mode).

## Check Firmware Size

- In the `build` folder:
   ```sh
   arm-none-eabi-size -A pico_main.elf
   ```
