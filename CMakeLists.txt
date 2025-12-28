cmake_minimum_required(VERSION 3.13)

# Import the Pico SDK
include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(embeddedResearchML C CXX ASM)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialize the SDK
pico_sdk_init()

add_executable(pico_main
    TinyML/Target/src/SNN_Main.c
    TinyML/Target/src/Heartbeat.c
    TinyML/Target/src/Serial.c
    TinyML/Target/src/SymplecticInference.c
    TinyML/Target/src/models/Sympnet.c
    TinyML/Target/tools/USB_Descriptors.c
)


target_include_directories(pico_main PRIVATE
    TinyML/Target/include
    TinyML/Target/include/SNN_ExportedParams
    TinyML/Target/src
    TinyML/Target/src/Models
    TinyML/Target/tools
)

target_link_libraries(pico_main pico_stdlib tinyusb_device tinyusb_board)

pico_enable_stdio_usb(pico_main 0)
pico_enable_stdio_uart(pico_main 0)

pico_add_extra_outputs(pico_main)
