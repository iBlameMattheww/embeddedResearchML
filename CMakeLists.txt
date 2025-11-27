cmake_minimum_required(VERSION 3.13)

# Import the Pico SDK
include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(embeddedResearchML C CXX ASM)

# Initialize the SDK
pico_sdk_init()

add_executable(pico_main
    TinyML/Target/src/main.c
    TinyML/Target/src/heartbeat.c
)


target_include_directories(pico_main PRIVATE
    TinyML/Target/include
    TinyML/Target/include/exported_weights
    TinyML/Target/src
    TinyML/Target/src/models
    TinyML/Target/tools
)

target_link_libraries(pico_main pico_stdlib)

pico_enable_stdio_usb(pico_main 1)
pico_enable_stdio_uart(pico_main 0)

pico_add_extra_outputs(pico_main)
