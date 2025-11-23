# CMakeLists.txt for Pico SDK project
cmake_minimum_required(VERSION 3.13)
include(pico-sdk/pico_sdk_init.cmake)

project(embeddedResearchML C CXX ASM)

# Initialize the SDK
pico_sdk_init()

# Add your source files
set(SOURCES
    TinyML/Target/src/main.c
    TinyML/Target/src/platform/pico/uart_pico.c
    TinyML/Target/src/models/vanillaNNTML.c
    # Add other source files as needed
)

add_executable(pico_main ${SOURCES})

# Link with Pico SDK libraries
pico_add_extra_outputs(pico_main)
target_include_directories(pico_main PRIVATE
    TinyML/Target/include
    TinyML/Target/include/exported_weights
    TinyML/Target/src
    TinyML/Target/src/models
    TinyML/Target/tools
)

target_link_libraries(pico_main pico_stdlib)

# Enable USB, UART, etc. as needed
# target_link_libraries(pico_main pico_stdio_uart pico_stdio_usb)
