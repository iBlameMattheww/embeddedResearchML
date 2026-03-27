cmake_minimum_required(VERSION 3.13)

# ------------------------------------------------------------
# Import Pico SDK
# ------------------------------------------------------------
include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(embeddedResearchML C CXX ASM)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialize the SDK
pico_sdk_init()

# ------------------------------------------------------------
# Common sources (shared by all firmwares)
# ------------------------------------------------------------
set(COMMON_SOURCES
    TinyML/Target/src/Heartbeat.c
    TinyML/Target/src/Serial.c
    TinyML/Target/tools/usb_descriptors.c
)

# ------------------------------------------------------------
# Symplectic Neural Net (SNN) sources
# ------------------------------------------------------------
set(SNN_SOURCES
    TinyML/Target/src/SNN_Main.c
    TinyML/Target/src/SymplecticInference.c
    TinyML/Target/src/models/Sympnet.c
)

# ------------------------------------------------------------
# Vanilla Neural Net (VNN) sources
# ------------------------------------------------------------
set(VNN_SOURCES
    TinyML/Target/src/VNN_Main.c
    TinyML/Target/src/VanillaInference.c
    TinyML/Target/src/models/Activations.c
)

# ------------------------------------------------------------
# Physics Informed Neural Net (PINN) sources
# ------------------------------------------------------------
set(PINN_SOURCES
    TinyML/Target/src/PINN_Main.c
    TinyML/Target/src/PINN_Inference.c
    TinyML/Target/src/models/PINN.c
)

# ------------------------------------------------------------
# Symplectic firmware target
# ------------------------------------------------------------
add_executable(pico_snn
    ${COMMON_SOURCES}
    ${SNN_SOURCES}
)

target_include_directories(pico_snn PRIVATE
    TinyML/Target/include
    TinyML/Target/include/SNN_ExportedParams
    TinyML/Target/src
    TinyML/Target/src/models
    TinyML/Target/tools
)

target_link_libraries(pico_snn
    pico_stdlib
    tinyusb_device
    tinyusb_board
)

pico_enable_stdio_usb(pico_snn 1)
pico_enable_stdio_uart(pico_snn 0)

pico_add_extra_outputs(pico_snn)

# ------------------------------------------------------------
# Vanilla firmware target
# ------------------------------------------------------------
add_executable(pico_vnn
    ${COMMON_SOURCES}
    ${VNN_SOURCES}
)

target_include_directories(pico_vnn PRIVATE
    TinyML/Target/include
    TinyML/Target/include/VNN_ExportedParams
    TinyML/Target/src
    TinyML/Target/src/models
    TinyML/Target/tools
)

target_link_libraries(pico_vnn
    pico_stdlib
    tinyusb_device
    tinyusb_board
)

pico_enable_stdio_usb(pico_vnn 1)
pico_enable_stdio_uart(pico_vnn 0)

pico_add_extra_outputs(pico_vnn)

# ------------------------------------------------------------
# PINN firmware target
# ------------------------------------------------------------
add_executable(pico_pinn
    ${COMMON_SOURCES}
    ${PINN_SOURCES}
)

target_include_directories(pico_pinn PRIVATE
    TinyML/Target/include
    TinyML/Target/include/PINN_ExportedParams
    TinyML/Target/src
    TinyML/Target/src/models
    TinyML/Target/tools
)

target_link_libraries(pico_pinn
    pico_stdlib
    tinyusb_device
    tinyusb_board
)

pico_enable_stdio_usb(pico_pinn 1)
pico_enable_stdio_uart(pico_pinn 0)

pico_add_extra_outputs(pico_pinn)