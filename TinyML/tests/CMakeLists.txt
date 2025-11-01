file(GLOB TEST_SOURCES "*.cpp") # or *.c if your tests are in C

add_executable(tinyml_tests ${TEST_SOURCES})
target_link_libraries(tinyml_tests PRIVATE tinyml_core)
# If Unity is in unity/src/unity.c:
target_sources(tinyml_tests PRIVATE ../unity/src/unity.c)