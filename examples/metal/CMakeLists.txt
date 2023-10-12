set(TEST_TARGET metal)
add_executable(${TEST_TARGET} metal.cpp)
install(TARGETS ${TARGET} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
