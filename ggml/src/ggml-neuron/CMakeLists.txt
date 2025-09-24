if (NOT DEFINED GGML_NEURON)
    set(GGML_NEURON OFF)
endif()

if (NOT GGML_NEURON)
    return()
endif()

# Find AWS Neuron SDK
# TODO: Add proper FindNeuron.cmake module
# find_package(Neuron REQUIRED)

ggml_add_backend_library(ggml-neuron
    ggml-neuron.cpp
)

target_include_directories(ggml-neuron PRIVATE
    .
    ..
    ../..
    ../../include
)

target_compile_definitions(ggml-neuron PRIVATE GGML_USE_NEURON)

# TODO: Link with actual Neuron SDK libraries
# target_link_libraries(ggml-neuron PRIVATE ${NEURON_LIBRARIES})

if (GGML_NEURON_DEBUG)
    target_compile_definitions(ggml-neuron PRIVATE GGML_NEURON_DEBUG)
endif()
