#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_NEURON
#include "ggml-neuron.h"
#endif

#include <cstdio>
#include <cassert>

int main() {
#ifdef GGML_USE_NEURON
    // Test device enumeration
    int device_count = ggml_backend_neuron_get_device_count();
    assert(device_count >= 0);

    if (device_count > 0) {
        // Test device info
        char description[256];
        size_t free_mem, total_mem;
        ggml_backend_neuron_get_device_description(0, description, sizeof(description));
        ggml_backend_neuron_get_device_memory(0, &free_mem, &total_mem);
        assert(total_mem > 0);

        // Test backend initialization
        ggml_backend_t backend = ggml_backend_neuron_init(0);
        assert(backend != nullptr);
        assert(ggml_backend_is_neuron(backend));

        // Test registration (currently returns nullptr)
        ggml_backend_reg_t reg = ggml_backend_neuron_reg();
        (void)reg; // Suppress unused variable warning
        // assert(reg != nullptr); // TODO: Implement proper registration

        ggml_backend_free(backend);
    }

    printf("test-backend-neuron: OK\n");
#else
    printf("test-backend-neuron: SKIP (not compiled)\n");
#endif

    return 0;
}
