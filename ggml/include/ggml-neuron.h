#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_NEURON_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_neuron_init(int device);

GGML_BACKEND_API bool ggml_backend_is_neuron(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_neuron_buffer_type(int device);

GGML_BACKEND_API int  ggml_backend_neuron_get_device_count(void);
GGML_BACKEND_API void ggml_backend_neuron_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_neuron_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_neuron_reg(void);

// internal
GGML_BACKEND_API ggml_guid_t ggml_backend_neuron_guid(void);

#ifdef __cplusplus
}
#endif
