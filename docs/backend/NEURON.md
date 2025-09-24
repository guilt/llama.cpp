# AWS Neuron Backend

AWS Neuron backend for GGML enables inference on AWS Inferentia and Trainium chips.

## Building

```bash
cmake .. -DGGML_NEURON=ON
make -j$(nproc)
```

## Usage

```bash
./llama-cli -m model.gguf --backend neuron
```

## Requirements

- AWS Neuron SDK
- AWS Inferentia or Trainium instance
