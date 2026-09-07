// Standalone validation of the OpenCL f32->bf16 conversion across the full
// value range, compared against the ggml reference (round-to-nearest-even).
// Usage: builds the f32_bf16 kernel from cpy.cl and checks every input value.
#include <CL/cl.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>

static uint16_t ref_bf16(float f) {
    uint32_t i;
    memcpy(&i, &f, 4);
    if ((i & 0x7fffffffU) > 0x7f800000U) return (uint16_t)((i >> 16) | 64U);
    return (uint16_t)((i + (0x7fffU + ((i >> 16) & 1U))) >> 16);
}

static std::string read_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::string s(sz, '\0');
    fread(&s[0], 1, sz, f);
    fclose(f);
    return s;
}

int main(int argc, char ** argv) {
    const char * kernpath = argc > 1 ? argv[1] : "ggml/src/ggml-opencl/kernels/cpy.cl";

    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    err = clGetPlatformIDs(1, &platform, NULL); if (err) { printf("no platform %d\n", err); return 1; }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL); if (err) { printf("no device %d\n", err); return 1; }
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue q = clCreateCommandQueue(ctx, device, 0, &err);

    // test values: edge cases + rounding boundaries + sweep
    std::vector<float> vals;
    const float special[] = {0.0f, -0.0f, 1.0f, -1.0f, 2.0f, 127.0f, 128.0f, 255.0f, 256.0f,
                             108.25f, 108.5f, 108.75f, -108.25f, 150.0f, -150.0f,
                             1.0e-38f, 3.0e38f, -3.0e38f, 0x1p-8f, 0x1p-9f,
                             NAN, INFINITY, -INFINITY, 3.4028235e38f};
    for (float v : special) vals.push_back(v);
    for (int i = 0; i < 10000; i++) {
        float v = (float)(rand() % 20001 - 10000) / 100.0f; // -100.0..100.0
        vals.push_back(v);
    }
    // denser sweep near rounding boundaries
    for (int i = -2000; i <= 2000; i++) {
        float base = i * 0.25f;
        for (float d : {0.0f, 0.25f, 0.5f, 0.75f}) vals.push_back(base + d);
    }
    size_t n = vals.size();

    cl_mem src = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n*sizeof(float), NULL, &err);
    cl_mem dst = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n*sizeof(uint16_t), NULL, &err);
    clEnqueueWriteBuffer(q, src, CL_TRUE, 0, n*sizeof(float), vals.data(), 0, NULL, NULL);

    std::string src_str = read_file(kernpath);
    const char * cs = src_str.c_str();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &cs, NULL, &err);
    err = clBuildProgram(prog, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192]; size_t sz; clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, &sz);
        fprintf(stderr, "build fail %d: %s\n", err, log); return 1;
    }
    cl_kernel k = clCreateKernel(prog, "kernel_cpy_f32_bf16", &err);
    if (err != CL_SUCCESS) { printf("kernel fail %d\n", err); return 1; }

    int ne00 = (int)n, one = 1; cl_ulong nb = sizeof(float), nb2 = 2, zero = 0;
    int arg = 0;
    clSetKernelArg(k, arg++, sizeof(cl_mem), &src);
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &zero);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &dst);
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &zero);
    clSetKernelArg(k, arg++, sizeof(int), &ne00); // ne00
    clSetKernelArg(k, arg++, sizeof(int), &one);  // ne01
    clSetKernelArg(k, arg++, sizeof(int), &one);  // ne02
    clSetKernelArg(k, arg++, sizeof(int), &one);  // ne03
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb);  // nb00
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb);  // nb01
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb);  // nb02
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb);  // nb03
    clSetKernelArg(k, arg++, sizeof(int), &ne00); // ne0
    clSetKernelArg(k, arg++, sizeof(int), &one);  // ne1
    clSetKernelArg(k, arg++, sizeof(int), &one);  // ne2
    clSetKernelArg(k, arg++, sizeof(int), &one);  // ne3
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb2); // nb0
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb2); // nb1
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb2); // nb2
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb2); // nb3

    size_t gws = (n + 63) / 64 * 64, lws = 64;
    clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
    std::vector<uint16_t> out(n);
    clEnqueueReadBuffer(q, dst, CL_TRUE, 0, n*sizeof(uint16_t), out.data(), 0, NULL, NULL);
    clFinish(q);

    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        uint16_t want = ref_bf16(vals[i]);
        if (out[i] != want) {
            if (bad < 20) {
                union { float f; uint32_t u; } x; x.f = vals[i];
                printf("MISMATCH[%zu] in=0x%08x (%.9g) want=0x%04x got=0x%04x\n",
                    i, x.u, (double)vals[i], want, out[i]);
            }
            bad++;
        }
    }
    printf("checked %zu values, mismatches=%d\n", n, bad);
    return bad ? 1 : 0;
}
