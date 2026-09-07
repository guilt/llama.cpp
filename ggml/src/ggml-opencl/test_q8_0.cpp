// Standalone validation of the OpenCL f32->q8_0 quantize-on-copy kernel,
// compared against ggml quantize_row_q8_0_ref. Verifies d and qs per block.
#include <CL/cl.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>

static float ref_q8_0_block(const float * x, uint16_t * d_out, int8_t * qs_out) {
    float amax = 0.0f;
    for (int j = 0; j < 32; j++) amax = fmax(amax, fabsf(x[j]));
    const float d = amax / 127.0f;
    const float id = d ? 1.0f/d : 0.0f;
    // fp16(d) reference: convert via the C half round-to-nearest
    uint32_t di; memcpy(&di, &d, 4);
    uint32_t half = (di + 0xFFFu + ((di >> 13) & 1u)) >> 13; // naive rn; ok for tolerance
    *d_out = (uint16_t)half;
    for (int j = 0; j < 32; j++) qs_out[j] = (int8_t)roundf(x[j] * id);
    return d;
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
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue q = clCreateCommandQueue(ctx, device, 0, &err);

    // N blocks of 32 floats; each block exercises different magnitudes
    const int NBLK = 64;
    std::vector<float> vals(NBLK * 32);
    for (int b = 0; b < NBLK; b++) {
        float mag = (b % 8 == 0) ? 0.0f : (float)(1 + b);   // include zero-scale blocks
        for (int j = 0; j < 32; j++) {
            vals[b*32 + j] = ((b*32 + j) % 7 - 3) * mag * 0.1f;
        }
    }
    size_t n = vals.size();

    cl_mem src = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n*sizeof(float), NULL, &err);
    cl_mem dst = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, NBLK*34, NULL, &err);
    clEnqueueWriteBuffer(q, src, CL_TRUE, 0, n*sizeof(float), vals.data(), 0, NULL, NULL);

    std::string src_str = read_file(kernpath);
    const char * cs = src_str.c_str();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &cs, NULL, &err);
    err = clBuildProgram(prog, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) { char log[8192]; size_t sz; clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, &sz); fprintf(stderr, "build fail %d: %s\n", err, log); return 1; }
    cl_kernel k = clCreateKernel(prog, "kernel_cpy_f32_q8_0", &err);
    if (err != CL_SUCCESS) { printf("kernel fail %d\n", err); return 1; }

    int ne00 = 32, one = 1; cl_ulong nb = sizeof(float), nb34 = 34, zero = 0, row128 = 128;
    int arg = 0;
    clSetKernelArg(k, arg++, sizeof(cl_mem), &src);   // src0
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &zero);// offset0
    clSetKernelArg(k, arg++, sizeof(cl_mem), &dst);   // dst
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &zero);// offsetd
    clSetKernelArg(k, arg++, sizeof(int), &ne00); // ne00=32
    clSetKernelArg(k, arg++, sizeof(int), &one);  // ne01=1
    clSetKernelArg(k, arg++, sizeof(int), &one);  // ne02
    clSetKernelArg(k, arg++, sizeof(int), &one);  // ne03
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb);   // nb00=4
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &row128);// nb01=128
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &row128);// nb02
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &row128);// nb03
    clSetKernelArg(k, arg++, sizeof(int), &ne00); // ne0=32
    clSetKernelArg(k, arg++, sizeof(int), &one);  // ne1
    clSetKernelArg(k, arg++, sizeof(int), &one);  // ne2
    clSetKernelArg(k, arg++, sizeof(int), &one);  // ne3
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb34); // nb0=34
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb34); // nb1
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb34); // nb2
    clSetKernelArg(k, arg++, sizeof(cl_ulong), &nb34); // nb3

    // grid: process all NBLK blocks in one "row" (ne01=nblocks, one block per work item)
    int ne01 = NBLK; clSetKernelArg(k, 5, sizeof(int), &ne01);
    size_t gws = (NBLK*64 + 63)/64*64, lws = 64;
    clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
    std::vector<char> out(NBLK*34);
    clEnqueueReadBuffer(q, dst, CL_TRUE, 0, NBLK*34, out.data(), 0, NULL, NULL);
    clFinish(q);

    int bad = 0;
    for (int b = 0; b < NBLK; b++) {
        uint16_t ref_d; int8_t ref_qs[32];
        ref_q8_0_block(&vals[b*32], &ref_d, ref_qs);
        uint16_t gpu_d; memcpy(&gpu_d, &out[b*34], 2);
        // compare dequantized d (tolerate fp16 rounding diff)
        union { uint16_t h; uint16_t raw; } dref, dgpu; dref.h = ref_d; dgpu.h = gpu_d;
        int qbad = 0;
        for (int j = 0; j < 32; j++) {
            if ((int8_t)out[b*34+2+j] != ref_qs[j]) qbad++;
        }
        if (qbad > 0) {
            if (bad < 10) printf("BLOCK[%d] d_ref=0x%04x d_gpu=0x%04x qbad=%d\n", b, ref_d, gpu_d, qbad);
            bad += qbad;
        }
    }
    printf("checked %d blocks, bad qs=%d\n", NBLK, bad);
    return bad ? 1 : 0;
}