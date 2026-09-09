// Conversion test for the q4_K/q5_K/q6_K SoA weight transposes used by the
// Adreno GEMM path. Reproduces the shapes that fail test-backend-ops MUL_MAT
// (small N, large M, K=1024): builds the MUL_MAT on OpenCL (which runs the
// quantized->SoA conversion + transpose in ggml_backend_opencl_buffer_set_tensor)
// and on CPU (standard dequant), then compares.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opencl.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static void fill_rand_f32(float * p, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; ++i) p[i] = ((float) rand() / RAND_MAX) * 2.0f - 1.0f;
}

static double nmse(const float * a, const float * b, size_t n) {
    double sab = 0, sa2 = 0;
    for (size_t i = 0; i < n; ++i) { double d = a[i] - b[i]; sab += d*d; sa2 += (double) a[i]*a[i]; }
    return sa2 > 0 ? sab / sa2 : 0;
}

static ggml_tensor * build_graph(ggml_context * ctx, ggml_type t, int K, int M, int N,
                                 ggml_tensor ** w, ggml_tensor ** x) {
    *w = ggml_new_tensor_2d(ctx, t, K, M);
    *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    return ggml_mul_mat(ctx, *w, *x);
}

// Replicate mul_mv_q4_k_f32's exact algorithm for output[row][token] (all lanes summed).
static float replicate_generic_q4k(const uint8_t * wgt, const float * x, int K, int M, int N, int row, int token) {
    // standard q4_K: row r at wgt + r*(K/256)*sizeof(block_q4_K); block = 144 bytes
    const size_t blk_bytes = 144;
    float sum = 0.0f;
    for (int ib = 0; ib < K/256; ++ib) {
        const uint8_t * b = wgt + ((size_t) row*(K/256) + ib)*blk_bytes;
        const uint16_t * dbytes = (const uint16_t *)(b + 0);
        const uint16_t * dmbytes = (const uint16_t *)(b + 2);
        const float d  = ggml_fp16_to_fp32(*dbytes);
        const float dm = ggml_fp16_to_fp32(*dmbytes);
        const uint8_t * scales = b + 4;
        const uint8_t * qs = b + 16;
        const float * y = x + (size_t) ib*256 + (size_t) token*K;
        for (int iq = 0; iq < 2; ++iq) {
            for (int ir = 0; ir < 4; ++ir) {
                const float * y4 = y + 64*iq + 8*ir;
                float yl[16], yh[16];
                float ss[4] = {0,0,0,0};
                for (int i = 0; i < 8; ++i) {
                    yl[i+0] = y4[i+0]; ss[0] += yl[i+0];
                    yl[i+8] = y4[i+32]; ss[1] += yl[i+8];
                    yh[i+0] = y4[i+128]; ss[2] += yh[i+0];
                    yh[i+8] = y4[i+160]; ss[3] += yh[i+8];
                }
                const uint16_t * sc = (const uint16_t *)(scales) + iq;
                const uint16_t * q1 = (const uint16_t *)(qs) + 16*iq + 4*ir;
                const uint16_t * q2 = q1 + 32;
                uint16_t sc16[4];
                uint8_t * sc8 = (uint8_t *)sc16;
                sc16[0] = sc[0] & 0x3f3f;
                sc16[1] = sc[2] & 0x3f3f;
                sc16[2] = ((sc[4] >> 0) & 0x0f0f) | ((sc[0] & 0xc0c0) >> 2);
                sc16[3] = ((sc[4] >> 4) & 0x0f0f) | ((sc[2] & 0xc0c0) >> 2);
                float a1[4] = {0,0,0,0}, a2[4] = {0,0,0,0};
                for (int i = 0; i < 8; i += 2) {
                    a1[0] += yl[i+0] * (q1[i/2] & 0x000F);
                    a1[1] += yl[i+1] * (q1[i/2] & 0x0F00);
                    a1[2] += yl[i+8] * (q1[i/2] & 0x00F0);
                    a1[3] += yl[i+9] * (q1[i/2] & 0xF000);
                    a2[0] += yh[i+0] * (q2[i/2] & 0x000F);
                    a2[1] += yh[i+1] * (q2[i/2] & 0x0F00);
                    a2[2] += yh[i+8] * (q2[i/2] & 0x00F0);
                    a2[3] += yh[i+9] * (q2[i/2] & 0xF000);
                }
                sum += d * ((a1[0] + 1.f/256.f*a1[1]) * sc8[0] +
                           (a1[2] + 1.f/256.f*a1[3]) * sc8[1] * 1.f/16.f +
                           (a2[0] + 1.f/256.f*a2[1]) * sc8[4] +
                           (a2[2] + 1.f/256.f*a2[3]) * sc8[5] * 1.f/16.f) -
                       dm * (ss[0]*sc8[2] + ss[1]*sc8[3] + ss[2]*sc8[6] + ss[3]*sc8[7]);
            }
        }
    }
    return sum;
}

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t cl  = ggml_backend_opencl_init();
    if (!cpu || !cl) { printf("backend init failed\n"); return 1; }

    struct { ggml_type t; int K, M, N; const char * name; } cases[] = {
        { GGML_TYPE_Q4_K, 1024, 6272, 2, "q4_K [1024,6272] x [1024,2]" },
        { GGML_TYPE_Q5_K, 1024, 4096, 2, "q5_K [1024,4096] x [1024,2]" },
        { GGML_TYPE_Q6_K, 1024, 4096, 2, "q6_K [1024,4096] x [1024,2]" },
        { GGML_TYPE_Q4_K, 1024, 64,   2, "q4_K [1024,64] x [1024,2]" },
        { GGML_TYPE_Q4_K, 1024, 128,  2, "q4_K [1024,128] x [1024,2]" },
        { GGML_TYPE_Q4_K, 1024, 256,  2, "q4_K [1024,256] x [1024,2]" },
        { GGML_TYPE_Q4_K, 1024, 512,  2, "q4_K [1024,512] x [1024,2]" },
        { GGML_TYPE_Q4_K, 1024, 768,  2, "q4_K [1024,768] x [1024,2]" },
        { GGML_TYPE_Q4_K, 1024, 1024, 2, "q4_K [1024,1024] x [1024,2]" },
        { GGML_TYPE_Q4_K, 1024, 2048, 2, "q4_K [1024,2048] x [1024,2]" },
        { GGML_TYPE_Q4_K, 1024, 4096, 2, "q4_K [1024,4096] x [1024,2] (flat)" },
        { GGML_TYPE_Q4_K, 1024, 8192, 2, "q4_K [1024,8192] x [1024,2]" },
        { GGML_TYPE_Q4_K, 2048, 1024, 2, "q4_K [2048,1024] x [2048,2]" },
        { GGML_TYPE_Q4_K, 2048, 4096, 2, "q4_K [2048,4096] x [2048,2] (model, flat)" },
        { GGML_TYPE_Q4_K, 512,  6272, 2, "q4_K [512,6272] x [512,2]" },
        { GGML_TYPE_Q6_K, 256,  16,    2, "q6_K [256,16] x [256,2] (small)" },
    };

    int failures = 0;
    for (auto & c : cases) {
        fprintf(stderr, "%s: start\n", c.name);
        ggml_init_params ip = { 64*1024*1024, NULL, true };
        const int64_t nelems = (int64_t) c.K * c.M;

        std::vector<float> src((size_t) nelems);
        fill_rand_f32(src.data(), src.size(), 1);
        std::vector<uint8_t> wgt((size_t) ggml_row_size(c.t, c.K) * c.M);
        fprintf(stderr, "%s: src/wgt alloc done (nelems=%lld)\n", c.name, (long long) nelems);
        ggml_quantize_init(c.t);
        size_t qsz = ggml_quantize_chunk(c.t, src.data(), wgt.data(), 0, c.M, c.K, nullptr);
        fprintf(stderr, "%s: quantized %zu bytes\n", c.name, qsz);

        std::vector<float> xin((size_t) c.K * c.N);
        fill_rand_f32(xin.data(), xin.size(), 2);

        // OpenCL
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * w, * x;
        ggml_tensor * out_cl = build_graph(ctx, c.t, c.K, c.M, c.N, &w, &x);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_cl);
        printf("%s: graph built\n", c.name);
        ggml_backend_buffer_t cl_buf = ggml_backend_alloc_ctx_tensors(ctx, cl);
        printf("%s: cl alloc done\n", c.name);
        if (!cl_buf) { printf("opencl alloc failed\n"); return 1; }
        ggml_backend_tensor_set(w, wgt.data(), 0, ggml_nbytes(w));
        printf("%s: cl set w done\n", c.name);
        // Round-trip the conversion: read the SoA back as standard q4_K/q5_K/q6_K
        // and compare the dequant against the original standard dequant.
        std::vector<uint8_t> rt(ggml_nbytes(w));
        ggml_backend_tensor_get(w, rt.data(), 0, ggml_nbytes(w));
        std::vector<float> dq_orig((size_t) nelems), dq_rt((size_t) nelems);
        ggml_get_type_traits(c.t)->to_float(wgt.data(), dq_orig.data(), nelems);
        ggml_get_type_traits(c.t)->to_float(rt.data(),  dq_rt.data(),  nelems);
        float rtmax = 0; size_t rtmaxi = 0;
        for (size_t i = 0; i < dq_orig.size(); ++i) {
            float d = fabsf(dq_orig[i] - dq_rt[i]);
            if (d > rtmax) { rtmax = d; rtmaxi = i; }
        }
        printf("%s: roundtrip maxdiff=%.6f worst[i=%zu] orig=%.6f rt=%.6f %s\n",
               c.name, rtmax, rtmaxi, dq_orig[rtmaxi], dq_rt[rtmaxi], rtmax < 0.05f ? "OK" : "FAIL");
        ggml_backend_tensor_set(x, xin.data(), 0, ggml_nbytes(x));
        ggml_backend_graph_compute(cl, gf);
        printf("%s: cl compute done\n", c.name);
        std::vector<float> ocl(ggml_nelements(out_cl));
        ggml_backend_tensor_get(out_cl, ocl.data(), 0, ggml_nbytes(out_cl));

        // CPU reference (standard dequant)
        ggml_context * ctxc = ggml_init(ip);
        ggml_tensor * wc, * xc;
        ggml_tensor * out_cpu = build_graph(ctxc, c.t, c.K, c.M, c.N, &wc, &xc);
        ggml_cgraph * gfc = ggml_new_graph(ctxc);
        ggml_build_forward_expand(gfc, out_cpu);
        ggml_backend_buffer_t cpu_buf = ggml_backend_alloc_ctx_tensors(ctxc, cpu);
        ggml_backend_tensor_set(wc, wgt.data(), 0, ggml_nbytes(wc));
        ggml_backend_tensor_set(xc, xin.data(), 0, ggml_nbytes(xc));
        ggml_backend_graph_compute(cpu, gfc);
        std::vector<float> ocpu(ggml_nelements(out_cpu));
        ggml_backend_tensor_get(out_cpu, ocpu.data(), 0, ggml_nbytes(out_cpu));

        double e = nmse(ocl.data(), ocpu.data(), ocl.size());
        float maxd = 0; size_t maxi = 0;
        for (size_t i = 0; i < ocl.size(); ++i) {
            float d = fabsf(ocl[i] - ocpu[i]);
            if (d > maxd) { maxd = d; maxi = i; }
        }
        // Half-accumulation reference: does the GPU match f16 rounding, or is it a miscompile?
        if (c.t == GGML_TYPE_Q4_K && c.N >= 1) {
            std::vector<float> wf2((size_t) nelems);
            ggml_get_type_traits(c.t)->to_float(wgt.data(), wf2.data(), nelems);
            for (int rr = 0; rr < 4 && rr < c.M; ++rr) {
                float rep = replicate_generic_q4k(wgt.data(), xin.data(), c.K, c.M, c.N, rr, 0);
                float g = ocl[(size_t) rr + 0*c.M];
                float cp = ocpu[(size_t) rr + 0*c.M];
                // dequantize_row_q4_K + f32 dot (true reference, no q8_K activation loss)
                float ref = 0.0f;
                for (int ki = 0; ki < c.K; ++ki) ref += wf2[(size_t) rr*c.K + ki] * xin[(size_t) ki*c.N + 0];
                printf("%s: row%d kernel-rep=%.6f gpu=%.6f cpuq8k=%.6f dq-f32=%.6f (rep-dq=%.6f dq-cpuq8k=%.6f)\n",
                       c.name, rr, rep, g, cp, ref, rep-ref, ref-cp);
            }
        }
        std::vector<float> wf((size_t) nelems);
        ggml_get_type_traits(c.t)->to_float(wgt.data(), wf.data(), nelems);
        float hm = 0; size_t hmaxi = 0; float hmaxv = 0;
        for (int j = 0; j < c.N; ++j) {
            for (int i = 0; i < c.M; ++i) {
                ggml_fp16_t acc = ggml_fp32_to_fp16(0.0f);
                for (int ki = 0; ki < c.K; ++ki) {
                    ggml_fp16_t p = ggml_fp32_to_fp16(wf[(size_t) i*c.K + ki] * xin[(size_t) ki*c.N + j]);
                    acc = ggml_fp32_to_fp16(ggml_fp16_to_fp32(acc) + ggml_fp16_to_fp32(p));
                }
                float hv = ggml_fp16_to_fp32(acc);
                float dd = fabsf(ocl[(size_t) i*c.N + j] - hv);
                if (dd > hm) { hm = dd; hmaxi = (size_t) i*c.N + j; hmaxv = hv; }
            }
        }
        printf("%s: halfacc maxdiff=%.6f worst[i=%zu] ocl=%.6f half=%.6f\n",
               c.name, hm, hmaxi, ocl[hmaxi], hmaxv);
        bool ok = e < 1e-4 && maxd < 0.1f;
        printf("%s: NMSE=%.6f maxdiff=%.6f worst[i=%zu] ocl=%.6f cpu=%.6f %s\n",
               c.name, e, maxd, maxi, ocl[maxi], ocpu[maxi], ok ? "OK" : "FAIL");
        if (!ok) failures++;

        ggml_backend_buffer_free(cpu_buf);
        ggml_free(ctxc);
        ggml_backend_buffer_free(cl_buf);
        ggml_free(ctx);
    }

    ggml_quantize_free();
    printf("conversion test: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}