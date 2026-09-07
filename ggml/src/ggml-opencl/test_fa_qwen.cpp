// Replicate Qwen3.5/Qwen3.8-2B flash-attn decode call: dk=256, n_head=8,
// n_head_kv=2, head-major Q/K/V cache views, f16 KV, f32 Q, f16 mask.
// Compares OpenCL vs CPU backend output.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opencl.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

static void fill_random(char * p, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; ++i) p[i] = (char) (rand() & 0xff);
}

static void fill_f32(char * p, size_t n, unsigned seed) {
    srand(seed);
    float * f = (float *) p;
    for (size_t i = 0; i < n/4; ++i) f[i] = ((float) rand() / RAND_MAX) * 2.0f - 1.0f;
}

static void fill_f16(char * p, size_t n, unsigned seed) {
    srand(seed);
    ggml_fp16_t * h = (ggml_fp16_t *) p;
    for (size_t i = 0; i < n/2; ++i) h[i] = ggml_fp32_to_fp16(((float) rand() / RAND_MAX) * 2.0f - 1.0f);
}

// build the graph with head-major Q/K/V and return the out tensor
static ggml_tensor * build_graph(ggml_context * ctx, int hsk, int hsv, int n_head,
                                 int n_head_kv, int n_q, int n_kv, int n_batch,
                                 ggml_tensor ** qc, ggml_tensor ** kc, ggml_tensor ** vc, ggml_tensor ** m) {
    ggml_tensor * qcc = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hsk, n_head, n_q, n_batch);
    ggml_tensor * q   = ggml_view_4d(ctx, qcc, hsk, n_q, n_head, n_batch,
                                     qcc->nb[2], qcc->nb[0], qcc->nb[2]*n_head, 0);
    ggml_tensor * kcc = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hsk, n_head_kv, n_kv, n_batch);
    ggml_tensor * k   = ggml_view_4d(ctx, kcc, hsk, n_kv, n_head_kv, n_batch,
                                     kcc->nb[2], kcc->nb[1], kcc->nb[3], 0);
    ggml_tensor * vcc = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hsv, n_head_kv, n_kv, n_batch);
    ggml_tensor * v   = ggml_view_4d(ctx, vcc, hsv, n_kv, n_head_kv, n_batch,
                                     vcc->nb[2], vcc->nb[1], vcc->nb[3], 0);
    ggml_tensor * mm  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q, 1, 1);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mm, 1.0f/sqrtf((float)hsk), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    *qc = qcc; *kc = kcc; *vc = vcc; *m = mm;
    return out;
}

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t cl  = ggml_backend_opencl_init();
    if (!cpu || !cl) { printf("backend init failed\n"); return 1; }

    const int hsk = 256, hsv = 256;
    const int n_head = 8, n_head_kv = 2;
    const int n_q = 1, n_kv = 256, n_kv_valid = 12, n_batch = 1;

    // host data for the physical buffers
    std::vector<char> qhd((size_t) hsk * n_head * n_q * n_batch * 4);
    std::vector<char> khd((size_t) hsk * n_head_kv * n_kv * n_batch * 2);
    std::vector<char> vhd((size_t) hsv * n_head_kv * n_kv * n_batch * 2);
    fill_f32(qhd.data(), qhd.size(), 11);
    fill_f16(khd.data(), khd.size(), 22);
    fill_f16(vhd.data(), vhd.size(), 33);
    // causal mask: first n_kv_valid positions attend, rest masked
    std::vector<ggml_fp16_t> mhd(n_kv*n_q);
    for (int i = 0; i < n_kv; ++i) mhd[i] = ggml_fp32_to_fp16(i < n_kv_valid ? 0.0f : -INFINITY);

    ggml_init_params ip = { 512*1024*1024, NULL, true };
    ggml_tensor * qc, * kc, * vc, * m;

    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * out = build_graph(ctx, hsk, hsv, n_head, n_head_kv, n_q, n_kv, n_batch, &qc, &kc, &vc, &m);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t cl_buf = ggml_backend_alloc_ctx_tensors(ctx, cl);
    if (!cl_buf) { printf("opencl alloc failed\n"); return 1; }

    ggml_backend_tensor_set(qc, qhd.data(), 0, qhd.size());
    ggml_backend_tensor_set(kc, khd.data(), 0, khd.size());
    ggml_backend_tensor_set(vc, vhd.data(), 0, vhd.size());
    ggml_backend_tensor_set(m,  mhd.data(),  0, mhd.size()*sizeof(ggml_fp16_t));
    ggml_backend_graph_compute(cl, gf);

    std::vector<float> cl_out(hsk*n_q*n_head*n_batch);
    ggml_backend_tensor_get(out, cl_out.data(), 0, hsk*n_q*n_head*n_batch*sizeof(float));

    ggml_context * ctx2 = ggml_init(ip);
    ggml_tensor * qc2, * kc2, * vc2, * m2;
    ggml_tensor * out2 = build_graph(ctx2, hsk, hsv, n_head, n_head_kv, n_q, n_kv, n_batch, &qc2, &kc2, &vc2, &m2);
    ggml_cgraph * gf2 = ggml_new_graph(ctx2);
    ggml_build_forward_expand(gf2, out2);
    ggml_backend_buffer_t cpu_buf = ggml_backend_alloc_ctx_tensors(ctx2, cpu);
    ggml_backend_tensor_set(qc2, qhd.data(), 0, qhd.size());
    ggml_backend_tensor_set(kc2, khd.data(), 0, khd.size());
    ggml_backend_tensor_set(vc2, vhd.data(), 0, vhd.size());
    ggml_backend_tensor_set(m2,  mhd.data(),  0, mhd.size()*sizeof(ggml_fp16_t));
    ggml_backend_graph_compute(cpu, gf2);

    std::vector<float> cpu_out(hsk*n_q*n_head*n_batch);
    ggml_backend_tensor_get(out2, cpu_out.data(), 0, hsk*n_q*n_head*n_batch*sizeof(float));

    double maxerr = 0.0, sumerr = 0.0;
    int bad = 0;
    for (size_t i = 0; i < cl_out.size(); ++i) {
        double e = fabs((double)cl_out[i] - (double)cpu_out[i]);
        double r = cpu_out[i] != 0 ? e / fabs((double)cpu_out[i]) : e;
        maxerr = fmax(maxerr, e);
        sumerr += e;
        if (r > 1e-3 && bad < 10) {
            printf("MISMATCH[%zu] cl=%.6f cpu=%.6f\n", i, cl_out[i], cpu_out[i]);
        }
        if (r > 1e-3) bad++;
    }
    printf("FA qwen head-major: values=%zu bad=%d maxerr=%.6g meanerr=%.6g\n",
           cl_out.size(), bad, maxerr, sumerr/cl_out.size());

    ggml_backend_buffer_free(cl_buf);
    ggml_backend_buffer_free(cpu_buf);
    ggml_free(ctx);
    ggml_free(ctx2);
    ggml_backend_free(cl);
    ggml_backend_free(cpu);
    return bad ? 1 : 0;
}