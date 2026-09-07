# Qwen3.8-2B on OpenCL - investigation findings (branch: opencl-qwen38)

## Model
- `empero-ai/Qwen3.8-2B-Distill-GGUF`, `Qwen3.8-2B-Q4_K_M.gguf` (1.31 GB), stored at `~/WS/Models/`.
- Arch `qwen35` (18 layers), hybrid Gated-Delta-Net + attention.
- Params: d_model=2048, n_head=8, n_head_kv=2 (GQA 4:1), dk=dv=256, ctx=262144.
- SSM: conv_kernel=4, state_size=128, group_count=16, dt_rank=16, inner_size=2048.
- Weights are Q6_K for most tensors (attn_qkv, ffn, token_embd [2048,248320]) and
  Q4_K for attn_q [2048,4096] / attn_output [2048,2048] - the "Q4_K_M" quant uses
  Q6_K in several places.

## Symptom
- Full GPU (`-ngl 999`) produces garbage: starts with a plausible token ("The"),
  then degenerates to "chenchen...天秤..." - classic recurrent-state corruption.
- Full CPU (`-ngl 0`) produces correct output: "...The question is asking for the
  capital of France. France's capital is a well-established fact."
- Generation speed with GPU ~1 t/s (model is heavily CPU-bound even with OpenCL).

## Root cause (isolated)
- Forcing MUL_MAT (opfilter `GGML_OPENCL_OPFILTER="MUL_MAT.*"`) to CPU makes the
  model produce correct output. All other op groups (SSM/GDN, ROPE, FA, activations)
  on CPU still give garbage. So the OpenCL MUL_MAT path is the culprit.
- CAVEAT: the opfilter only works with a trailing `.*` (op_desc includes params,
  e.g. "MUL_MAT(type_a=...)"; plain `regex_match("MUL_MAT")` matches nothing).
  Earlier "FA on CPU" / "MUL_MAT on CPU" successes without `.*` were no-ops/flukes.

## What passes (test-backend-ops, GPUOpenCL)
- All the model's MUL_MAT shapes at n=1/2/4/7/11:
  - q4_K m=4096 k=2048 (attn_q, incl. our flat workaround)
  - q4_K m=2048 k=6144, m=2048 k=2048
  - q6_k m=6144 k=2048, m=2048 k=6144, m=2048 k=2048, m=8192 k=2048
  - q6_k m=248320 k=2048 (vocab-scale output projection, flat GEMM)
- FLASH_ATTN_EXT at the model's config (dk=256, f16 KV, GQA nh=2/nr23=[4,1],
  kv_view, n_kv_max) - OK.
- Custom probe `test_fa_qwen.cpp` (ggml graph with head-major Q/K/V views) shows
  the OpenCL FA matches CPU within f16 precision (max abs err ~5.6e-4, no garbage).

## ROOT CAUSE (found via per-token logits + per-tensor dumps)

Deterministic, NOT a race (same seed -> identical garbage). Failure is at the
first decode: pos=0 logits match CPU, pos=1..N are FROZEN (GPU logits nearly
identical across positions: top token 7286 with ~constant logits).

Tensor-dump comparison (GPU vs CPU) for the first GDN layer (layer 0):
- linear_attn_qkv_mixed (MUL_MAT projection):    MATCHES CPU (diff < 0.08)
- conv_states (mctx conv state read):            MATCHES CPU (diff < 0.03)
- conv_output_raw (SSM_CONV output):             GPU near-zero (~0.0007) vs CPU ~0.14  <-- FIRST DIVERGENCE
- q_conv_predelta (after conv + L2 norm):        wildly different (GPU ~0.005 vs CPU ~4.6)
- new_state (GATED_DELTA_NET output/state):     wildly different even with conv on CPU
- gate_reshaped (alpha projection m=16):         MATCHES CPU (with conv on CPU)

Conclusion:
1. OpenCL SSM_CONV kernel produces near-zero output for this model's data
   (data-dependent miscompile on Adreno E031) - passes test-backend-ops with
   random data, fails with real activations.
2. OpenCL GATED_DELTA_NET produces a wrong recurrent state even when the conv
   is fixed (conv forced to CPU) - a second data-dependent miscompile.
3. Forcing MUL_MAT to CPU fixes the model because graph partitioning then also
   moves SSM_CONV + GDN (and their inputs) onto the CPU backend.

These are value-sensitive compiler miscompiles, not shape bugs and not races.

## FIX PROGRESS

- opencl-fixes c89134530: SSM_CONV `_4` kernel no longer uses float4 `dot()`.
  The vectorized path was selected whenever ne10 % 4 == 0, but input rows are
  only 16-byte aligned for i2 == 0 (other output tokens read misaligned data),
  and `dot()` miscompiles on the Adreno E031 compiler for certain values
  (reproduced: n_t=7 conv output went from near-zero to correct with scalar
  mad). Verified: test-backend-ops SSM_CONV still passes with the fix.
- HOST-VERIFY (GGML_OPENCL_GDN_VERIFY / GGML_OPENCL_CONV_VERIFY): SSM_CONV and
  GATED_DELTA_NET kernels compute correctly from their inputs.
- VERIFIED CORRECT on GPU vs CPU (tensor dumps): SSM state (cache_s_l0),
  attention K cache (cache_k_l3), final_output (GDN layer out), pos=0 logits.
- DIVERGENCE POINT (n_t=4 ubatch): attn_output/attn_pregate of the attention
  sublayer diverge wildly (e.g. -0.663 vs 0.350). The RoPE'd query tensor
  `Qcur-3` is TINY on GPU ([-0.005...]) vs CPU ([-0.317...]) for the n_t=4
  batch. Tiny Q -> tiny attention -> frozen logits.
- ROPE (incl. MRoPE) suite passes on OpenCL; MUL_MAT, FA (nb=4 dk=128/256),
  SSM_CONV, GDN all pass isolated with random data. So the n_t=4 failure is a
  DATA-DEPENDENT miscompile in the projection->reshape->MRoPE query path (or a
  view/reshape interaction), not reproducible by random-data tests. The
  model's MRoPE: rope.dimension_count=64, sections=[11,11,10,0], freq_base=1e7.
- Next: isolate whether Qcur_full (Q projection MUL_MAT out) is tiny or the
  reshape/view/MRoPE collapses it - needs a host-verify in the MUL_MAT or ROPE
  dispatch (dump-hook reads are unreliable due to shared buffers).

## Not yet resolved
- The specific MUL_MAT config that breaks is NOT reproduced by isolated
  test-backend-ops cases (all shapes/data pass with random data). Likely a
  value-dependent kernel bug OR a CPU/GPU data-flow interaction in the full
  hybrid graph (some ops fall to CPU, tensors cross backend boundaries).
- Next step candidates:
  1. Dump a hidden-state / logits vector from GPU vs CPU for the same prompt to
     find the first diverging tensor (llama-cli lacks a logits dump here).
  2. Force the suspect kernel variants (flat / o4 / nsg splits) for Q6_K.
  3. Test with quant-KV q8_0 (--cache-type-k q8_0) - also fails, so not cache-type.
- Note: FA head-major Q/K/V layouts are correct within precision; the head-major
  layout theory did NOT explain the garbage.

## Branch state
- `opencl-qwen38` = `opencl-fixes` + pending FA `is_f32_quant` change (q4_1/q5_0/
  q5_1/iq4_nl KV via host-dequant fallback, uncommitted) + `test_fa_qwen.cpp` probe.
- Working tree otherwise clean (all debug prints removed).