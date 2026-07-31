# Quantization recipes and calibration

## Calibration data: get this right or nothing downstream matters

**Capture at the ONNX boundary, not the PyTorch module boundary.** If the export
used a wrapper, the graph inputs are the *wrapper's*. Dumping tensors from the
underlying module gives you keys that either `KeyError` or — much worse —
silently calibrate on the wrong distribution. Assert it:

```python
assert set(npz.keys()) == {i.name for i in sess.get_inputs()}
```

Other traps, each of which cost real time:

- A tensor-only numpy converter **silently drops a Python list of tensors**
  (returns None). That is exactly how a `deepstack` list of hidden states goes
  missing with no error.
- `CalibrationDataReader.get_next()` must return a dict keyed by exact graph input
  names, shapes matching the declared (usually batch-1) inputs, and `None` to
  terminate. The reader is **single-pass** — build a fresh one per
  `quantize_static` call.
- **Interior chunks of a split model must have feeds *produced*, not captured**:
  run the fp32 chain and hand the resulting dicts to the reader.
- Size the per-component capture cap to (calls per inference) × (inference
  passes), so every component covers the same passes. With 3 policy calls and
  `num_inference_timesteps=4`, the denoising-loop components need 12 caps and the
  single-call ones 3. The reason is **range coverage**, not weighting — each
  denoise step sits at a different timestep/noise level.
- Normalize calibration inputs **exactly** as at inference (or not at all, if
  normalization is baked into the graph). Pick one side of that fence and use the
  same code path in export, calibration, host comparison, and the device loop.
  Watch for `OBS_MEAN`/`OBS_STD` left as unset TODOs — the normalize function then
  silently no-ops.
- Keep **two disjoint sets**: calibration and held-out parity. Ranges are fit on
  the calibration set, so parity measured there is optimistically biased.

Sample counts, by source: QNN porting docs 50–200; QAIRT troubleshooting FAQ
floor ~25–30; AI Hub 500–1000 (its tutorial uses 100). Cover edge states, not just
steady state.

## Golden-capture plumbing

- Capture calibration data and the golden reference in **one pass** with
  `register_forward_hook(..., with_kwargs=True)`, one `.npz` per call keyed
  `in_arg<i>` / `in_<kwarg>` / `out<j>`. Hook the module whose signature matches
  the **exported graph**.
- Golden `.npz` keys generally do **not** match ONNX input names (wrappers rename
  them, e.g. `in_hidden_states` → `sa_embs`). Build an explicit name map and
  assert it against `sess.get_inputs()`; don't rely on ordering or prefix-stripping.
- When auto-wiring npz → ONNX inputs (exact name → `in_<name>` → dtype+trailing-dims
  heuristic → positional), cast each value to the graph's declared dtype. ORT
  rejects mismatches with `InvalidArgument : Unexpected input data type`, and npz
  float64 is the usual culprit. Cover INT32/FP16, not just FLOAT/INT64/BOOL.
  Print the resolved name→shape/dtype map once and eyeball it. Known bug pattern:
  the exact-match branch forgetting to mark the key consumed, letting the shape
  heuristic silently reuse it for another input.
- If a re-capture pass **deletes** the earlier `call*.npz`, it also deletes the
  torch goldens, silently downgrading validation to quant-vs-fp32-ONNX. Keep the
  two capture sets in separate directories.
- `np.savez`/`np.load` round-trip keys containing `/` fine (verified numpy 2.2.6
  and 2.4.4) — slashes are not a reason to avoid caching interior-chunk feeds.

## bf16 → fp32 first

ORT's quantizer only accepts fp32/fp16 initializers, and calibration runs on the
CPU EP which has essentially no bf16 op coverage. A conversion pass must cover
**all five** of:

1. `graph.initializer` raw data
2. `elem_type` on `graph.input` / `output` / `value_info`
3. `Cast` node `to` attributes
4. TENSOR-typed node attributes
5. subgraph attributes (`AttributeProto.GRAPH`/`GRAPHS`) and `sparse_initializer`

A top-level-only pass silently leaves 3–5 as bf16. Prefer re-exporting fp32
(`model.float()`) when you still control the export. Note bf16→fp32 roughly
doubles size, which is what pushes a ~2 GB model over the protobuf limit.

## Transformers: quantize MatMuls only

```python
quantize_static(fp32, out, calibration_data_reader=Reader(feeds),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt16, weight_type=QuantType.QInt16,
    use_external_data_format=True,
    op_types_to_quantize=["MatMul"],
    calibrate_method=CalibrationMethod.MinMax)
```

Measured on a GR00T N1.7 DiT (host ORT CPU vs a bf16 PyTorch golden):

| Config | cos |
|---|---|
| MinMax over **all** op types | **0.85** — unusable |
| MatMul-only, uint16/int16 | **0.999921** (fp32-ONNX reference itself: 0.999922) |

Same recipe held for the LLM (0.99994), ViT (1.00000), vl_self_attention (0.99998).

**Root cause of the collapse**, reproduced on a minimal mask+MatMul graph
(ORT 1.22): tensors carrying `±inf` attention-mask values produce **inf/nan scale
and zero_point**, and the quantizer emits a silently corrupted model with no
error. `CalibrationMethod.Percentile` and `.Entropy` instead raise
`ValueError: ... range of [-inf, inf] is not finite` from
`calibrate.py::collect_absolute_value` (they build `np.histogram`s). If you need a
histogram method, replace `-inf` in the mask with a large finite negative (-1e4)
first.

`op_types_to_quantize` is an **include**-list; use `nodes_to_exclude` /
`nodes_to_quantize` to drop specific nodes.

This is the plain-ORT lever. The vendor path for accuracy-critical transformers is
AIMET / QAIRT per-op encoding overrides, or ORT's `get_qnn_qdq_config` — and note
**AI Hub's default range scheme is `mse_minimizer`** (histogram with outlier
clipping), which is why it survives whole-transformer quantization where naive
MinMax-over-everything does not.

## Bit width is per component

Measured, GR00T N1.7 on QCS8275/v75, QAIRT 2.46, ORT-QNN 1.23 — a **worked
example**, not a rule table:

| Component | Shipped | cos | Note |
|---|---|---|---|
| action_decoder (small MLP) | INT8 | 0.9991 host / 0.9992 HTP | maxErr 0.048, 1.74 ms |
| action_encoder (shallow MLP) | **INT16** | 0.99999 | INT8 gave 0.9918 with **maxErr 21** |
| DiT (16-layer cross-attn) | INT8 MatMul-only | 0.990 host / 0.9975 HTP | INT16 scored 0.999921 but was dropped for footprint |
| LLM (2B decoder) | **INT16** | 0.99994 | INT8 **collapses to cos 0.041, maxErr 6460** |
| ViT | INT16 | 1.0000 | |
| stitched all-HTP end-to-end | — | 0.998850, MAE 0.029 | |

The lesson: **the lever is MatMul-only, and escalation is per component**. Neither
"INT8 for MLPs" nor "INT16 for transformers" holds — the shipped DiT is INT8 and a
shipped shallow MLP is INT16. And because this is naive MinMax PTQ, low bit width
per se is not the problem: Qualcomm's own LLM path reaches good accuracy at
**W4A16** via AIMET advanced PTQ.

## Size expectations

- **INT16 buys zero size reduction over bf16** — both are 2 bytes. Verified:
  `llm_bf16.onnx` 1.611 GB → `llm_uint16_int16.onnx.data` 1.611 GB. INT8 halves
  it to 0.806 GB.
- `op_types_to_quantize=["MatMul"]` leaves every non-MatMul weight at fp32, so the
  output can be **larger** than the bf16 source (an INT8 MatMul-only DiT still
  held ~638 MB of fp32 initializers).
- Do **not** conclude the leftover fp32 softmax/layernorm falls back to CPU. HTP
  executes float ops; the measured MatMul-only DiT put 3084/3101 nodes on QNN.
  Fragmentation comes from unsupported **op types**, not float precision.

## Preprocessing bugs that masquerade as quantization loss

- **Never infer preprocessing from input shape.** `face_det_lite`'s
  `(1,480,640,1)` input looks grayscale but `app.py` takes the **blue channel**
  (`img_tensor[:, :, :, -1]` of the /255.0 RGB tensor). Read the model's `app.py`.
- AI Hub classification models generally output **raw logits** with no final
  Softmax — top-k "scores" aren't probabilities. Check whether values sum to ~1
  before adding softmax; applying it twice silently corrupts scores.
- **Don't infer the quantized model's I/O contract from the fact that it's
  quantized.** In QDQ form the public I/O normally stays **FLOAT** with Q/DQ at
  the boundary while activations are integer internally (an AI Hub W8A16 model kept
  a FLOAT `[1,1,320,320]` output fed by a uint16 tensor through a
  DequantizeLinear). Read it from `sess.get_inputs()`/`get_outputs()`, and note
  layout — AI Hub ONNX inputs are often NHWC.
- Quant-param access is asymmetric: LiteRT exposes them at runtime
  (`input_details[0]['quantization']` → `(scale, zero_point)`); ORT's Python API
  does **not**, which is why the Dragonwing example hard-codes `scale=1/255, zp=0`
  and warns you to change it. The params **are** in the QDQ `.onnx` as
  initializers of the Q/DQ nodes — read them with `onnx` or Netron.

## Fixed shapes vs variable-length inputs

HTP compiles at fixed shapes, so any input whose length varies per observation (a
VLM's `vl_seq_len`: 149 at rollout vs 156 captured at export) must be handled at
the **wrapper**: pad the sequence tensors *and their masks* up to the compiled max
with `mask=False`, and `assert S <= max` so an over-long prompt fails loudly
instead of truncating.

Padding is numerically exact **only where every consumer of the padded axis is
masked**. Masked cross-attention softmax is; an unmasked mean/sum-pool is not.
**A component with no mask input cannot be padded at all** — pad positions leak
into every real token through the softmax, and trimming the output does not
rescue it; it must stay fp32 or be re-exported with a mask/dynamic axis.

Design rule: export every variable-length component **with** its mask input.
