# Path B — QAIRT native converter + quantizer (local, offline)

**This is the default route** whenever the host can run the tools. See the host
support table in SKILL.md; in short, converters/quantizers ship for **x86_64
Linux, x86_64 Windows and arm64x Windows**, but *not* aarch64 Linux and not macOS.
Confirm on your own install:

```bash
for d in "$QAIRT_SDK_ROOT"/bin/*/; do
  n=$(ls "$d" 2>/dev/null | grep -cE '^qairt-(converter|quantizer)$')
  [ "$n" -gt 0 ] && echo "host tools present: $(basename "$d")"
done
# 2.46.0.260424: x86_64-linux-clang, x86_64-windows-msvc, arm64x-windows-msvc
```

## Get the SDK (no login, no QPM)

**Never hardcode a version.** The URL embeds a full 4-part build id whose last
field is the build **date** (YYMMDD), and the endpoint is a plain object lookup
with no listing API — a wrong key returns S3 `NoSuchKey`, and
softwarecenter.qualcomm.com is a client-rendered SPA, so scraping it without a
browser yields only the page shell. Use the bundled resolver:

```bash
scripts/fetch-latest-qairt.sh                      # print the latest version + URL
scripts/fetch-latest-qairt.sh --download ~/qairt   # download (~1.8 GB) + unzip
scripts/fetch-latest-qairt.sh --from 2.46          # start the sweep higher
```

Two strategies, cheapest first:

1. **AI Hub `get_frameworks()`** — authoritative and one request, but needs a
   configured `qai-hub` token. Note `full_version` carries a long build stamp
   (`2.48.0.260626120635`) while the download path uses only its first 6 digits
   (`2.48.0.260626`).
2. **Ranged-GET sweep** — no auth. Walks minors down from a ceiling and fans a
   minor's whole date window out through `xargs -P 24` (~0.3 s per probe, so the
   cost is request *count*, not bandwidth).

Latest verified 2026-08-04: **2.48.0.260626**. Doing it by hand:

```bash
VER=2.48.0.260626          # full 4-part build id, in BOTH places
curl -L -O "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/$VER/v$VER.zip"
unzip "v$VER.zip" && cd "qairt/$VER"        # extracts to qairt/<ver>/, NOT v<ver>/qairt
export QAIRT_ROOT=$(pwd)
source "$QAIRT_ROOT/bin/envsetup.sh"        # exports QAIRT_SDK_ROOT, PATH, PYTHONPATH, LD_LIBRARY_PATH
```

301-redirects to apigwx-aws.qualcomm.com. To test whether a build id exists, use a
**ranged GET** — a HEAD request 403s:

```bash
curl -sL -r 0-100 -o /dev/null -w '%{http_code}\n' "<url>"   # 206 = exists, 404 = not
```

Do **not** set `QNN_SDK_ROOT` yourself — `envsetup.sh` derives it, and warns that
it exists only for backwards compatibility and will be deprecated. Use
`$QAIRT_SDK_ROOT`.

Two directories, both needed, easy to mix up:

```bash
BIN="$QAIRT_SDK_ROOT/bin/x86_64-linux-clang"   # executables: qairt-*, qnn-*, snpe-*
LIB="$QAIRT_SDK_ROOT/lib/x86_64-linux-clang"   # libQnnHtp.so, libQnnModelDlc.so, libQnn*Extensions.so
```

There are **zero `.so` files under `bin/`** — passing `$BIN/libQnnHtp.so` fails.

## Host prereqs

```bash
sudo bash "$QAIRT_ROOT/bin/check-linux-dependency.sh"        # -n for dry run
sudo apt-get install python3.10 python3.10-venv python3-distutils \
                     libpython3.10 libstdc++-12-dev clang libc++-dev libc++abi-dev
python3.10 -m venv .venv && . .venv/bin/activate && pip install -U pip pyyaml
"$QAIRT_ROOT/bin/check-python-dependency"                    # SDK deps only — see below
pip install onnx onnxruntime onnxsim                         # NOT covered by the above
```

`check-python-dependency` pins ~34 packages (numpy, pyyaml, protobuf, scipy …) and
contains **no reference at all** to `onnx`, `onnxruntime`, `onnxsim`, `torch`,
`tensorflow` or `tflite`. It will pass while `qairt-converter -i model.onnx` still
fails on a missing/mismatched `onnx`. Take exact pins from your release's own
docs — the 2.47.0.260601 docs pin `onnx==1.19.1 onnxruntime==1.23.2
onnxsim==0.6.2`, TF 2.10.1, tflite 2.18.0, torch 2.4.1. It also enforces an
Ubuntu-version → Python-version map, so "3.8/3.10 only" is necessary but not
sufficient.

Two environment failures you will hit:

- **Python tools** (`qairt-quantizer`, `qairt-converter`) →
  `ImportError: libscipy_openblas64_-<hash>.so: cannot open shared object file`.
  Cause: broken user-site numpy on the system python3. `PYTHONNOUSERSITE=1` alone
  does **not** fix it (there is no system numpy to fall back to). Use a clean venv.
- **C++ tools** (`qnn-context-binary-generator`) →
  `error while loading shared libraries: libc++.so.1`. The SDK ships **no**
  `libc++.so.1` anywhere; pointing `LD_LIBRARY_PATH` at the SDK cannot help.
  `sudo apt install clang libc++-dev libc++abi-dev`.

## Version-lock before compiling anything

```bash
grep -iE '^(version|build_id):' "$QAIRT_SDK_ROOT/sdk.yaml"     # host
strings /usr/lib/libQnnHtp.so | grep AISW_VERSION              # device
```

A mismatch is **silent at compile time** and only fails at load
(`QNN_COMMON_ERROR_INCOMPATIBLE_BINARIES` / blob-version error). The rule is
**directional, not equality**: runtime SDK must be **≥** prepare SDK.
Older-binary-on-newer-runtime loads (init acceleration auto-disables — slower
init only); newer-binary-on-older-runtime fails.

## Calibration data

`--input_list` is a plain text file, one line per inference:

```
/abs/path/sample0.raw
# multi-input: all inputs on ONE line
inputA:=/abs/path/a0.raw inputB:=/abs/path/b0.raw
```

Generate it:

```python
import numpy as np, pathlib
OUT = pathlib.Path("/tmp/calib"); OUT.mkdir(parents=True, exist_ok=True)
paths = []
for i, sample in enumerate(representative_samples):        # NOT random data
    p = OUT / f"input_{i}.raw"
    # exact same preprocessing as inference; C-contiguous; headerless
    np.ascontiguousarray(sample, dtype=np.float32).tofile(p)
    paths.append(p.resolve())
pathlib.Path("calib.txt").write_text("".join(f"{p}\n" for p in paths))
```

- Byte size must equal `prod(input_shape) * 4` — float32 is the default read type.
  `--use_native_input_files` switches to the model-native dtype and changes the
  expected size.
- Use **absolute** paths (relative resolves against CWD and breaks when the list
  moves).
- The quantizer requires the DLC's input **batch dim to be 1** — set that at
  convert time.
- **Watch layout.** The vendor sample writes NHWC `(1,299,299,3)` raws while
  passing `--input_dim 1,3,299,299`. Byte sizes coincide, so it is accepted and
  wrong.
- Sample count: QNN porting docs say 50–200 representative preprocessed samples;
  the troubleshooting FAQ gives ~25–30 as a floor; AI Hub recommends 500–1000.
  Cover edge states (max commanded velocity, contact transients), not just
  steady state.

## The modern flow

```bash
# 1. convert (host tools only — see the ABI check above). --target_backend defaults to HTP.
"$BIN/qairt-converter" -i model.onnx -o model.dlc \
  --source_model_input_shape 'obs' 1,48 \
  --target_backend HTP --target_soc_model qcs8275 \    # SoC NAME string, not an enum int
  [--quantization_overrides overrides.json] [--export_format DLC_STRIP_QUANT]

# 2. quantize
"$BIN/qairt-quantizer" --input_dlc model.dlc --output_dlc model_q.dlc \
  --input_list calib.txt \
  --act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 8 \
  --param_quantizer_schema symmetric --act_quantizer_schema asymmetric \
  --use_per_channel_quantization

# 3. discover the real graph name BEFORE writing htp_config.json
"$BIN/snpe-dlc-info" -i model_q.dlc | head -40        # or qairt-dlc-info

# 4. compile to a context binary (offline, no target HW needed)
cat > wrapper.json <<EOF
{"backend_extensions": {"shared_library_path": "$LIB/libQnnHtpNetRunExtensions.so",
                        "config_file_path": "./htp_config.json"}}
EOF
"$BIN/qnn-context-binary-generator" \
  --backend "$LIB/libQnnHtp.so" \
  --model "$LIB/libQnnModelDlc.so" --dlc_path model_q.dlc \
  --config_file wrapper.json --binary_file model --output_dir out \
  --log_level verbose

# 5. run it on the device — do not ship an unverified binary
scp out/model.bin <board>:/tmp/
ssh <board> 'qnn-net-run --retrieve_context /tmp/model.bin --backend libQnnHtp.so \
    --input_list /tmp/calib.txt --output_dir /tmp/out_htp'
```

### Non-obvious behaviour

- **`--input_list` is what triggers quantization.** Without it, every other
  quantization option is **silently ignored** and you get a float DLC with exit 0
  (`Skipping quantization, no input_list provided`). Assert the output is
  actually quantized. The docs call one of `--input_list` /
  `--enable_float_fallback` "mandatory" — it is **not enforced**.
- `--input_list` and `--enable_float_fallback` are mutually exclusive
  (the legacy converter spells it `--float_fallback`). `--enable_float_fallback`
  also conflicts with `--ignore_quantization_overrides`.
- Bit widths: `--bias_bitwidth 8|32`, `--act_bitwidth 8|16`,
  `--weights_bitwidth 2|4|8|16`, `--float_bitwidth 32|16`,
  `--float_bias_bitwidth 32|16` (**default 0**, not 32). There is **no `choices=`
  validation** — a bad value surfaces as an obscure native error, not a usage
  message. Legacy `qnn-*-converter` documents weights as 4 or 8 only.
- On the **legacy** converters `--bias_bw`/`--act_bw`/`--weight_bw` still parse as
  deprecated aliases, and passing both forms raises `Invalid combination:
  --X_bw and --X_bitwidth cannot be provided at the same time.` On
  `qairt-quantizer` the `_bw` names **do not exist** — hard argparse error.
  Other renames: `--algorithms`→`--apply_algorithms`,
  `--float_fallback`→`--enable_float_fallback`,
  `--ignore_encodings`→`--ignore_quantization_overrides`,
  `--config_file`→`--config`, `--act_quantizer`→`--act_quantizer_calibration`.
- Prefer the modern calibration/schema pair over legacy `--param_quantizer` /
  `--act_quantizer`: `--act_quantizer_calibration` /
  `--param_quantizer_calibration` ∈ {min-max (default), sqnr, entropy, mse,
  percentile} with `--act_quantizer_schema` / `--param_quantizer_schema` ∈
  {asymmetric (default), symmetric, unsignedsymmetric, signedasymmetric}.
  `--percentile_calibration_value` must be in [90,100] (default 99.99) and only
  with `percentile`. Mixing legacy + new raises — but the guard reads `sys.argv`,
  so it fires only on the CLI, not via the Python API or a response file.
- **HTP is the default backend** in qairt-converter/qairt-quantizer (legacy left
  it empty), so HTP-specific behaviour is on even if you never pass
  `--target_backend`. Quantized numerics can therefore differ from a legacy
  pipeline with identical flags.
- `--use_per_channel_quantization` covers convolution-family weights
  (Conv2d/Conv3d/TransposeConv2d/DepthwiseConv2d). **QNN does not support
  per-channel MatMul** — use `--use_per_row_quantization` for
  MatMul/FullyConnected (`--enable_per_row_quantized_bias` adds the FC bias).
  Its QAT-encoding precedence is **contradictory in-SDK**: qairt-quantizer's help
  says per-channel applies only if QAT encodings are absent; the legacy
  converter's says it *replaces* them. Verify with `--dump_encoding_json`.
- `--apply_algorithms` on the default IR quantizer exposes only `cle`.
  AdaRound/AMP/AutoQuant need `--use_aimet_quantizer` plus pip-installed
  aimet-torch. AdaRound can run with no config (calibrating from `--input_list`);
  `amp`/`autoquant` require a YAML `--config`. One algorithm per run; not
  combinable with `--enable_float_fallback`. The AIMET quantizer silently forces
  `--bias_bitwidth 32` and rejects `--restrict_quantization_steps`,
  `--pack_4_bit_weights`, `--use_dynamic_16_bit_weights`, `--op_package_lib`,
  `--keep_weights_quantized`, `--adjust_bias_encoding`.
- `--export_format`: `DLC_DEFAULT` (default), `DLC_STRIP_QUANT` (drop all quant
  data → float graph; the documented way to discard source/QAT encodings and
  re-calibrate), plus a hidden `DLC_FLOAT` that the error message won't mention.
- In QAIRT, overrides and source QAT encodings are applied at **convert** time.
  So on qairt-quantizer `--enable_float_fallback` is a true no-op, and
  `--ignore_quantization_overrides` is a no-op *alone* — but **paired with
  `--input_list` it forces re-derivation from calibration data**, which is what
  you need to re-calibrate an already-quantized DLC (otherwise: `Skipping
  calibration`).

## `--quantization_overrides` (AIMET-format JSON)

Belongs to **qairt-converter**, not qairt-quantizer (which only has
`--ignore_quantization_overrides`; the two are mutually exclusive).

Two mandatory top-level sections, either may be empty: `"activation_encodings"`
(op outputs) and `"param_encodings"` (weights/biases). Per entry supply
`bitwidth`, `min`, `max` — scale and offset are **always recomputed** from those.
The file carries a schema `version`; with 2.0.0 (or `--use_quantize_v2`) bias
bitwidth becomes backend-derived and `--dump_encoding_json` is rejected.

**Per-channel:** N encoding dicts under one tensor name where N == that tensor's
output-channel count. Each must be symmetric with `offset == -2^(bitwidth-1)`.
Match the doc's literal formatting: `is_symmetric` is the quoted string `"True"`
and `offset` a float `-128.0`.

**Mixed precision:** per tensor, `{"bitwidth": N, "dtype": "int"|"float"}` (no
min/max needed). Invariant: for any op, all inputs/outputs/params are either float
or integer — the converter inserts convert ops at boundaries. For
Conv2d/Conv3d/TransposeConv2d/DepthwiseConv2d/FullyConnected/MatMul, marking
**any** of output/weights/bias float forces all three float. Use
`--float_bias_bitwidth 16|32` when the bias encoding is absent but output/weights
are float. Tensors absent from the JSON stay fixed-point at the `--*_bitwidth`
defaults. (That doc section still uses the deprecated `--act_bw` spellings.)

**INT32 overrides** must be given for **all** of an op's inputs and outputs; the
converter then injects `Dequantize → Cast(INT32)` on inputs and
`Cast(FP32) → Quantize` on the output — extra ops, real latency cost. Only
supported for ops with no weights and no bias.

**Numeric contract:** encoding range floored at 0.0001
(`encoding_max := max(true_max, true_min + 0.0001)`); float 0.0 is always exactly
representable (the range shifts if needed — `[-5.1, 5.1]` at bw=8 becomes
`[-5.12, 5.08]`); `scale = (max-min)/(2^bw - 1)`; `offset = round(min/scale)`,
which is ≤ 0 in the JSON (the "zero point" in examples is `-offset`).

**Packed 4-bit** (`--pack_4_bit_weights`): two 4-bit weights per byte, lower
nibble first — `(10,4)` → `0100 1010`. Per-channel packing is along the channel,
so 27 values/channel → `ceil(27/2) = 14` bytes. Packed carries
`QNN_DATATYPE_[SU]FIXED_POINT_4`; **unpacked** 4-bit carries `..._8`. Check the
backend OpDef supplement for which ops accept packed 4-bit.

## HTP backend-extensions config

`--config_file` takes a **two-level wrapper**, not the HTP config itself. Handing
the HTP JSON directly loads no extensions at all, so `dsp_arch` / `soc_model` are
silently ignored and you get a **mistargeted binary with no error**. Paths in the
wrapper resolve against CWD — use `./` or absolute. (At Genie *runtime* the
wrapper is not used: `genie_config.json`'s `engine.backend.extensions` points
straight at the HTP JSON.)

```json
{
  "graphs": [{ "graph_names": ["<real name from snpe-dlc-info>"],
               "O": 3, "vtcm_mb": 0, "hvx_threads": 4 }],
  "devices": [{ "soc_model": 72, "dsp_arch": "v81",
                "cores": [{ "core_id": 0, "perf_profile": "burst",
                            "rpc_control_latency": 100 }] }],
  "context": { "weight_sharing_enabled": true },
  "memory": { "mem_type": "shared_buffer" }
}
```

- **Unknown keys are not an error** — `libQnnHtpNetRunExtensions.so` only logs
  `Unknown Key: ... provided in HTP extensions config`, so a misspelled key
  silently does nothing. Grep the embedded schema:
  `strings "$LIB/libQnnHtpNetRunExtensions.so" | grep -A40 'schema validation'`.
- **Section placement matters** (both `HtpGraphConfig` and `HtpContextConfig` are
  `extra="forbid"` in the Python API): `extended_udma` and
  `weight_sharing_enabled` are **context**-level. A graph-level
  `"extended_udma": 1` is silently ignored.
- `graph_names` must match the name passed to `QnnGraph_create`. The vendor
  sample's `graph_name_1` / `graph_name_2` are placeholders — leave them in and
  the whole block never binds.
- `O`: 2 (default) or 3; level 1 is reserved for internal use. Schema type is
  `number`, so `3.0` is legal. Higher = longer prepare, faster execute.
- `vtcm_mb: 0` = device maximum, which **requires** `soc_model`/`soc_id` in
  `devices[]`. A `vtcm_size` in bytes **overrides** it (`Ignoring 'vtcm_mb',
  since 'vtcm_size' overrides and has been processed`). The vendor sample's
  `vtcm_mb: 2` is far below what v75-class parts have — raise it deliberately.
- `hvx_threads`: if omitted, offline prepare bakes 4. One of the few keys
  re-settable at runtime (`QnnGraph_setConfig` before first execute).
- **Drop `fp16_relaxed_precision`** — deprecated since 2.35.0 and explicitly
  ignored (`libQnnHtp.so`: "Precision custom config and fp16_relaxed_precision
  values are ignored in HTP backend").
- Use `soc_model`, not `soc_id` (deprecated; `soc_id: 0` selects nothing). Note
  `Qnn_SocModel_t` itself is on a deprecation path in favour of the published
  Supported Snapdragon Devices table.
- `pd_session`: `"unsigned"` (default) or `"secure"`. `perf_profile` default
  `"high_performance"`; `qnn-net-run --perf_profile` **overrides** the config.
  `rpc_control_latency` default 100 µs; `rpc_polling_time` default 9999 µs for
  burst/high_performance/sustained_high_performance and 0 otherwise;
  `hmx_timeout_us` default 300000.
- The minimal *runtime* form
  `{"devices":[{"cores":[{"perf_profile":"burst","rpc_control_latency":100}]}]}`
  is valid — it is literally what the SDK ships as
  `examples/Genie/configs/htp_backend_ext_config.json`. `soc_model`/`dsp_arch`
  are optional at runtime when executing pre-generated binaries. Do **not**
  generalize that to compile time.
- Inside the wrapper, `config_file_path` and `per_soc_config_file_path` are
  mutually exclusive, and `per_soc_config_file_path` cannot be combined with
  `--htp_socs`.

CLI overrides that beat the config file: `--vtcm_override <MB>` (unset or -1 → SoC
max), `--optimization_level_override <0-3>`, `--dlbc_override`,
`--hvx_threads_override`, `--reference_weight_sharing_enabled_override`.
Multi-SoC offline caches: `--htp_socs sm8350,sm8450,qcs6490` (comma-separated, no
whitespace). `--soc_model` **on this tool** sets the *simulated* soc model and is
not a targeting flag.

**DLBC and weight sharing are incompatible**: since 2.36, creating a binary with
both `dlbc_weights` and `weight_sharing_enabled` is unsupported (pre-2.36
binaries silently got DLBC WTS off). DLBC is offline-prepare only, unsupported
with spill-fill buffer sharing, and caps at 32 graphs (input OR weight
compression) / 16 (both).

**`extended_udma`** is the 64-bit far-mapping switch: **v81+ only**,
**prepare-time only** (`Ignoring context config to use extended UDMA which is
applicable to prepare only` means you set it too late), goes in the `context`
block as a JSON boolean `true` (not `1`). Grep at `--log_level verbose` for
`Enabled extended UDMA successfully` vs `Extended UDMA not enabled or not
supported by platform for graph %llu, skipping far mapping allocation` vs
`Failed to enable extended UDMA which is only available for DSP architectures
>= 81`. Watch for the same "skipping far mapping" tail from a different cause:
`Core spill-fill buffer size X MB is below far mapping minimum Y MB`.

## Legacy paths (still shipped)

SNPE DLC chain:

```bash
"$BIN/snpe-onnx-to-dlc" --input_network m.onnx --output_path m.dlc --input_dim 'obs' 1,48
"$BIN/snpe-dlc-quant"  --input_dlc m.dlc --output_dlc m_q.dlc --input_list calib.txt \
    [--act_bitwidth 16] [--weights_bitwidth 16]         # INT8 by default
"$BIN/snpe-dlc-graph-prepare" --input_dlc m_q.dlc --output_dlc m_q_htp.dlc --htp_socs <soc>
"$BIN/snpe-dlc-info" -i m_q_htp.dlc                     # confirm the HTP cache record
```

Note the flags are `--act_bitwidth` / `--weights_bitwidth` — *not* `--act_bw`.

`--htp_socs` bakes a per-chipset HTP execution-strategy cache; skipping it only
costs network-init time on device. `qcs6490` is the docs' RB3-Gen-2 example — pass
your **actual** SoC id (QCS9075 and QCS8275 do appear in the 2.46 identifier
list). If yours isn't listed, use `--htp_archs v68,v73,...`, which is mutually
exclusive with `--htp_socs` and `--vtcm_override`. No `snpe-*-converter` ships in
QAIRT 2.46 — produce the input `.dlc` with `qairt-converter`.

QNN one-step converters:

```bash
"$BIN/qnn-onnx-converter" --input_network m.onnx --output_path m.cpp \
  --input_dim 'obs' 1,48 --input_list calib.txt      # --input_list = quantize inline, INT8
"$BIN/qnn-model-lib-generator" -c m.cpp -b m.bin -o libs/ -t aarch64-oe-linux-gcc11.2
# -> libs/aarch64-oe-linux-gcc11.2/libm.so   (a model library, NOT a context binary)
```

Toolchain target must match the board rootfs: `aarch64-oe-linux-gcc11.2` for
Qualcomm Linux/Yocto, `aarch64-ubuntu-gcc9.4` for the Ubuntu Dragonwing images,
`x86_64-linux-clang` for host runs. The aarch64 target needs the Yocto/OE platform
SDK cross-toolchain sourced first (check `$SDKTARGETSYSROOT` and `$TARGET_PREFIX`
are non-empty) **before** `envsetup.sh`. For the x86 host target,
`sudo apt install libstdc++-12-dev` — the SDK's bundled clang uses GCC's
libstdc++ headers, and missing it shows up as "file not found" on `<limits>`.

Don't confuse the `.bin` emitted alongside the `.cpp` by `qnn-<fw>-converter`
(weights) with a **context**-binary `.bin`.

## soc_model / dsp_arch

Read them out of the SDK, never from a secondhand table. The `Qnn_SocModel_t`
integer enum in `include/QNN/QnnTypes.h` is explicitly `@deprecated ... will no
longer be updated` and has **no entry for QCS8275 or QCS9075** — target those by
SoC *name string* (`--target_soc_model qcs8275`, `--htp_socs qcs9075`) instead.
Full explanation and the enumeration commands: [soc-targets.md](soc-targets.md).

## Docs worth reading, in preference order

The SDK's own offline docs beat the web:

- `$QAIRT_SDK_ROOT/docs/QAIRT-Docs/QNN/general/{quantization,tools}.html` — full
  `usage:` blocks for every flag.
- `.../general/htp/htp_backend.html` — backend-extension schema, graph configs.
- `.../QNN/HTP/htp-network-design-recommendations/*`.

`docs.qualcomm.com` is client-side rendered, so a plain HTTP fetch returns only
the shell. The **Dragonwing** doc set never mentions `qairt-converter` /
`qairt-quantizer` at all (it uses the legacy `snpe-*` / `qnn-*` names) — that does
not mean the unified CLI is missing; 2.46/2.47 ship both in
`bin/x86_64-linux-clang`. The docs are behind.
