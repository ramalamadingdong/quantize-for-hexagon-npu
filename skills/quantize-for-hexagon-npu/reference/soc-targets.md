# SoC targeting: name strings, dsp_arch, soc_model

## There are two targeting mechanisms, and only one of them still gets updated

**1. SoC *name string* — prefer this.** Taken by
`qairt-converter --target_soc_model <name>` and
`snpe-dlc-graph-prepare --htp_socs <name>[,<name>...]`. This is the live,
maintained surface and it covers current Dragonwing parts.

**2. `soc_model` *integer*** — the `Qnn_SocModel_t` enum, used by
`devices[].soc_model` in the HTP backend-extensions JSON and by ORT's `soc_model`
provider option. `QnnTypes.h` marks it:

```c
/**
 * @brief An enum which identifies SOC models.
 *
 * @deprecated This enumeration will no longer be updated.
 */
```

That deprecation is not cosmetic — it is why newer parts have **no integer at
all** (see below). Do not go hunting for one.

Separately, **Linux `soc_id`** (`/sys/devices/soc0/soc_id`, e.g. 675) is a third
unrelated number. You never pass it to QNN, but note
`libQnnHtpNetRunExtensions.so` reads `/sys/devices/soc0/soc_id` itself to
auto-detect the device — which is why `dsp_arch` alone is often enough at runtime.

## Read it out of the SDK you compile with — don't trust a table

Everything below is reproducible from a QAIRT install. `$QAIRT_SDK_ROOT` = your SDK.

```bash
# (a) the authoritative SoC NAME list the tools accept
strings "$QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
  | grep -oE '^(QCS|SM|SA|QCM|QRB|SC|SXR|SAR|CQ|IPQ|XRV|SW|IQX|X1E)[0-9]{3,5}[A-Z]*$' \
  | sort -u                                     # 149 entries in 2.46.0.260424

# (b) the frozen soc_model integers
grep -nE 'QNN_SOC_MODEL_[A-Z0-9_]+ *=' "$QAIRT_SDK_ROOT/include/QNN/QnnTypes.h"

# (c) which Hexagon arches this SDK can even build for
ls -d "$QAIRT_SDK_ROOT"/lib/hexagon-v* | xargs -n1 basename
# 2.46.0.260424: v66 v68 v69 v73 v75 v79 v81
```

`snpe-dlc-graph-prepare` carries the same name list — `strings` it and grep if you
want a second confirmation that your part is spelled right before a long compile.

**Do not use `htp_v2.json` for this.** It is an op-kernel/dtype-support database
(~1.8 MB of per-op `supported_kernels`), not a chipset→arch map.

## The parts in play

`dsp_arch` (QAIRT config) and `htp_arch` (ORT provider option) are the same
number, spelled `"v75"` vs `"75"`.

| Part | Name string valid in 2.46? | dsp_arch | `soc_model` int | Notes |
|---|---|---|---|---|
| QCS8275 (Dragonwing IQ-8275 EVK) | ✅ `QCS8275` | v75 | **none — not in the frozen enum** | family sibling of QCS8300 / SA8620P |
| QCS9075 (Dragonwing IQ-9075 EVK) | ✅ `QCS9075` | v73 | **none — not in the frozen enum** | family sibling of QCS9100 / SA8255 / SA8775P |
| QCS6490 (RB3 Gen 2, RUBIK Pi 3) | ✅ `QCS6490` | v68 | — | |
| SA8797 / QAM8797P (automotive) | ✅ `SA8797` | v81 | 72 | Linux `soc_id` 690 |
| QCS8300 | ✅ | v75 | 82 | |
| SA8620P | ✅ | v75 | 67 | |
| QCS9100 | ✅ | v73 | 77 | |
| QCS8550 | ✅ | — | 66 | |
| SM8550 / SD 8 Gen 2 | ✅ | v73 | 43 | |
| SM8650 / SD 8 Gen 3 | ✅ | v75 | 57 | |
| SM8750 / SD 8 Elite | ✅ | v79 | 69 | |
| SM8850 / SD 8 Elite Gen 5 | ✅ | v81 | 87 | |
| SC8380XP / X Elite | ✅ | v73 | 60 | aka `X1E80100` |

Confirmed by grep in 2.46.0.260424: **`QnnTypes.h` contains no `8275` and no
`9075` anywhere.** The enum ends at `QNN_SOC_MODEL_SM8850 = 87` (with gaps at
14, 22, 44). So for the two Dragonwing EVKs:

- Target them **by name**: `--target_soc_model qcs8275` / `--htp_socs qcs9075`.
- In the HTP backend-extensions JSON, `soc_model` is schema-typed `integer` with
  no string alias, so you **cannot** name them there. Set `dsp_arch` (typed
  `string`) and let the backend auto-detect the SoC, or target by name at the tool
  level instead. `dsp_arch` alone is legal at runtime for pre-generated binaries.

### Why 67, 82 and 77 keep showing up for these boards

Not a bug and not a collision — **family aliasing**. `libQnnHtp.so`'s name table
groups part numbers under chip-family codenames, and the adjacency is:

- `MonacoAU` → **QCS8275, QCS8300, SA8620P**
- `Lemans` → **QCS9100, SA8255, SA8775P, QCS9075**

Two independent corroborations for the QCS8275/QCS8300 pairing:

1. The board's device-tree `compatible` is a **fallback list** naming both:
   `qcom,qcs8275` *and* `qcs8300`.
2. The QAIRT runtime deb ships the QCS8275's own v75 skel under a QCS8300 path:
   `/usr/share/qcom/qcs8300/Qualcomm/QCS8300-RIDE/dsp/cdsp/libQnnHtpV75Skel.so`.

So when `devices_and_chipsets.yaml` maps QCS8275 → 67 (`SA8620P`), or AI Hub emits
77 (`QCS9100`) for `qualcomm_qcs9075`, they are substituting the nearest
*enumerated* family member because the enum was frozen before these part numbers
shipped. That is a reasonable fallback, but it means the integer identifies the
**family**, not your exact part — so prefer the name string wherever a tool
accepts one, and validate on device either way.

*(The codename→parts grouping is read off adjacency in the binary's string table;
the pattern is not perfectly regular across all families, so treat the two
groupings above as well-corroborated and don't extrapolate the method.)*

## Gotchas

- **Board numbers don't order by arch.** IQ-8275 is v75; IQ-9075 is v73.
- **Don't copy `soc_model 43` for QCS9075** because a Dragonwing container sample
  ships it. 43 is SM8550 — a different family that merely happens to also be v73.
  If you need an integer for a Lemans part, 77 is the family member; better, target
  by name.
- **Same `dsp_arch` ≠ binary portability.** VTCM size, HVX/NSP core count and clock
  plan differ across families, and `QnnContext_createFromBinary` runs a
  STRICT/PERMISSIVE compatibility check. Counter-example (n=1, diagnostic only):
  an AI Hub Qwen3-8B W4A16 bundle built with QAIRT 2.45 for `soc_model 87`
  executed correctly on a QAIRT 2.46 SA8797P (`soc_model 72`), all 5 parts on HTP
  — matching v81 sufficed. Treat that as a cheap way to isolate which failure axis
  dominates, never as a supported configuration.
- **Prebuilt bundles cannot be retargeted by editing JSON.** `soc_model` /
  `dsp_arch` are baked into each `.bin` at `qnn-context-binary-generator` time; the
  runtime HTP JSON only declares the *current* device.
- **ORT `htp_arch` accepts only `"0"|"68"|"69"|"73"|"75"|"81"`** as of onnxruntime
  main (2026-07) — **v79 is absent**, even though the SDK ships `hexagon-v79`.
  Anything else logs a WARNING and leaves `QNN_HTP_DEVICE_ARCH_NONE`. Grep
  `ParseHtpArchitecture` in your ORT source (verify for your version).
- On QCS8275, ORT requires `htp_arch:"75"` explicitly or device creation fails
  with `QNN_DEVICE_ERROR_INVALID_CONFIG`.

## VTCM

VTCM is on-chip and **cannot be grown from DDR**. Measured on QCS8275 (v75, QAIRT
2.46, ORT-QNN 1.23) by sweeping the ORT `vtcm_mb` provider option: 1,2,4,6,7,8
create a session; 9,10,12,16,24,32,64 fail immediately with `FAIL : Failed to
initialize qnn_model_wrapper` — so **8 MB** for that board/SDK pair.

The sweep measures *what the backend accepts for the detected SoC*, not physical
silicon, and v75 is a class spanning families (SM8650 is also v75 and a very
different tier) — re-run the probe per part. Probe code is in
[ort-qnn-ep.md](ort-qnn-ep.md). For reference, QNN's own htp_config sample requests
only `vtcm_mb: 2`.

## Dual-NPU targeting

`"device_id": 0|1` inside `devices[]` of `htp_backend_ext_config.json`, documented
for the **IQ-9075 EVK only** ("only work on the IQ-9075 EVK as it has dual NPUs";
RB3 Gen 2 and RUBIK Pi 3 excluded).

ORT's separate `device_id` provider option is documented merely as "the device to
use when setting `htp_arch`". On QCS8275, `device_id` 0/1/None all create a session
with no error, which proves **nothing** — ORT may silently alias to NPU0. The
decisive test (two concurrent burst contexts pinned to 0 and 1; does the
VTCM-contention 6001 disappear?) has not been run.
