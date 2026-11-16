#!/usr/bin/env bash
# Resolve and optionally download the latest QAIRT (Qualcomm AI Runtime Community) SDK.
#
# Why this exists: the download URL embeds a full 4-part build id whose last field is
# the build DATE (YYMMDD), e.g.
#   .../Qualcomm_AI_Runtime_Community/All/2.52.0.261104/v2.52.0.261104.zip
# There is no listing API — the path is a plain object lookup (a bad key returns S3
# NoSuchKey), and softwarecenter.qualcomm.com itself is a client-rendered SPA, so
# scraping it without a browser yields only the page shell. Hardcoding a version in a
# doc guarantees it goes stale. So: discover, then verify.
#
# Strategy:
#   1. If `qai-hub` is installed AND configured, ask AI Hub which QAIRT versions it
#      accepts (hub.get_frameworks()) and use the newest as the sweep floor. AI Hub
#      lags the download site (2.51 vs 2.52 on 2026-11-16), so it is a floor, not
#      the answer.
#   2. Sweep minor versions and build dates with ranged GETs, walking UP from the
#      floor until minors stop resolving. No auth, no browser, no ceiling.
#
# Usage:
#   ./fetch-latest-qairt.sh                 # print the latest resolved version + URL
#   ./fetch-latest-qairt.sh --download DIR  # also download and unzip into DIR
#   ./fetch-latest-qairt.sh --from 2.46     # start the sweep at minor 46 (default 52)
#
# Verified 2026-11-16: latest = 2.52.0.261104.
set -uo pipefail

BASE="https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All"
FROM_MINOR=52          # newest minor known to exist; a stale floor only costs time
DOWNLOAD_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --download) DOWNLOAD_DIR="${2:?--download needs a target dir}"; shift 2 ;;
    --from)     FROM_MINOR="${2#2.}"; FROM_MINOR="${FROM_MINOR%%.*}"; shift 2 ;;
    -h|--help)  sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

url_for() { echo "$BASE/$1/v$1.zip"; }

# A ranged GET distinguishes exists (206) from missing (404).
# Do NOT use curl -I / HEAD here: this endpoint answers HEAD with 403.
exists() {
  local code
  code=$(curl -sL -r 0-0 -o /dev/null -w '%{http_code}' --max-time 10 "$(url_for "$1")" 2>/dev/null)
  [ "$code" = "206" ] || [ "$code" = "200" ]
}

# --- Strategy 1: ask AI Hub (authoritative, needs a configured token) ---------
from_aihub() {
  command -v python3 >/dev/null 2>&1 || return 1
  python3 - <<'PY' 2>/dev/null
import sys
try:
    import qai_hub as hub
    vs = [f.full_version for f in hub.get_frameworks() if f.name.upper() == "QAIRT"]
except Exception:
    sys.exit(1)
if not vs:
    sys.exit(1)
def key(v): return [int(x) for x in v.split(".")[:3]]
best = max(vs, key=key)
# full_version carries a long build stamp (2.52.0.261104115025); the download path
# uses only the first 6 digits of it (the YYMMDD date).
p = best.split(".")
print("%s.%s.%s.%s" % (p[0], p[1], p[2], p[3][:6]))
PY
}

# --- Strategy 2: sweep (no auth) ---------------------------------------------
# Probing is ~0.3 s per request, so the cost is request COUNT, not bandwidth. A
# release's build date is unknown, so a minor needs a whole date window probed —
# serially that is minutes. Fan out with xargs -P instead: one minor's full
# ~365-day window resolves in a couple of seconds.
#
# Walk up from FROM_MINOR and stop after two consecutive minors with no build
# (one gap tolerated, since minors are occasionally skipped). Build dates can't be
# in the future, so only this year and last year are probed.
probe_minor() {
  # echo the newest full version found for minor $1, or nothing.
  local minor="$1" y m d this last
  this=$(date +%y); last=$(printf '%02d' $(( 10#$this - 1 )))
  {
    for y in "$this" "$last"; do
      for m in 12 11 10 09 08 07 06 05 04 03 02 01; do
        for d in $(seq -w 1 31); do echo "2.${minor}.0.${y}${m}${d}"; done
      done
    done
  } | xargs -P 24 -I{} sh -c \
      'c=$(curl -sL -r 0-0 -o /dev/null -w "%{http_code}" --max-time 10 "'"$BASE"'/{}/v{}.zip" 2>/dev/null); [ "$c" = 206 ] && echo {}' \
    | sort -t. -k4,4nr | head -1
}

from_sweep() {
  # Walk UP from the floor, keeping the newest hit. There is deliberately no
  # ceiling: a hardcoded one silently caps the answer once releases pass it.
  local minor hit best="" misses=0
  for (( minor=FROM_MINOR; misses<2; minor++ )); do
    hit="$(probe_minor "$minor")"
    if [ -n "$hit" ]; then best="$hit"; misses=0; else misses=$((misses + 1)); fi
  done
  [ -n "$best" ] && { echo "$best"; return 0; }
  return 1
}

HUB_VER="$(from_aihub || true)"
if [ -n "$HUB_VER" ]; then
  hub_minor="${HUB_VER#2.}"; hub_minor="${hub_minor%%.*}"
  [ "$hub_minor" -gt "$FROM_MINOR" ] && FROM_MINOR="$hub_minor"
fi
echo "resolving latest QAIRT by probing from 2.$FROM_MINOR up..." >&2
VER="$(from_sweep || true)"
SRC="ranged-GET sweep"
if [ -z "${VER:-}" ] && [ -n "$HUB_VER" ] && exists "$HUB_VER"; then
  VER="$HUB_VER"; SRC="AI Hub get_frameworks()"
fi

if [ -z "${VER:-}" ]; then
  cat >&2 <<EOF
Could not resolve a QAIRT version.
Check connectivity, or pass a known build explicitly:
  VER=2.52.0.261104
  curl -L -O "$BASE/\$VER/v\$VER.zip"
EOF
  exit 1
fi

echo "QAIRT version : $VER   (via $SRC)"
echo "download URL  : $(url_for "$VER")"
if [ -n "$HUB_VER" ] && [ "$HUB_VER" != "$VER" ]; then
  echo "note: AI Hub's newest accepted QAIRT is $HUB_VER; use that one if a local artifact must match an AI Hub job"
fi

[ -z "$DOWNLOAD_DIR" ] && exit 0

mkdir -p "$DOWNLOAD_DIR" && cd "$DOWNLOAD_DIR"
echo "downloading (~1.8 GB) ..."
curl -L --fail --retry 3 -O "$(url_for "$VER")"
unzip -q -o "v$VER.zip"
# The archive extracts to qairt/<version>/ — NOT v<version>/qairt/.
ROOT="$PWD/qairt/$VER"
[ -d "$ROOT" ] || ROOT="$(find "$PWD" -maxdepth 3 -type d -name "$VER" | head -1)"
cat <<EOF

Extracted to: $ROOT

Next:
  export QAIRT_ROOT="$ROOT"
  source "\$QAIRT_ROOT/bin/envsetup.sh"     # sets QAIRT_SDK_ROOT, PATH, PYTHONPATH, LD_LIBRARY_PATH
  sudo bash "\$QAIRT_ROOT/bin/check-linux-dependency.sh"
  "\$QAIRT_ROOT/bin/check-python-dependency"
  # then, separately (NOT covered by check-python-dependency):
  pip install onnx onnxruntime onnxsim
EOF
