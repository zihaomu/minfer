#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${ROOT_DIR}/build/minfer_benchmark"

MODEL_PATH=""
PROMPT_LEN=128
DECODE_TOKENS=128
WARMUP=1
RUNS=5
RUNS_PROFILE=3
PROGRESS_INTERVAL=128

THREADS_T4=4
THREADS_T16=16

DECODE_T4_MAX_MS="${MINFER_GATE_DECODE_T4_MAX_MS:-2.20}"
DECODE_T16_MAX_MS="${MINFER_GATE_DECODE_T16_MAX_MS:-1.30}"
LINEAR_DECODE_MAX_MS="${MINFER_GATE_LINEAR_DECODE_MAX_MS:-0.60}"

USE_AFFINITY=1
KEEP_LOGS=0
LOG_DIR=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Performance gate for decode hotspot regression checks.

Defaults:
  decode@threads=${THREADS_T4} <= ${DECODE_T4_MAX_MS} ms/tok
  decode@threads=${THREADS_T16} <= ${DECODE_T16_MAX_MS} ms/tok
  LinearLayer_19(decode avg, threads=${THREADS_T4}) <= ${LINEAR_DECODE_MAX_MS} ms

Options:
  --binary <path>              minfer_benchmark binary (default: ${BINARY})
  --model <path>               model path (default: minfer_benchmark internal default)
  --prompt-len <n>             prompt length (default: ${PROMPT_LEN})
  --decode-tokens <n>          decode tokens (default: ${DECODE_TOKENS})
  --warmup <n>                 warmup runs (default: ${WARMUP})
  --runs <n>                   runs for decode checks (default: ${RUNS})
  --runs-profile <n>           runs for layer-profile check (default: ${RUNS_PROFILE})
  --progress-interval <n>      decode progress interval (default: ${PROGRESS_INTERVAL})
  --threads-t4 <n>             thread count for low-thread gate (default: ${THREADS_T4})
  --threads-t16 <n>            thread count for high-thread gate (default: ${THREADS_T16})
  --decode-t4-max <float>      max decode ms/tok for low-thread gate
  --decode-t16-max <float>     max decode ms/tok for high-thread gate
  --linear-decode-max <float>  max LinearLayer_19 decode avg ms
  --no-affinity                do not set OMP_PROC_BIND/OMP_PLACES
  --log-dir <dir>              keep logs under this directory
  -h, --help                   show help

Environment overrides:
  MINFER_GATE_DECODE_T4_MAX_MS
  MINFER_GATE_DECODE_T16_MAX_MS
  MINFER_GATE_LINEAR_DECODE_MAX_MS
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)
            BINARY="${2:?missing value for --binary}"
            shift 2
            ;;
        --model)
            MODEL_PATH="${2:?missing value for --model}"
            shift 2
            ;;
        --prompt-len)
            PROMPT_LEN="${2:?missing value for --prompt-len}"
            shift 2
            ;;
        --decode-tokens)
            DECODE_TOKENS="${2:?missing value for --decode-tokens}"
            shift 2
            ;;
        --warmup)
            WARMUP="${2:?missing value for --warmup}"
            shift 2
            ;;
        --runs)
            RUNS="${2:?missing value for --runs}"
            shift 2
            ;;
        --runs-profile)
            RUNS_PROFILE="${2:?missing value for --runs-profile}"
            shift 2
            ;;
        --progress-interval)
            PROGRESS_INTERVAL="${2:?missing value for --progress-interval}"
            shift 2
            ;;
        --threads-t4)
            THREADS_T4="${2:?missing value for --threads-t4}"
            shift 2
            ;;
        --threads-t16)
            THREADS_T16="${2:?missing value for --threads-t16}"
            shift 2
            ;;
        --decode-t4-max)
            DECODE_T4_MAX_MS="${2:?missing value for --decode-t4-max}"
            shift 2
            ;;
        --decode-t16-max)
            DECODE_T16_MAX_MS="${2:?missing value for --decode-t16-max}"
            shift 2
            ;;
        --linear-decode-max)
            LINEAR_DECODE_MAX_MS="${2:?missing value for --linear-decode-max}"
            shift 2
            ;;
        --no-affinity)
            USE_AFFINITY=0
            shift
            ;;
        --log-dir)
            LOG_DIR="${2:?missing value for --log-dir}"
            KEEP_LOGS=1
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -x "${BINARY}" ]]; then
    echo "Binary not found or not executable: ${BINARY}" >&2
    exit 1
fi

if (( KEEP_LOGS )); then
    mkdir -p "${LOG_DIR}"
else
    LOG_DIR="$(mktemp -d "${ROOT_DIR}/benchmark/perf_gate_tmp.XXXXXX")"
fi

cleanup() {
    if (( KEEP_LOGS == 0 )); then
        rm -rf "${LOG_DIR}"
    fi
}
trap cleanup EXIT

build_common_args() {
    local threads="$1"
    local runs="$2"
    local enable_layer_profile="${3:-0}"

    local -a args
    args=(
        "--threads" "${threads}"
        "--prompt-lens" "${PROMPT_LEN}"
        "--decode-tokens" "${DECODE_TOKENS}"
        "--warmup" "${WARMUP}"
        "--runs" "${runs}"
        "--progress-interval" "${PROGRESS_INTERVAL}"
    )
    if [[ -n "${MODEL_PATH}" ]]; then
        args+=("--model" "${MODEL_PATH}")
    fi
    if [[ "${enable_layer_profile}" == "1" ]]; then
        args+=("--layer-profile")
    fi
    printf '%s\n' "${args[@]}"
}

run_bench() {
    local log_file="$1"
    shift
    local -a args=("$@")

    if (( USE_AFFINITY )); then
        OMP_PROC_BIND=close OMP_PLACES=cores "${BINARY}" "${args[@]}" | tee "${log_file}" >/dev/null
    else
        "${BINARY}" "${args[@]}" | tee "${log_file}" >/dev/null
    fi
}

extract_decode_avg_ms() {
    local log_file="$1"
    awk '
        /^decode  : avg=/ {
            line = $0
        }
        END {
            if (line == "") {
                exit 1
            }
            sub(/^decode  : avg=/, "", line)
            split(line, a, " ")
            print a[1]
        }
    ' "${log_file}"
}

extract_linear_decode_avg_ms() {
    local log_file="$1"
    awk -F'│' '
        /── Decode ──/ {
            in_decode = 1
            next
        }
        /── Prefill ──/ {
            in_decode = 0
        }
        in_decode && /LinearLayer_19/ {
            value = $4
            gsub(/^[ \t]+|[ \t]+$/, "", value)
            print value
            found = 1
            exit
        }
        END {
            if (!found) {
                exit 1
            }
        }
    ' "${log_file}"
}

float_le() {
    local lhs="$1"
    local rhs="$2"
    awk -v a="${lhs}" -v b="${rhs}" 'BEGIN { exit (a <= b) ? 0 : 1 }'
}

echo "== Perf Gate: decode@${THREADS_T4}, decode@${THREADS_T16}, LinearLayer_19 decode =="
echo "Logs: ${LOG_DIR}"

mapfile -t args_t4 < <(build_common_args "${THREADS_T4}" "${RUNS}" 0)
mapfile -t args_t16 < <(build_common_args "${THREADS_T16}" "${RUNS}" 0)
mapfile -t args_profile < <(build_common_args "${THREADS_T4}" "${RUNS_PROFILE}" 1)

LOG_T4="${LOG_DIR}/decode_t${THREADS_T4}.log"
LOG_T16="${LOG_DIR}/decode_t${THREADS_T16}.log"
LOG_PROFILE="${LOG_DIR}/layer_profile_t${THREADS_T4}.log"

run_bench "${LOG_T4}" "${args_t4[@]}"
run_bench "${LOG_T16}" "${args_t16[@]}"
run_bench "${LOG_PROFILE}" "${args_profile[@]}"

DECODE_T4_MS="$(extract_decode_avg_ms "${LOG_T4}")"
DECODE_T16_MS="$(extract_decode_avg_ms "${LOG_T16}")"
LINEAR_DECODE_MS="$(extract_linear_decode_avg_ms "${LOG_PROFILE}")"

STATUS=0

check_and_print() {
    local label="$1"
    local value="$2"
    local limit="$3"
    if float_le "${value}" "${limit}"; then
        printf "[PASS] %s: %s <= %s\n" "${label}" "${value}" "${limit}"
    else
        printf "[FAIL] %s: %s > %s\n" "${label}" "${value}" "${limit}" >&2
        STATUS=1
    fi
}

check_and_print "decode_avg_ms@threads=${THREADS_T4}" "${DECODE_T4_MS}" "${DECODE_T4_MAX_MS}"
check_and_print "decode_avg_ms@threads=${THREADS_T16}" "${DECODE_T16_MS}" "${DECODE_T16_MAX_MS}"
check_and_print "LinearLayer_19_decode_avg_ms@threads=${THREADS_T4}" "${LINEAR_DECODE_MS}" "${LINEAR_DECODE_MAX_MS}"

if (( STATUS == 0 )); then
    echo "Perf gate PASSED."
else
    echo "Perf gate FAILED." >&2
fi

exit "${STATUS}"
