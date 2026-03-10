#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${ROOT_DIR}/build/minfer_op_benchmark"
WARMUP=10
ITERS=30
THREADS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
GROUP="all"
MODE="entry"
FILTER=""
OUTPUT_DIR=""
LIST_ONLY=0
DRY_RUN=0
REPORT_FILE=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --group <name>       decode | prefill | projection | tail | all
  --mode <name>        entry | micro | runtime | both | full
  --threads <n>        benchmark threads (default: detected CPU count)
  --warmup <n>         warmup iterations (default: ${WARMUP})
  --iters <n>          measured iterations (default: ${ITERS})
  --filter <text>      only run cases whose name contains the text
  --binary <path>      benchmark binary path (default: ${BINARY})
  --output-dir <path>  save logs under this directory
  --list               list cases and exit
  --dry-run            print commands without executing
  -h, --help           show this help

Examples:
  $(basename "$0") --group decode --mode both --threads 32
  $(basename "$0") --group decode --mode runtime --threads 32
  $(basename "$0") --group tail --filter 4096 --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --group)
            GROUP="${2:?missing value for --group}"
            shift 2
            ;;
        --mode)
            MODE="${2:?missing value for --mode}"
            shift 2
            ;;
        --threads)
            THREADS="${2:?missing value for --threads}"
            shift 2
            ;;
        --warmup)
            WARMUP="${2:?missing value for --warmup}"
            shift 2
            ;;
        --iters)
            ITERS="${2:?missing value for --iters}"
            shift 2
            ;;
        --filter)
            FILTER="${2:?missing value for --filter}"
            shift 2
            ;;
        --binary)
            BINARY="${2:?missing value for --binary}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?missing value for --output-dir}"
            shift 2
            ;;
        --list)
            LIST_ONLY=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
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

case "${GROUP}" in
    decode|prefill|projection|tail|all) ;;
    *)
        echo "Invalid --group: ${GROUP}" >&2
        exit 1
        ;;
esac

case "${MODE}" in
    entry|micro|runtime|both|full) ;;
    *)
        echo "Invalid --mode: ${MODE}" >&2
        exit 1
        ;;
esac

if [[ ! -x "${BINARY}" ]]; then
    echo "Benchmark binary not found or not executable: ${BINARY}" >&2
    exit 1
fi

declare -a CASES=()

add_case() {
    CASES+=("$1|$2|$3|$4|$5|$6")
}

add_case "decode" "decode_m1_h512_ffn" 1 1 512 2048
add_case "decode" "decode_m1_h1024_ffn" 1 1 1024 4096
add_case "decode" "decode_m1_h2048_ffn" 1 1 2048 8192
add_case "decode" "decode_m2_h2048_ffn" 2 1 2048 8192
add_case "decode" "decode_m4_h4096_ffn" 4 1 4096 16384
add_case "decode" "decode_m8_h4096_ffn" 8 1 4096 16384

add_case "prefill" "prefill_m16_h512_ffn" 1 16 512 2048
add_case "prefill" "prefill_m32_h1024_ffn" 1 32 1024 4096
add_case "prefill" "prefill_m64_h2048_ffn" 1 64 2048 8192
add_case "prefill" "prefill_m128_h2048_ffn" 1 128 2048 8192
add_case "prefill" "prefill_m256_h4096_ffn" 1 256 4096 16384
add_case "prefill" "prefill_m512_h4096_ffn" 1 512 4096 16384

add_case "projection" "attn_qkv_h512" 1 128 512 1536
add_case "projection" "attn_o_h512" 1 128 512 512
add_case "projection" "ffn_up_h512" 1 128 512 2048
add_case "projection" "ffn_down_h512" 1 128 2048 512
add_case "projection" "attn_qkv_h4096" 1 128 4096 12288
add_case "projection" "attn_o_h4096" 1 128 4096 4096
add_case "projection" "ffn_up_h4096" 1 128 4096 16384
add_case "projection" "ffn_down_h4096" 1 128 16384 4096

add_case "tail" "tail_small_63_63_63" 1 63 63 63
add_case "tail" "tail_small_64_64_64" 1 64 64 64
add_case "tail" "tail_small_65_65_65" 1 65 65 65
add_case "tail" "tail_prefill_127_4096_4095" 1 127 4096 4095
add_case "tail" "tail_prefill_128_4096_4096" 1 128 4096 4096
add_case "tail" "tail_prefill_129_4096_4097" 1 129 4096 4097
add_case "tail" "tail_decode_m1_k4096_n4095" 1 1 4096 4095
add_case "tail" "tail_decode_m1_k4096_n4096" 1 1 4096 4096
add_case "tail" "tail_decode_m1_k4096_n4097" 1 1 4096 4097

should_keep_case() {
    local case_group="$1"
    local case_name="$2"
    if [[ "${GROUP}" != "all" && "${GROUP}" != "${case_group}" ]]; then
        return 1
    fi
    if [[ -n "${FILTER}" && "${case_name}" != *"${FILTER}"* ]]; then
        return 1
    fi
    return 0
}

list_cases() {
    printf "group\tname\tbatch\tseq_len\thidden\tgemm_out\n"
    for entry in "${CASES[@]}"; do
        IFS='|' read -r case_group case_name batch seq_len hidden gemm_out <<< "${entry}"
        if should_keep_case "${case_group}" "${case_name}"; then
            printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${case_group}" "${case_name}" "${batch}" "${seq_len}" "${hidden}" "${gemm_out}"
        fi
    done
}

if (( LIST_ONLY )); then
    list_cases
    exit 0
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${ROOT_DIR}/benchmark/results/gemm/$(date +%Y%m%d_%H%M%S)"
fi

detect_commit() {
    git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo "unknown"
}

detect_build_type() {
    if [[ -f "${ROOT_DIR}/build/CMakeCache.txt" ]]; then
        sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' "${ROOT_DIR}/build/CMakeCache.txt" | head -n 1
    fi
}

detect_cpu_model() {
    if command -v lscpu >/dev/null 2>&1; then
        lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -n 1
        return
    fi
    sed -n 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo 2>/dev/null | head -n 1
}

TIMESTAMP="$(date -Iseconds)"
GIT_COMMIT="$(detect_commit)"
BUILD_TYPE="$(detect_build_type)"
CPU_MODEL="$(detect_cpu_model)"

if (( ! DRY_RUN )); then
    mkdir -p "${OUTPUT_DIR}"
    REPORT_FILE="${OUTPUT_DIR}/report.md"

    {
        printf "timestamp=%s\n" "${TIMESTAMP}"
        printf "commit=%s\n" "${GIT_COMMIT}"
        printf "cpu_model=%s\n" "${CPU_MODEL}"
        printf "build_type=%s\n" "${BUILD_TYPE}"
        printf "binary=%s\n" "${BINARY}"
        printf "group=%s\n" "${GROUP}"
        printf "mode=%s\n" "${MODE}"
        printf "threads=%s\n" "${THREADS}"
        printf "warmup=%s\n" "${WARMUP}"
        printf "iters=%s\n" "${ITERS}"
        printf "filter=%s\n" "${FILTER}"
        printf "pwd=%s\n" "${ROOT_DIR}"
    } > "${OUTPUT_DIR}/run_config.txt"

    printf "group\tmode\tname\tbatch\tseq_len\thidden\tgemm_out\tlog\n" > "${OUTPUT_DIR}/index.tsv"

    {
        printf "# GEMM Benchmark Report\n\n"
        printf "## Context\n\n"
        printf -- "- Timestamp: %s\n" "${TIMESTAMP}"
        printf -- "- Commit: %s\n" "${GIT_COMMIT}"
        printf -- "- CPU: %s\n" "${CPU_MODEL}"
        printf -- "- Build: %s\n" "${BUILD_TYPE}"
        printf -- "- Binary: %s\n" "${BINARY}"
        printf -- "- Group: %s\n" "${GROUP}"
        printf -- "- Mode: %s\n" "${MODE}"
        printf -- "- Threads: %s\n" "${THREADS}"
        printf -- "- Warmup: %s\n" "${WARMUP}"
        printf -- "- Iters: %s\n" "${ITERS}"
        printf -- "- Filter: %s\n\n" "${FILTER}"
        printf "## Results\n\n"
    } > "${REPORT_FILE}"
fi

run_one() {
    local mode_name="$1"
    local case_group="$2"
    local case_name="$3"
    local batch="$4"
    local seq_len="$5"
    local hidden="$6"
    local gemm_out="$7"
    local mode_flag=""
    local log_file=""
    local cmd=()

    case "${mode_name}" in
        entry) mode_flag="--gemm-only" ;;
        micro) mode_flag="--gemm-micro-only" ;;
        runtime) mode_flag="--gemm-runtime-only" ;;
        *)
            echo "Unsupported mode: ${mode_name}" >&2
            exit 1
            ;;
    esac

    log_file="${OUTPUT_DIR}/${mode_name}_${case_group}_${case_name}.log"
    cmd=(
        "${BINARY}"
        "${mode_flag}"
        --warmup "${WARMUP}"
        --iters "${ITERS}"
        --threads "${THREADS}"
        --head-count 1
        --head-count-kv 1
        --batch "${batch}"
        --seq-len "${seq_len}"
        --hidden "${hidden}"
        --gemm-out "${gemm_out}"
    )

    printf '$'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    if (( DRY_RUN )); then
        return 0
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${case_group}" "${mode_name}" "${case_name}" "${batch}" "${seq_len}" "${hidden}" "${gemm_out}" "${log_file}" \
        >> "${OUTPUT_DIR}/index.tsv"

    "${cmd[@]}" | tee "${log_file}"

    {
        printf "### %s (%s)\n\n" "${case_name}" "${mode_name}"
        printf -- "- Command:\n\n"
        printf '%s\n' '```bash'
        printf '%q ' "${cmd[@]}"
        printf '\n'
        printf '%s\n\n' '```'
        printf -- "- Log: %s\n\n" "${log_file#${ROOT_DIR}/}"
        printf '%s\n' '```text'
        grep 'avg=' "${log_file}" || true
        printf '%s\n\n' '```'
    } >> "${REPORT_FILE}"
}

MATCHED=0

for entry in "${CASES[@]}"; do
    IFS='|' read -r case_group case_name batch seq_len hidden gemm_out <<< "${entry}"
    if ! should_keep_case "${case_group}" "${case_name}"; then
        continue
    fi
    MATCHED=1
    case "${MODE}" in
        entry)
            run_one entry "${case_group}" "${case_name}" "${batch}" "${seq_len}" "${hidden}" "${gemm_out}"
            ;;
        micro)
            run_one micro "${case_group}" "${case_name}" "${batch}" "${seq_len}" "${hidden}" "${gemm_out}"
            ;;
        runtime)
            run_one runtime "${case_group}" "${case_name}" "${batch}" "${seq_len}" "${hidden}" "${gemm_out}"
            ;;
        both)
            run_one entry "${case_group}" "${case_name}" "${batch}" "${seq_len}" "${hidden}" "${gemm_out}"
            run_one micro "${case_group}" "${case_name}" "${batch}" "${seq_len}" "${hidden}" "${gemm_out}"
            ;;
        full)
            run_one entry "${case_group}" "${case_name}" "${batch}" "${seq_len}" "${hidden}" "${gemm_out}"
            run_one micro "${case_group}" "${case_name}" "${batch}" "${seq_len}" "${hidden}" "${gemm_out}"
            run_one runtime "${case_group}" "${case_name}" "${batch}" "${seq_len}" "${hidden}" "${gemm_out}"
            ;;
    esac
done

if (( ! MATCHED )); then
    echo "No cases matched the requested filters." >&2
    exit 1
fi

if (( DRY_RUN )); then
    echo "Dry run only; planned output directory: ${OUTPUT_DIR}"
else
    echo "Saved logs under ${OUTPUT_DIR}"
fi
