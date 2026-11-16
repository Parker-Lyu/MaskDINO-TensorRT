#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: $0 <archived-stage3-runs> <output-directory>" >&2
  exit 2
fi
plan_root=$(realpath "$1")
mkdir -p "$2"
run_dir=$(realpath "$2")
selectors=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
image=sha256:24a707391a7592235352cb91d58e11da356aefe3f7512a698a5402671cd83d1e
input="$plan_root/layer-profile-b1/149770.f32"

for label in control fpn ffn; do
  case "$label" in
    control) plan="$plan_root/fpn-cache-pair/control/network.engine";;
    fpn) plan="$plan_root/fpn-cache-pair/fpn-half/network.engine";;
    ffn) plan="$plan_root/ffn-full-half/build/network.engine";;
  esac
  filters=()
  while IFS=$'\t' read -r layer kernel; do
    case "$layer" in
      __myl_*|*/MatMul_myl*) domain=myelin-exec;;
      *) domain=TensorRT;;
    esac
    layer=${layer//\\/\\\\\\\\}
    for special in '/' '[' ']' ',' '@' '*' '+'; do
      layer=${layer//"$special"/\\$special}
    done
    filters+=(--nvtx-include "$domain@$layer/")
  done < "$selectors/$label.targets.txt"
  count=$(wc -l < "$selectors/$label.targets.txt")
  docker run --rm --gpus all --network none --cap-add SYS_ADMIN -e HOME=/tmp \
    -v "$plan:/network.engine:ro" -v "$input:/input.f32:ro" -v "$run_dir:/out" -w /out \
    --entrypoint bash "$image" -c '
      set -eu
      label=$1
      count=$2
      shift 2
      metrics=(
        gpu__time_duration.sum
        gpc__cycles_elapsed.avg.per_second
        sm__throughput.avg.pct_of_peak_sustained_elapsed
        sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed
        sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
        sm__warps_active.avg.pct_of_peak_sustained_active
        sm__issue_active.avg.pct_of_peak_sustained_elapsed
        smsp__warps_eligible.avg.per_cycle_active
        smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
        smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
        smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
        lts__throughput.avg.pct_of_peak_sustained_elapsed
        lts__t_bytes.sum
        lts__t_sector_hit_rate.pct
        gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
        dram__bytes_read.sum
        dram__bytes_write.sum
      )
      metric_list=$(IFS=,; printf "%s" "${metrics[*]}")
      ncu --version
      printf "NVTX filter argument: %s\n" "$@"
      ncu --nvtx "$@" --replay-mode application --app-replay-mode strict \
        --clock-control none --cache-control none \
        --launch-skip "$((8 * count))" --launch-count "$((3 * count))" \
        --section LaunchStats \
        --metrics "$metric_list" \
        --export "$label" /opt/tensorrt/bin/trtexec --loadEngine=/network.engine \
        --loadInputs=image:/input.f32 --infStreams=1 --noDataTransfers \
        --warmUp=0 --duration=0 --iterations=12
      ncu --import "$label.ncu-rep" --page raw --csv --print-units base --print-kernel-base function > "$label.ncu.csv"
    ' _ "$label" "$count" "${filters[@]}" > "$run_dir/$label.ncu.log" 2>&1
  echo "profile $label complete"
done
