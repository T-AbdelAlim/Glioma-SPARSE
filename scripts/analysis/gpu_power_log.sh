#!/bin/bash
# Continuous GPU power/utilization sampler (1 Hz) for the paper's energy numbers.
# Usage: gpu_power_log.sh <output_csv>
# Runs until killed. Each row: iso_timestamp,power_w,util_pct,mem_used_mib
OUT="${1:?usage: gpu_power_log.sh <output_csv>}"
echo "timestamp,power_w,util_pct,mem_used_mib" > "$OUT"
while true; do
  line=$(nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)
  ts=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")
  echo "$ts,$line" >> "$OUT"
  sleep 1
done
