#!/usr/bin/env bash
set -e
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CASE="${1:-CASE_A}"
ACTION="${2:-manual}"
if [[ ! "$CASE" =~ ^CASE_[ABCD]$ || "$ACTION" != manual && "$ACTION" != auto && "$ACTION" != capture ]]; then
  echo 'Usage: bash scripts/start_all.sh CASE_A [manual|auto|capture]'; exit 2
fi
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export IGN_PARTITION=pai_award_v3
export IGN_GAZEBO_RESOURCE_PATH="$ROOT/assets:${IGN_GAZEBO_RESOURCE_PATH:-}"
python3 "$ROOT/scripts/build_worlds.py"
exec python3 -u "$ROOT/scripts/process_supervisor.py" "$CASE" cnn "$ACTION"
