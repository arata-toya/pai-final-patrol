#!/usr/bin/env bash
set -e
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec ign gazebo -r "$PROJECT_ROOT/worlds/patrol.sdf"
