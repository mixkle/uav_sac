#!/usr/bin/env bash
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$uav_script_dir/launch_motion_only_server.sh" --resume-10m "$@"
