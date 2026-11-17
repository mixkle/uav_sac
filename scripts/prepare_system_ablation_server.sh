#!/usr/bin/env bash
set -euo pipefail
uav_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$uav_script_dir/dppo_server_env.sh"
cd "$uav_dppo_root"
# G0 must be fully trained and its selected best frozen before collection.
"$uav_dppo_root/.venv-dppo/bin/python" - "$UAV_DATA_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])/'runs/system_ablation_1m/G0_seed0'
m=json.loads((root/'manifest.json').read_text())
if m['status']!='complete' or m['env_steps']!=1_000_000:raise ValueError('Complete new G0 1M first')
PY
bash scripts/train_system_ablation_server.sh collect --group A3
for uav_student in A3 A5 F2a;do
 bash scripts/train_system_ablation_server.sh pretrain --group "$uav_student"
done
