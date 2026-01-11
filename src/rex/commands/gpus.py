"""GPU monitoring command."""

import json

from rex.ssh.executor import SSHExecutor

# ANSI colors
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
NC = "\033[0m"


def show_gpus(ssh: SSHExecutor, target: str, json_output: bool = False) -> int:
    """Show GPU availability and utilization."""
    # Get GPU info
    code, gpu_info, _ = ssh.exec(
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits 2>/dev/null"
    )

    if not gpu_info.strip():
        if json_output:
            print("[]")
        else:
            from rex.output import warn
            warn("No GPU information available (nvidia-smi failed or no GPUs)")
        return 0

    # Get process info
    proc_script = '''
declare -A uuid_to_idx
while IFS=, read -r idx uuid; do
    idx=$(echo "$idx" | tr -d " ")
    uuid=$(echo "$uuid" | tr -d " ")
    uuid_to_idx[$uuid]=$idx
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)

nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null | while read line; do
    uuid=$(echo "$line" | cut -d, -f1 | tr -d " ")
    pid=$(echo "$line" | cut -d, -f2 | tr -d " ")
    mem=$(echo "$line" | cut -d, -f3 | tr -d " ")
    gpu_idx=${uuid_to_idx[$uuid]}
    user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d " ")
    [ -n "$user" ] && [ -n "$gpu_idx" ] && echo "$gpu_idx,$pid,$mem,$user"
done
'''
    code, proc_info, _ = ssh.exec(f"bash -c '{proc_script}'")

    # Parse process info into dict by GPU index
    procs_by_gpu: dict[int, list[dict]] = {}
    for line in proc_info.strip().split("\n"):
        if not line:
            continue
        parts = line.split(",")
        if len(parts) >= 4:
            gpu_idx = int(parts[0])
            if gpu_idx not in procs_by_gpu:
                procs_by_gpu[gpu_idx] = []
            procs_by_gpu[gpu_idx].append({
                "user": parts[3],
                "pid": int(parts[1]),
                "memory": int(parts[2]),
            })

    # Parse GPU info
    gpus = []
    for line in gpu_info.strip().split("\n"):
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue

        idx = int(parts[0])
        name = parts[1]
        mem_used = int(parts[2])
        mem_total = int(parts[3]) or 1  # Avoid division by zero
        util = int(parts[4]) if parts[4].isdigit() else 0

        mem_pct = mem_used * 100 // mem_total

        # Determine status
        if util < 10 and mem_pct < 10:
            status = "free"
        elif util < 50:
            status = "partial"
        else:
            status = "busy"

        gpu = {
            "index": idx,
            "name": name,
            "memory_used": mem_used,
            "memory_total": mem_total,
            "memory_percent": mem_pct,
            "utilization": util,
            "status": status,
            "processes": procs_by_gpu.get(idx, []),
        }
        gpus.append(gpu)

    if json_output:
        print(json.dumps(gpus, indent=2))
    else:
        for gpu in gpus:
            # Color status
            if gpu["status"] == "free":
                status_colored = f"{GREEN}free{NC}"
            elif gpu["status"] == "partial":
                status_colored = f"{YELLOW}partial{NC}"
            else:
                status_colored = f"{RED}busy{NC}"

            print(
                f"GPU {gpu['index']}: {gpu['name']:<20} "
                f"{gpu['memory_used']:>5}MB/{gpu['memory_total']:>5}MB ({gpu['memory_percent']:>2}%)  "
                f"util: {gpu['utilization']:>3}%  {status_colored}"
            )

            for proc in gpu["processes"]:
                print(f"       └─ {proc['user']:<12} PID {proc['pid']:<8} {proc['memory']:>5}MB")

    return 0


def show_slurm_gpus(ssh: SSHExecutor, partition: str | None = None) -> int:
    """Show SLURM GPU availability."""
    partition_opt = f"-p {partition}" if partition else ""
    code, stdout, _ = ssh.exec(
        f"sinfo {partition_opt} -N -o '%N %G' --noheader 2>/dev/null | grep -v '(null)' | sort -u"
    )

    if not stdout.strip():
        print("No GPUs in partition")
    else:
        print(stdout.strip())

    return 0
