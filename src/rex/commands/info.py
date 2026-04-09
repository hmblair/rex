"""Remote host info command (CPU, memory, GPUs).

Architecture:
  - fetch_*  : run SSH commands, return raw stdout strings
  - parse_*  : parse raw strings into dataclasses
  - display_*: format and print dataclasses to stdout
  - show_*   : top-level entry points that wire fetch -> parse -> display
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from rex.ssh.executor import SSHExecutor

# ANSI colors
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
NC = "\033[0m"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class GpuProcess:
    user: str
    pid: int
    memory_mb: int


@dataclass
class GpuInfo:
    index: int
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization: int
    processes: list[GpuProcess] = field(default_factory=list)

    @property
    def memory_percent(self) -> int:
        return self.memory_used_mb * 100 // self.memory_total_mb if self.memory_total_mb else 0

    @property
    def status(self) -> str:
        if self.utilization < 10 and self.memory_percent < 10:
            return "free"
        if self.utilization < 50:
            return "partial"
        return "busy"


@dataclass
class CpuMemInfo:
    cpus: int
    memory_total_mb: int
    memory_used_mb: int
    memory_available_mb: int

    @property
    def memory_percent(self) -> int:
        return self.memory_used_mb * 100 // self.memory_total_mb if self.memory_total_mb else 0


@dataclass
class SlurmNodeInfo:
    hostname: str
    cpus: int
    memory_total_mb: int
    gpu_total: int
    gpu_used: int
    state: str
    gpu_type: str = ""

    @property
    def gpu_free(self) -> int:
        return self.gpu_total - self.gpu_used


@dataclass
class SlurmPartitionInfo:
    def_mem_per_cpu_mb: int | None = None
    max_mem_per_cpu_mb: int | None = None
    max_mem_per_node_mb: int | None = None
    nodes: list[SlurmNodeInfo] = field(default_factory=list)

    @property
    def total_gpus(self) -> int:
        return sum(n.gpu_total for n in self.nodes)

    @property
    def used_gpus(self) -> int:
        return sum(n.gpu_used for n in self.nodes)

    @property
    def free_gpus(self) -> int:
        return self.total_gpus - self.used_gpus


# ---------------------------------------------------------------------------
# Fetch: run SSH commands, return raw stdout
# ---------------------------------------------------------------------------


def fetch_cpu_mem(ssh: SSHExecutor) -> str:
    """Return raw output of nproc + free."""
    _, out, _ = ssh.exec(
        "nproc && free -m | awk '/^Mem:/ {print $2, $3, $7}'"
    )
    return out


def fetch_gpu_info(ssh: SSHExecutor) -> str:
    """Return raw nvidia-smi CSV output, or empty string if no GPUs."""
    _, out, _ = ssh.exec(
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits 2>/dev/null"
    )
    return out


def fetch_gpu_processes(ssh: SSHExecutor) -> str:
    """Return per-GPU process info as 'gpu_idx,pid,mem,user' lines."""
    script = '''
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
    _, out, _ = ssh.exec(f"bash -c '{script}'")
    return out


def fetch_slurm_partition(ssh: SSHExecutor, partition: str | None) -> str:
    """Return raw scontrol output for a partition."""
    arg = partition or ""
    _, out, _ = ssh.exec(f"scontrol show partition {arg} 2>/dev/null")
    return out


def fetch_slurm_nodes(ssh: SSHExecutor, partition: str | None) -> str:
    """Return per-node info (hostname, gres, state, features)."""
    opt = f"-p {partition}" if partition else ""
    _, out, _ = ssh.exec(
        f"sinfo {opt} -N -O 'NodeHost:30,Gres:20,StateLong:15,CPUs:10,Memory:15,Features:200' "
        f"--noheader 2>/dev/null | sort -u"
    )
    return out


def fetch_slurm_jobs(ssh: SSHExecutor, partition: str | None) -> str:
    """Return running job node + GPU allocations."""
    opt = f"-p {partition}" if partition else ""
    _, out, _ = ssh.exec(
        f"squeue {opt} -t running -O 'NodeList:30,tres-alloc:80' --noheader 2>/dev/null"
    )
    return out


# ---------------------------------------------------------------------------
# Parse: raw strings -> dataclasses
# ---------------------------------------------------------------------------


def parse_cpu_mem(raw: str) -> CpuMemInfo:
    """Parse nproc + free output."""
    lines = raw.strip().split("\n")
    cpus = int(lines[0])
    mem_parts = lines[1].split()
    return CpuMemInfo(
        cpus=cpus,
        memory_total_mb=int(mem_parts[0]),
        memory_used_mb=int(mem_parts[1]),
        memory_available_mb=int(mem_parts[2]),
    )


def parse_gpu_processes(raw: str) -> dict[int, list[GpuProcess]]:
    """Parse 'gpu_idx,pid,mem,user' lines into a dict keyed by GPU index."""
    procs: dict[int, list[GpuProcess]] = {}
    for line in raw.strip().split("\n"):
        if not line:
            continue
        parts = line.split(",")
        if len(parts) >= 4:
            idx = int(parts[0])
            procs.setdefault(idx, []).append(
                GpuProcess(user=parts[3], pid=int(parts[1]), memory_mb=int(parts[2]))
            )
    return procs


def parse_gpu_info(raw: str, procs: dict[int, list[GpuProcess]]) -> list[GpuInfo]:
    """Parse nvidia-smi CSV into GpuInfo list."""
    gpus: list[GpuInfo] = []
    for line in raw.strip().split("\n"):
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx = int(parts[0])
        gpus.append(GpuInfo(
            index=idx,
            name=parts[1],
            memory_used_mb=int(parts[2]),
            memory_total_mb=int(parts[3]) or 1,
            utilization=int(parts[4]) if parts[4].isdigit() else 0,
            processes=procs.get(idx, []),
        ))
    return gpus


def parse_slurm_partition(raw: str) -> SlurmPartitionInfo:
    """Parse scontrol show partition output into memory limits."""
    info = SlurmPartitionInfo()
    for key, attr in [
        ("DefMemPerCPU", "def_mem_per_cpu_mb"),
        ("MaxMemPerCPU", "max_mem_per_cpu_mb"),
        ("MaxMemPerNode", "max_mem_per_node_mb"),
    ]:
        m = re.search(rf"{key}=(\d+)", raw)
        if m:
            setattr(info, attr, int(m.group(1)))
    return info


def parse_slurm_nodes(raw: str) -> list[SlurmNodeInfo]:
    """Parse sinfo node output into SlurmNodeInfo list."""
    nodes: list[SlurmNodeInfo] = []
    for line in raw.strip().split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue

        hostname = parts[0]
        gres = parts[1]
        state = parts[2]
        cpus = int(parts[3])
        memory = int(parts[4])
        features = parts[5] if len(parts) > 5 else ""

        gpu_total = 0
        if "gpu:" in gres:
            try:
                num_str = gres.split("gpu:")[1].split("(")[0]
                gpu_total = int(num_str)
            except (IndexError, ValueError):
                pass

        gpu_type = ""
        if "GPU_SKU:" in features:
            try:
                sku = features.split("GPU_SKU:")[1].split(",")[0]
                gpu_type = sku.split("_")[0]
            except IndexError:
                pass

        if True:  # include CPU-only nodes
            nodes.append(SlurmNodeInfo(
                hostname=hostname,
                cpus=cpus,
                memory_total_mb=memory,
                gpu_total=gpu_total,
                gpu_used=0,
                state=state,
                gpu_type=gpu_type,
            ))
    return nodes


def parse_slurm_jobs(raw: str, nodes: list[SlurmNodeInfo]) -> None:
    """Update gpu_used on nodes in-place from squeue output."""
    node_map = {n.hostname: n for n in nodes}
    for line in raw.strip().split("\n"):
        if not line or "gpu" not in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        job_node = parts[0]
        tres = parts[1]
        if "gres/gpu=" in tres:
            try:
                count = int(tres.split("gres/gpu=")[1].split(",")[0])
                if job_node in node_map:
                    node_map[job_node].gpu_used += count
            except (ValueError, IndexError):
                pass


# ---------------------------------------------------------------------------
# Display: dataclasses -> formatted terminal output
# ---------------------------------------------------------------------------


def _format_mb(mb: int) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f}GB"
    return f"{mb}MB"


def display_cpu_mem(info: CpuMemInfo) -> None:
    print(f"{BOLD}CPU:{NC}    {info.cpus} cores")
    print(
        f"{BOLD}Memory:{NC} {_format_mb(info.memory_used_mb)} / {_format_mb(info.memory_total_mb)} "
        f"({info.memory_percent}% used, {_format_mb(info.memory_available_mb)} available)"
    )


def display_gpus(gpus: list[GpuInfo]) -> None:
    if not gpus:
        return
    print()
    for gpu in gpus:
        colors = {"free": GREEN, "partial": YELLOW, "busy": RED}
        status_colored = f"{colors[gpu.status]}{gpu.status}{NC}"
        print(
            f"GPU {gpu.index}: {gpu.name:<20} "
            f"{gpu.memory_used_mb:>5}MB/{gpu.memory_total_mb:>5}MB ({gpu.memory_percent:>2}%)  "
            f"util: {gpu.utilization:>3}%  {status_colored}"
        )
        for proc in gpu.processes:
            print(f"       └─ {proc.user:<12} PID {proc.pid:<8} {proc.memory_mb:>5}MB")


def display_slurm_partition(info: SlurmPartitionInfo) -> None:
    if info.nodes:
        node = info.nodes[0]
        print(f"{BOLD}CPUs per node:{NC}    {node.cpus}")
        print(f"{BOLD}Memory per node:{NC}  {_format_mb(node.memory_total_mb)}")
    if info.def_mem_per_cpu_mb is not None:
        print(f"{BOLD}Default mem/CPU:{NC} {_format_mb(info.def_mem_per_cpu_mb)}")
    if info.max_mem_per_cpu_mb is not None:
        print(f"{BOLD}Max mem/CPU:{NC}     {_format_mb(info.max_mem_per_cpu_mb)}")


def display_slurm_gpus(info: SlurmPartitionInfo) -> None:
    gpu_nodes = [n for n in info.nodes if n.gpu_total > 0]
    if not gpu_nodes:
        return

    print()
    for node in sorted(gpu_nodes, key=lambda n: n.hostname):
        free = node.gpu_free
        status = f"{GREEN}{free} free{NC}" if free > 0 else f"{RED}0 free{NC}"
        gpu_label = f" {node.gpu_type}" if node.gpu_type else ""
        print(f"{node.hostname}:{gpu_label} {node.gpu_used}/{node.gpu_total} used  {status}  ({node.state})")

    total = sum(n.gpu_total for n in gpu_nodes)
    used = sum(n.gpu_used for n in gpu_nodes)
    print(f"\nTotal: {used}/{total} GPUs used, {total - used} free")


# ---------------------------------------------------------------------------
# Entry points: fetch -> parse -> display
# ---------------------------------------------------------------------------


def show_info(ssh: SSHExecutor, target: str, json_output: bool = False) -> int:
    """Show CPU, memory, and GPU info for a non-SLURM host."""
    cpu_mem = parse_cpu_mem(fetch_cpu_mem(ssh))

    gpu_raw = fetch_gpu_info(ssh)
    gpus: list[GpuInfo] = []
    if gpu_raw.strip():
        procs = parse_gpu_processes(fetch_gpu_processes(ssh))
        gpus = parse_gpu_info(gpu_raw, procs)

    if json_output:
        data: dict[str, Any] = {
            "cpus": cpu_mem.cpus,
            "memory_total_mb": cpu_mem.memory_total_mb,
            "memory_used_mb": cpu_mem.memory_used_mb,
            "memory_available_mb": cpu_mem.memory_available_mb,
            "gpus": [
                {
                    "index": g.index, "name": g.name,
                    "memory_used": g.memory_used_mb, "memory_total": g.memory_total_mb,
                    "memory_percent": g.memory_percent, "utilization": g.utilization,
                    "status": g.status,
                    "processes": [
                        {"user": p.user, "pid": p.pid, "memory": p.memory_mb}
                        for p in g.processes
                    ],
                }
                for g in gpus
            ],
        }
        print(json.dumps(data, indent=2))
        return 0

    display_cpu_mem(cpu_mem)
    display_gpus(gpus)
    return 0


def show_slurm_info(ssh: SSHExecutor, partition: str | None = None) -> int:
    """Show SLURM partition info: CPU/memory limits and GPU availability."""
    part_info = parse_slurm_partition(fetch_slurm_partition(ssh, partition))
    nodes = parse_slurm_nodes(fetch_slurm_nodes(ssh, partition))
    parse_slurm_jobs(fetch_slurm_jobs(ssh, partition), nodes)
    part_info.nodes = nodes

    display_slurm_partition(part_info)
    display_slurm_gpus(part_info)
    return 0
