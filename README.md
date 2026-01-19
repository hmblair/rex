# rex

Remote execution tool for Python and shell commands on HPC clusters.

## Features

- Run Python scripts on remote hosts via SSH or SLURM
- Detached execution with job management (status, logs, kill, watch)
- SSH connection multiplexing for fast repeated commands
- Project configuration via `.rex.toml`
- File transfer (push, pull, sync) with rsync
- GPU monitoring
- Remote venv building

## Installation

Requires Python 3.9+.

```bash
pip install -e .
```

## Quick Start

```bash
# Run a script on a remote host
rex user@host train.py

# Run via SLURM
rex user@host -s train.py

# Detach (run in background)
rex user@host -d train.py

# Check job status
rex user@host --jobs
rex user@host --log --last

# Shell commands
rex user@host --exec "nvidia-smi"

# File transfer
rex user@host --push ./data ~/data
rex user@host --pull ~/results ./

# Sync project
rex user@host --sync

# Connection management (speeds up subsequent commands)
rex user@host --connect
rex user@host --exec "ls"  # fast
rex user@host --disconnect
```

## Configuration

### Aliases (`~/.config/rex`)

Define shortcuts for frequently used hosts:

```
gpu = user@gpu.server.com -p /opt/python3.12
cluster = user@login.cluster.edu --slurm --partition gpu --gres gpu:1
```

Then use: `rex gpu train.py`

### Project Config (`.rex.toml`)

Place in your project root:

```toml
host = "user@cluster.edu"
code_dir = "/home/user/projects/myproject"
run_dir = "/scratch/user/myproject"
modules = ["python/3.12", "cuda/12.0"]

# SLURM partitions
cpu_partition = "normal"
gpu_partition = "gpu"

# SLURM resources
gres = "gpu:1"
time = "2:00:00"
cpus = 4
mem = "16G"
constraint = "GPU_SKU:A100_SXM4"
prefer = "GPU_SKU:H100_SXM5"

# Environment variables
[env]
CUDA_VISIBLE_DEVICES = "0"
MY_VAR = "value"

# Optional: default to GPU partition (default: false)
default_gpu = false
```

With a project config, you can run commands without specifying the host:

```bash
cd myproject/
rex --sync              # syncs to code_dir
rex -s train.py         # uses cpu_partition (default)
rex -s --gpu train.py   # uses gpu_partition
rex --build             # builds on cpu_partition
rex --build --gpu       # builds on gpu_partition
```

## Commands

| Command | Description |
|---------|-------------|
| `rex host script.py [args]` | Run Python script |
| `rex host -s script.py` | Run via SLURM (srun) |
| `rex host -d script.py` | Run detached (background) |
| `rex host -s -d script.py` | Submit via sbatch |
| `rex host --exec "cmd"` | Run shell command |
| `rex host --exec-login "cmd"` | Run command on login node |
| `rex host --read PATH` | Read file or list directory |
| `rex host --jobs` | List all jobs |
| `rex host --status JOB` | Check job status |
| `rex host --log JOB [-f]` | Show job log (follow with -f) |
| `rex host --kill JOB` | Kill job |
| `rex host --watch JOB` | Wait for job completion |
| `rex host --gpu-info` | Show GPU utilization |
| `rex host --push local [remote]` | Upload files |
| `rex host --pull remote [local]` | Download files |
| `rex host --sync [path]` | Rsync project |
| `rex host --build [--wait]` | Build remote venv |
| `rex host --connect` | Open persistent SSH connection |
| `rex host --disconnect` | Close persistent connection |
| `rex host --connection` | Show connection status |

## Options

| Option | Description |
|--------|-------------|
| `-V, --version` | Show version |
| `-d, --detach` | Run in background |
| `-s, --slurm` | Use SLURM scheduler |
| `-n, --name NAME` | Job name for detached jobs |
| `-p, --python PATH` | Python interpreter path |
| `-m, --module MOD` | Load module (repeatable) |
| `--gpu` | Use GPU partition |
| `--cpu` | Use CPU partition (override) |
| `--partition NAME` | SLURM partition (overrides --gpu/--cpu) |
| `--gres SPEC` | SLURM GPU resources |
| `--time LIMIT` | SLURM time limit |
| `--cpus N` | SLURM CPUs per task |
| `--mem SIZE` | SLURM memory (e.g., 4G, 16000M) |
| `--constraint NAME` | SLURM node constraint |
| `--prefer NAME` | SLURM node preference (soft) |
| `--last` | Use most recent job |
| `--json` | JSON output |
| `-f, --follow` | Follow log output |
| `--wait` | Wait for build to complete |
| `--clean` | Clean venv before build |
| `--debug` | Enable verbose SSH output |

## Shell Completions

Zsh completions are provided in `completions/_rex`. To install:

```bash
# Copy to your zsh completions directory
cp completions/_rex ~/.zsh/completions/

# Or symlink
ln -s $(pwd)/completions/_rex ~/.zsh/completions/_rex

# Then add to your .zshrc (if not already):
fpath=(~/.zsh/completions $fpath)
autoload -Uz compinit && compinit
```

## License

MIT
