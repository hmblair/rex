<!-- rex:start -->
# Remote Execution with rex

When you need to run code that requires a GPU (training models, large batch inference, CUDA operations), use `rex` to execute on the remote cluster. The local machine has no GPU.

## When to Use rex

- **Training neural networks** - Any PyTorch/TensorFlow training loop
- **GPU-accelerated inference** - Batch predictions on large datasets
- **CUDA operations** - Anything requiring `torch.cuda` or GPU tensors
- **Memory-intensive operations** - When local RAM is insufficient
- **Remote shell commands** - Any command needing remote execution

## Available Aliases

Defined in `~/.config/rex/config.toml`. Check the file for the full list. Common aliases include:

```
sherlock = hmblair@sherlock (SLURM cluster, default_slurm=true)
imp      = hmblair@imp (direct SSH, no SLURM)
```

Host-specific defaults (code_dir, run_dir, modules, partitions, etc.) are configured per-host in the global config.

Most examples below use `sherlock`. Other aliases work similarly (with or without SLURM depending on host config).

## Project Config (.rex.toml)

Projects only need a `name` field. All other settings inherit from the host config:

```toml
name = "projectname"

# Optional overrides
time = "4:00:00"        # override host default

[env]
PROJECT_VAR = "value"                   # project-specific env vars
PATH = "$PATH:/project/bin"             # variable expansion supported
```

Paths are computed automatically from host config:
- `code_dir` = `/home/groups/rhiju/hmblair/projectname`
- `run_dir` = `/scratch/users/hmblair/projectname`

Config merge priority: CLI args > project .rex.toml > host config > defaults

## Basic Workflow

```bash
# Run a script on CPU partition (default)
rex sherlock script.py

# Run on GPU partition
rex sherlock --gpu script.py

# For long-running jobs, detach to survive disconnection
rex sherlock -d --gpu train.py -- --epochs 100
# Returns: job ID like 20251229-161516

# Monitor detached jobs
rex --jobs                              # list jobs on all connected hosts
rex sherlock --jobs                     # list jobs on specific host
rex sherlock --jobs --since 30          # include finished jobs from last 30 mins
rex sherlock --status 20251229-161516   # check specific job
rex sherlock --status                   # check most recent job
rex sherlock --log 20251229-161516      # view output
rex sherlock --log -f                   # follow most recent job's output
rex sherlock --kill 20251229-161516     # stop specific job

# Wait for job(s) to complete
rex sherlock --watch 20251229-161516    # blocks until done
rex sherlock --watch                    # watch most recent job
rex sherlock --watch job1 job2 job3     # watch multiple jobs
```

## File Transfer & Sync

```bash
rex sherlock --push ./data ~/data                 # upload to remote
rex sherlock --pull ~/checkpoints ./checkpoints   # download from remote
rex sherlock --sync                               # sync project to code_dir
```

## Shell Commands

```bash
rex sherlock --exec "nvidia-smi"          # runs from run_dir
rex sherlock --exec-code-dir "pytest"     # runs from code_dir
rex sherlock --exec-login "ls"            # runs on login node (no SLURM allocation)
rex sherlock -d --exec "wget https://..." # detached
```

## Reading Remote Files

```bash
rex sherlock --read                       # list code_dir contents
rex sherlock --read ~/data                # list directory contents
rex sherlock --read ~/config.yaml         # display file contents
```

Always runs on login node. Automatically detects files vs directories.

## SLURM Options

Override `.rex.toml` settings from command line:

```bash
rex sherlock --mem 32G script.py              # memory allocation
rex sherlock --time 4:00:00 script.py         # time limit
rex sherlock --cpus 8 script.py               # CPUs per task
rex sherlock --gres gpu:2 script.py           # GPU resources
rex sherlock --partition rhiju script.py      # specific partition
rex sherlock --constraint a100 script.py      # node constraint (hard)
rex sherlock --prefer fast script.py          # node preference (soft)
```

## New Project Setup

```bash
cd /path/to/project              # must have .rex.toml and pyproject.toml
rex sherlock --sync              # rsync code to code_dir
rex sherlock --build             # create venv, pip install -e .
rex sherlock --watch --last      # wait for build to complete
rex sherlock -d --gpu script.py  # run with GPU partition
```

## Venv Management

- **Location**: `code_dir/.venv`
- **Creation**: `rex sherlock --build` (returns job ID, trackable with `--log`, `--watch`)
- **Rebuild**: `rex sherlock --build --clean` (deletes venv first)
- **Install mode**: Editable (`-e .`), so `--sync` updates code without reinstall
- **Check logs**: `rex sherlock --log build-xxx` (or `--log --last` right after build)

## Workflow for Long Jobs

1. `rex sherlock --sync` - sync latest code
2. `rex sherlock -d --gpu script.py` - launch detached on GPU
3. `rex sherlock --watch --last` - wait for completion
4. `rex sherlock --pull ~/output ./output` - fetch results

## SSH Connections

Use `rex sherlock --connection` to check for active connections. If none exist, prompt the user to run `rex sherlock --connect` rather than connecting yourself.

## Checking GPU Availability

Before submitting GPU jobs, check partition-specific GPU availability:

```bash
rex sherlock -s --gpu --gpu-info    # check GPUs in GPU partition (from project config)
```
<!-- rex:end -->
