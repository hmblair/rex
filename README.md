# rex

Remote execution tool for Python and shell commands on HPC clusters.

## Features

- Run commands on remote hosts via SSH or SLURM
- Detached execution with job management (status, logs, kill, watch)
- SSH connection multiplexing for fast repeated commands
- Project configuration via `.rex.toml`
- File transfer (push, pull, sync) with rsync
- GPU monitoring
- Remote venv building

## Installation

Requires Python 3.10+.

```bash
pip install -e .
```

## Quick Start

```bash
# Run a command on a remote host
rex <host> --exec "python train.py"

# Detach (run in background)
rex <host> -d --exec "python train.py"

# Check job status
rex <host> --jobs
rex <host> --log --last

# File transfer
rex <host> --push ./data ~/data
rex <host> --pull ~/results ./

# Sync project
rex <host> --sync

# Connection management (speeds up subsequent commands)
rex <host> --connect
rex <host> --exec "ls"  # fast
rex <host> --disconnect
```

`<host>` is either an alias from `~/.config/rex/config.toml` or a `user@host` string.

## Configuration

### Global Config (`~/.config/rex/config.toml`)

Define host aliases and per-host defaults:

```toml
[aliases]
sherlock = "user@login.sherlock.stanford.edu"
imp = "user@imp"

[hosts.sherlock]
code_dir = "/home/groups/lab/user"      # base path, project name appended
run_dir = "/scratch/users/user"         # base path, project name appended
modules = ["python/3.12", "cuda/12.4"]
cpu_partition = "normal"
gpu_partition = "gpu"
gres = "gpu:1"
time = "8:00:00"
prefer = "GPU_SKU:H100_SXM5"
slurm = true                    # use SLURM for this host

[hosts.sherlock.env]
MY_VAR = "value"
PATH = "$PATH:/custom/bin"              # supports variable expansion

[hosts.imp]
code_dir = "/home/user"
run_dir = "/tmp/user"
# No SLURM settings (direct SSH)
```

Then use: `rex <host> --exec "python train.py"` where `<host>` is an alias like `sherlock`.

### Project Config (`.rex.toml`)

Place in your project root. Only `name` is required:

```toml
name = "myproject"

# Optional overrides (inherit from host config if not specified)
time = "12:00:00"                       # override host default
modules = ["python/3.12", "special/1.0"] # replaces host modules if specified

[env]
PROJECT_VAR = "value"
PATH = "$PATH:/project/bin"             # variable expansion supported
```

Paths are computed automatically:
- `code_dir` = `/home/groups/lab/user/myproject`
- `run_dir` = `/scratch/users/user/myproject`

Config merge priority: CLI args > project .rex.toml > host config > defaults

### Usage with Project Config

```bash
cd myproject/
rex <host> --sync                           # syncs to code_dir
rex <host> --exec "python train.py"         # uses cpu_partition (default)
rex <host> --gpu --exec "python train.py"   # uses gpu_partition
rex <host> --build                          # builds on cpu_partition
rex <host> --build --gpu                    # builds on gpu_partition
```

## Commands

| Command | Description |
|---------|-------------|
| `rex <host> --exec "cmd"` | Run command (from run_dir) |
| `rex <host> -d --exec "cmd"` | Run detached (background) |
| `rex <host> --exec --code-dir "cmd"` | Run shell command (from code_dir) |
| `rex <host> --exec --login-node "cmd"` | Run on login node (bypass SLURM) |
| `rex <host> --read PATH` | Read file or list directory |
| `rex --jobs` | List jobs on all connected hosts |
| `rex <host> --jobs` | List jobs on specific host |
| `rex <host> --jobs --since 30` | Include finished jobs from last 30 mins |
| `rex <host> --status JOB` | Check job status |
| `rex <host> --log JOB [-f]` | Show job log (follow with -f) |
| `rex <host> --kill JOB` | Kill job |
| `rex <host> --watch JOB [JOB ...]` | Wait for job(s) to complete |
| `rex <host> --info` | Show CPU, memory, and GPU info |
| `rex <host> --push local [remote]` | Upload files |
| `rex <host> --pull remote [local]` | Download files |
| `rex <host> --sync [path]` | Rsync project |
| `rex <host> --build` | Build remote venv (trackable job) |
| `rex <host> --connect` | Open persistent SSH connection |
| `rex <host> --disconnect` | Close persistent connection |
| `rex --connection` | List all active connections |
| `rex <host> --connection` | Show connection status for host |

## Options

| Option | Description |
|--------|-------------|
| `-V, --version` | Show version |
| `-d, --detach` | Run in background |
| `-n, --name NAME` | Job name for detached jobs |
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
| `--since MINS` | Include finished jobs from last N minutes |
| `--json` | JSON output |
| `-f, --follow` | Follow log output |
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
