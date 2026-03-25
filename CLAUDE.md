<!-- rex:start -->
# rex — Remote Execution CLI

Use `rex` to run anything requiring a GPU or remote execution. The local machine has no GPU.

Aliases are defined in `~/.config/rex/config.toml` (e.g. `sherlock`, `imp`). Project config lives in `.rex.toml` (only `name` is required; everything else inherits from host config).

## Common Commands

```bash
rex sherlock script.py                        # run on CPU partition
rex sherlock --gpu script.py                  # run on GPU partition
rex sherlock -d --gpu train.py -- --epochs 100  # detached GPU job

# Job management
rex sherlock --jobs                           # list jobs
rex sherlock --status                         # check most recent job
rex sherlock --log -f                         # follow most recent job output
rex sherlock --watch                          # block until most recent job completes
rex sherlock --kill <job-id>                  # stop a job

# File transfer
rex sherlock --sync                           # sync project to remote code_dir
rex sherlock --push ./local ~/remote          # upload
rex sherlock --pull ~/remote ./local          # download

# Remote shell
rex sherlock --exec "nvidia-smi"              # run from run_dir
rex sherlock --exec --code-dir "pytest"       # run from code_dir
rex sherlock --exec --login-node "ls"         # run on login node (bypass SLURM)
rex sherlock --read ~/path                    # read remote file or list directory

# SLURM overrides
rex sherlock --mem 32G --time 4:00:00 --cpus 8 --gres gpu:2 script.py
rex sherlock --constraint a100 script.py      # node constraint
```

## Typical Workflow

```bash
rex sherlock --sync                           # 1. push code
rex sherlock -d --gpu script.py               # 2. launch job
rex sherlock --watch                          # 3. wait
rex sherlock --pull ~/output ./output         # 4. fetch results
```

## Setup & Build

```bash
rex sherlock --sync && rex sherlock --build    # sync code, create venv + pip install -e .
rex sherlock --build --clean                  # rebuild venv from scratch
```

## Notes

- Check `rex sherlock --connection` before running. If no connection, prompt the user to run `rex sherlock --connect`.
- Check GPU availability with `rex sherlock -s --gpu --gpu-info` before submitting GPU jobs.
<!-- rex:end -->
