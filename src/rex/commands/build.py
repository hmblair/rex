"""Build command for remote venv setup."""

from __future__ import annotations

import subprocess
import time

from rex.config.resolved import ResolvedConfig
from rex.exceptions import ConfigError, SlurmError
from rex.output import info, success
from rex.ssh.executor import SSHExecutor


def build(
    ssh: SSHExecutor,
    config: ResolvedConfig,
    wait: bool = False,
    clean: bool = False,
) -> int:
    """Create/update venv on remote.

    Submits build job via sbatch.

    Args:
        config: Resolved config with code_dir, modules, partition, etc.

    Raises:
        ConfigError: If code_dir is not configured.
        SlurmError: If job submission or build fails.
    """
    if not config.execution.code_dir:
        raise ConfigError("code_dir not configured")

    code_dir = config.execution.code_dir

    from rex.utils import generate_job_name
    job = f"build-{generate_job_name()}"
    remote_log = f"{code_dir}/.rex-build.log"
    remote_script = f"{code_dir}/.rex-build.sh"

    # Build module load commands
    module_cmds = ""
    if config.execution.modules:
        module_cmds = f"module load {' '.join(config.execution.modules)}"

    # Partition option (use resolved partition from config)
    partition_opt = ""
    if config.slurm and config.slurm.partition:
        partition_opt = f"--partition={config.slurm.partition}"

    # Clean command
    clean_cmd = ""
    if clean:
        clean_cmd = "rm -rf .venv"

    # Build script
    script_content = f'''#!/bin/bash -l
set -e
echo "=== Rex Build ==="
echo "Started: $(date)"

{module_cmds}

cd {code_dir}
{clean_cmd}

if [[ ! -d .venv ]]; then
    echo "=== Creating venv ==="
    python3 -m venv .venv
fi

echo "=== Installing package ==="
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --only-binary :all: -e .

echo "=== Build complete ==="
echo "Finished: $(date)"
'''

    info(f"Building in {code_dir}")

    # Write script
    subprocess.run(
        ["ssh", *ssh._opts, ssh.target, f"cat > {remote_script} && chmod +x {remote_script}"],
        input=script_content.encode(),
        check=True,
    )

    # Submit job
    code, stdout, _ = ssh.exec(
        f"sbatch --parsable {partition_opt} --time=00:30:00 "
        f"--job-name=rex-{job} --output={remote_log} {remote_script}"
    )

    if not stdout.strip():
        raise SlurmError("Failed to submit build job")

    slurm_id = stdout.strip()
    success(f"Submitted: {job} (SLURM {slurm_id})")
    print(f"Log: rex {ssh.target} --exec 'cat {remote_log}'")

    # Wait if requested
    if wait:
        info("Waiting for build to complete...")
        poll_interval = 5

        while True:
            code, stdout, _ = ssh.exec(f"squeue -j {slurm_id} -h -o %T 2>/dev/null")
            state = stdout.strip()

            if not state:
                # Job done - check result
                code, stdout, _ = ssh.exec(f"tail -1 {remote_log} 2>/dev/null")
                last_line = stdout.strip()

                if "Build complete" in last_line:
                    success("Build complete")
                    return 0
                else:
                    raise SlurmError(
                        f"Build failed - check log: rex {ssh.target} --exec 'cat {remote_log}'"
                    )

            time.sleep(poll_interval)

    return 0
