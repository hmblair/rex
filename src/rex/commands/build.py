"""Build command for remote venv setup."""

from __future__ import annotations

import subprocess
import time

from rex.config.project import ProjectConfig
from rex.exceptions import ConfigError, SlurmError
from rex.output import info, success
from rex.ssh.executor import SSHExecutor


def build(
    ssh: SSHExecutor,
    project: ProjectConfig,
    wait: bool = False,
    clean: bool = False,
    use_gpu: bool = False,
) -> int:
    """Create/update venv on remote.

    Requires .rex.toml with code_dir.
    Submits build job via sbatch.
    use_gpu: If True, use gpu_partition; otherwise use cpu_partition.

    Raises:
        ConfigError: If code_dir is not configured.
        SlurmError: If job submission or build fails.
    """
    if not project.code_dir:
        raise ConfigError("No .rex.toml found with code_dir")

    from rex.utils import generate_job_name
    job = f"build-{generate_job_name()}"
    remote_log = f"{project.code_dir}/.rex-build.log"
    remote_script = f"{project.code_dir}/.rex-build.sh"

    # Build module load commands
    module_cmds = ""
    if project.modules:
        module_cmds = f"module load {' '.join(project.modules)}"

    # Partition option
    partition_opt = ""
    partition = project.gpu_partition if use_gpu else project.cpu_partition
    if partition:
        partition_opt = f"--partition={partition}"

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

cd {project.code_dir}
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

    info(f"Building in {project.code_dir}")

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
