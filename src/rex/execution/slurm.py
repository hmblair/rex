"""SLURM execution (srun/sbatch)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from rex.execution.base import ExecutionContext, JobInfo, JobResult, JobStatus
from rex.execution.script import SbatchBuilder, ScriptBuilder
from rex.output import success, warn
from rex.ssh.executor import SSHExecutor
from rex.utils import generate_job_name, generate_script_id


@dataclass
class SlurmOptions:
    """SLURM-specific options."""

    partition: str | None = None
    gres: str | None = None
    time: str | None = None
    cpus: int | None = None
    constraint: str | None = None


class SlurmExecutor:
    """SLURM-based execution (srun/sbatch).

    Uses ~/.rex directory for logs/scripts (shared filesystem).
    Uses squeue/scancel for job management.
    """

    def __init__(self, ssh: SSHExecutor, options: SlurmOptions | None = None):
        self.ssh = ssh
        self.options = options or SlurmOptions()

    def _build_slurm_opts(self) -> str:
        """Build srun/sbatch resource options string."""
        opts = ""
        if self.options.partition:
            opts += f" --partition={self.options.partition}"
        if self.options.gres:
            opts += f" --gres={self.options.gres}"
        if self.options.time:
            opts += f" --time={self.options.time}"
        if self.options.cpus:
            opts += f" --cpus-per-task={self.options.cpus}"
        if self.options.constraint:
            opts += f" --constraint={self.options.constraint}"
        return opts

    def run_foreground(
        self, ctx: ExecutionContext, script_path: Path, args: list[str]
    ) -> int:
        """Run Python script via srun with streaming output."""
        script_id = generate_script_id()
        script_dir = ctx.run_dir or "$HOME/.rex"
        remote_py = f"{script_dir}/rex-run-{script_id}.py"
        remote_sh = f"{script_dir}/rex-run-{script_id}.sh"

        # Copy Python script to remote
        with open(script_path) as f:
            script_content = f.read()

        self.ssh.exec(f"mkdir -p {script_dir}")

        import subprocess
        subprocess.run(
            ["ssh", *self.ssh._opts, self.ssh.target, f"cat > {remote_py}"],
            input=script_content.encode(),
            check=True,
        )

        # Build wrapper script
        builder = ScriptBuilder().shebang(login=True)
        if ctx.modules:
            builder.module_load(ctx.modules)
        if ctx.gpus:
            builder.export("CUDA_VISIBLE_DEVICES", ctx.gpus)
        if ctx.code_dir:
            builder.source(f"{ctx.code_dir}/.venv/bin/activate")
        if ctx.run_dir:
            builder.cd(ctx.run_dir)
        builder.run_python(ctx.python, remote_py, args)

        wrapper = builder.build()

        # Write wrapper to remote
        subprocess.run(
            ["ssh", *self.ssh._opts, self.ssh.target, f"cat > {remote_sh} && chmod +x {remote_sh}"],
            input=wrapper.encode(),
            check=True,
        )

        # Execute via srun
        slurm_opts = self._build_slurm_opts()
        exit_code = self.ssh.exec_streaming(
            f"srun{slurm_opts} {remote_sh}; _e=$?; rm -f {remote_py} {remote_sh}; exit $_e",
            tty=None,
        )

        return exit_code

    def run_detached(
        self,
        ctx: ExecutionContext,
        script_path: Path,
        args: list[str],
        job_name: str | None = None,
    ) -> JobInfo:
        """Run Python script via sbatch."""
        if job_name is None:
            job_name = generate_job_name()

        # Get remote home for absolute paths
        code, stdout, _ = self.ssh.exec("echo $HOME")
        remote_home = stdout.strip()
        if not remote_home:
            warn("Failed to get remote home directory")
            return JobInfo(job_id=job_name, log_path="", is_slurm=True, slurm_id=None)
        remote_dir = f"{remote_home}/.rex"
        remote_script = f"{remote_dir}/rex-{job_name}.py"
        remote_sbatch = f"{remote_dir}/rex-{job_name}.sbatch"
        remote_log = f"{remote_dir}/rex-{job_name}.log"

        # Create directory and copy script
        self.ssh.exec("mkdir -p ~/.rex")
        import subprocess
        subprocess.run(
            ["scp", "-q", str(script_path), f"{self.ssh.target}:{remote_script}"],
            check=True,
        )

        # Build sbatch script
        builder = SbatchBuilder().shebang(login=True)
        builder.job_name(f"rex-{job_name}")
        builder.output(remote_log)
        if self.options.partition:
            builder.partition(self.options.partition)
        if self.options.gres:
            builder.gres(self.options.gres)
        if self.options.time:
            builder.time(self.options.time)

        builder.blank_line()
        builder.run_command('echo "[rex] Started: $(date)"')
        builder.run_command('echo "[rex] Host: $(hostname)"')
        builder.run_command(f'echo "[rex] Script: {remote_script}"')
        builder.run_command('echo "---"')

        if ctx.modules:
            builder.module_load(ctx.modules)
        if ctx.gpus:
            builder.export("CUDA_VISIBLE_DEVICES", ctx.gpus)
        if ctx.code_dir:
            builder.source(f"{ctx.code_dir}/.venv/bin/activate")
        if ctx.run_dir:
            builder.run_command(f"mkdir -p {ctx.run_dir} && cd {ctx.run_dir}")

        builder.blank_line()
        args_str = " ".join(f"'{a}'" for a in args) if args else ""
        python_cmd = f"{ctx.python} -u {remote_script}"
        if args_str:
            python_cmd += f" {args_str}"
        builder.run_command(python_cmd)

        builder.blank_line()
        builder.run_command('_rex_exit=$?')
        builder.run_command('echo "---"')
        builder.run_command('echo "[rex] Finished: $(date)"')
        builder.run_command('echo "[rex] Exit code: $_rex_exit"')
        builder.run_command('exit $_rex_exit')

        sbatch_content = builder.build()

        # Write sbatch script
        subprocess.run(
            ["ssh", *self.ssh._opts, self.ssh.target, f"cat > {remote_sbatch}"],
            input=sbatch_content.encode(),
            check=True,
        )

        # Submit job
        code, stdout, stderr = self.ssh.exec(f"sbatch --parsable {remote_sbatch}")
        if code != 0 or not stdout.strip():
            err_msg = stderr.strip() or stdout.strip() or "unknown error"
            warn(f"sbatch failed: {err_msg}")
            return JobInfo(
                job_id=job_name,
                log_path=remote_log,
                is_slurm=True,
                slurm_id=None,
            )

        slurm_id = int(stdout.strip())
        target = self.ssh.target
        success(f"Submitted: {job_name} (SLURM {slurm_id})")
        print(f"Status: rex {target} --status {job_name}")
        print(f"Log:    rex {target} --log {job_name}")
        print(f"Kill:   rex {target} --kill {job_name}")
        return JobInfo(
            job_id=job_name,
            log_path=remote_log,
            is_slurm=True,
            slurm_id=slurm_id,
        )

    def exec_foreground(self, ctx: ExecutionContext, cmd: str) -> int:
        """Execute shell command via srun."""
        script_id = generate_script_id()
        script_dir = ctx.run_dir or "$HOME/.rex"
        remote_script = f"{script_dir}/rex-exec-{script_id}.sh"

        # Build script
        builder = ScriptBuilder().shebang(login=True)
        if ctx.modules:
            builder.module_load(ctx.modules)
        if ctx.gpus:
            builder.export("CUDA_VISIBLE_DEVICES", ctx.gpus)
        if ctx.code_dir:
            builder.source(f"{ctx.code_dir}/.venv/bin/activate")
        if ctx.run_dir:
            builder.cd(ctx.run_dir)
        builder.run_command(cmd)

        script_content = builder.build()

        # Write script to remote
        self.ssh.exec(f"mkdir -p {script_dir}")
        import subprocess
        subprocess.run(
            ["ssh", *self.ssh._opts, self.ssh.target, f"cat > {remote_script} && chmod +x {remote_script}"],
            input=script_content.encode(),
            check=True,
        )

        # Execute via srun
        slurm_opts = self._build_slurm_opts()
        exit_code = self.ssh.exec_streaming(
            f"srun{slurm_opts} {remote_script}; _e=$?; rm -f {remote_script}; exit $_e",
            tty=None,
        )

        return exit_code

    def exec_detached(
        self, ctx: ExecutionContext, cmd: str, job_name: str | None = None
    ) -> JobInfo:
        """Execute shell command via sbatch."""
        if job_name is None:
            job_name = generate_job_name()

        # Get remote home
        code, stdout, _ = self.ssh.exec("echo $HOME")
        remote_home = stdout.strip()
        if not remote_home:
            warn("Failed to get remote home directory")
            return JobInfo(job_id=job_name, log_path="", is_slurm=True, slurm_id=None)
        remote_dir = f"{remote_home}/.rex"
        remote_sbatch = f"{remote_dir}/rex-{job_name}.sbatch"
        remote_log = f"{remote_dir}/rex-{job_name}.log"

        self.ssh.exec("mkdir -p ~/.rex")

        # Build sbatch script
        builder = SbatchBuilder().shebang(login=True)
        builder.job_name(f"rex-{job_name}")
        builder.output(remote_log)
        if self.options.partition:
            builder.partition(self.options.partition)
        if self.options.gres:
            builder.gres(self.options.gres)
        if self.options.time:
            builder.time(self.options.time)

        builder.blank_line()
        builder.run_command('echo "[rex] Started: $(date)"')
        builder.run_command('echo "[rex] Host: $(hostname)"')
        builder.run_command(f'echo "[rex] Command: {cmd}"')
        builder.run_command('echo "---"')

        if ctx.modules:
            builder.module_load(ctx.modules)
        if ctx.gpus:
            builder.export("CUDA_VISIBLE_DEVICES", ctx.gpus)
        if ctx.code_dir:
            builder.source(f"{ctx.code_dir}/.venv/bin/activate")
        if ctx.run_dir:
            builder.cd(ctx.run_dir)

        builder.blank_line()
        builder.run_command(cmd)

        builder.blank_line()
        builder.run_command('_rex_exit=$?')
        builder.run_command('echo "---"')
        builder.run_command('echo "[rex] Finished: $(date)"')
        builder.run_command('echo "[rex] Exit code: $_rex_exit"')
        builder.run_command('exit $_rex_exit')

        sbatch_content = builder.build()

        # Write and submit
        import subprocess
        subprocess.run(
            ["ssh", *self.ssh._opts, self.ssh.target, f"cat > {remote_sbatch}"],
            input=sbatch_content.encode(),
            check=True,
        )

        code, stdout, stderr = self.ssh.exec(f"sbatch --parsable {remote_sbatch}")
        if code != 0 or not stdout.strip():
            err_msg = stderr.strip() or stdout.strip() or "unknown error"
            warn(f"sbatch failed: {err_msg}")
            return JobInfo(
                job_id=job_name,
                log_path=remote_log,
                is_slurm=True,
                slurm_id=None,
            )

        slurm_id = int(stdout.strip())
        target = self.ssh.target
        success(f"Submitted: {job_name} (SLURM {slurm_id})")
        print(f"Log:    rex {target} --log {job_name}")
        print(f"Kill:   rex {target} --kill {job_name}")
        return JobInfo(
            job_id=job_name,
            log_path=remote_log,
            is_slurm=True,
            slurm_id=slurm_id,
        )

    def list_jobs(self, target: str) -> list[JobStatus]:
        """List all rex SLURM jobs."""
        code, stdout, _ = self.ssh.exec(
            "squeue -u $USER -o '%.10i %.20j %.8T %.10M' 2>/dev/null | grep rex"
        )

        jobs = []
        for line in stdout.strip().split("\n"):
            if not line or "rex" not in line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                slurm_id = int(parts[0])
                name = parts[1]
                status = parts[2].lower()
                # Extract job_id from name (rex-<job_id>)
                job_id = name.replace("rex-", "") if name.startswith("rex-") else name
                jobs.append(JobStatus(
                    job_id=job_id,
                    status="running" if status in ("pending", "running") else status,
                    slurm_id=slurm_id,
                ))
        return jobs

    def get_status(self, target: str, job_id: str) -> JobStatus:
        """Get status of specific SLURM job."""
        code, stdout, _ = self.ssh.exec(
            f"squeue -u $USER -n rex-{job_id} -h -o %T 2>/dev/null"
        )

        state = stdout.strip()
        if state:
            return JobStatus(job_id=job_id, status=state.lower())
        return JobStatus(job_id=job_id, status="done")

    def get_log_path(self, target: str, job_id: str) -> str | None:
        """Get log file path."""
        cmd = (
            f'if [ -f ~/.rex/rex-{job_id}.log ]; then echo ~/.rex/rex-{job_id}.log; '
            f'elif [ -f /tmp/rex-{job_id}.log ]; then echo /tmp/rex-{job_id}.log; fi'
        )
        code, stdout, _ = self.ssh.exec(cmd)
        return stdout.strip() if stdout.strip() else None

    def kill_job(self, target: str, job_id: str) -> bool:
        """Cancel SLURM job."""
        code, _, _ = self.ssh.exec(f"scancel -n rex-{job_id}")
        if code == 0:
            success(f"Cancelled job {job_id}")
            return True
        warn(f"Failed to cancel job {job_id}")
        return False

    def watch_job(
        self, target: str, job_id: str, poll_interval: int = 10
    ) -> JobResult:
        """Wait for SLURM job to complete."""
        max_failures = 3
        failures = 0

        while True:
            code, stdout, _ = self.ssh.exec(
                f"squeue -u $USER -n rex-{job_id} -h -o %T 2>/dev/null"
            )

            if code != 0:
                failures += 1
                if failures >= max_failures:
                    warn(f"Lost connection after {max_failures} attempts")
                    return JobResult(job_id=job_id, status="unknown", exit_code=1)
                warn(f"Connection failed (attempt {failures}/{max_failures}), retrying...")
                time.sleep(poll_interval)
                continue

            failures = 0
            state = stdout.strip()

            if not state:
                # Job no longer in queue - check sacct for final status
                code, stdout, _ = self.ssh.exec(
                    f"sacct -n -X --name=rex-{job_id} --format=State 2>/dev/null | head -1 | tr -d ' '"
                )
                sacct_status = stdout.strip()

                if sacct_status == "COMPLETED":
                    success(f"Job {job_id} completed: success")
                    return JobResult(job_id=job_id, status="success", exit_code=0)
                elif sacct_status in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                    warn(f"Job {job_id} finished: {sacct_status.lower()}")
                    return JobResult(job_id=job_id, status="failed", exit_code=1)
                else:
                    success(f"Job {job_id} completed")
                    return JobResult(job_id=job_id, status="done", exit_code=0)

            time.sleep(poll_interval)
