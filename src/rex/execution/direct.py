"""Direct SSH execution (non-SLURM)."""

from __future__ import annotations

import time
from pathlib import Path

from rex.execution.base import ExecutionContext, JobInfo, JobResult, JobStatus
from rex.execution.script import ScriptBuilder
from rex.output import success, warn
from rex.ssh.executor import SSHExecutor
from rex.utils import generate_job_name, generate_script_id, job_pattern


class DirectExecutor:
    """Direct SSH execution (non-SLURM).

    Uses nohup for detached jobs, pgrep/kill for job management.
    Logs/scripts stored in /tmp on the remote.
    """

    def __init__(self, ssh: SSHExecutor):
        self.ssh = ssh

    def run_foreground(
        self, ctx: ExecutionContext, script_path: Path, args: list[str]
    ) -> int:
        """Run Python script in foreground with streaming output."""
        script_id = generate_script_id()
        remote_py = f"/tmp/rex-run-{script_id}.py"
        remote_sh = f"/tmp/rex-run-{script_id}.sh"

        # Copy Python script to remote
        with open(script_path) as f:
            script_content = f.read()

        code, _, _ = self.ssh.exec(f"cat > {remote_py}")
        # Actually send the content via stdin
        import subprocess

        subprocess.run(
            ["ssh", *self.ssh._opts, self.ssh.target, f"cat > {remote_py}"],
            input=script_content.encode(),
            check=True,
        )

        # Build wrapper script
        builder = ScriptBuilder().shebang()
        if ctx.gpus:
            builder.export("CUDA_VISIBLE_DEVICES", ctx.gpus)
        if ctx.run_dir:
            builder.cd(ctx.run_dir)
        builder.run_python(ctx.python, remote_py, args)

        wrapper = builder.build()

        # Execute via script method
        exit_code = self.ssh.exec_script_streaming(wrapper, tty=True)

        # Cleanup
        self.ssh.exec(f"rm -f {remote_py} {remote_sh}")

        return exit_code

    def run_detached(
        self,
        ctx: ExecutionContext,
        script_path: Path,
        args: list[str],
        job_name: str | None = None,
    ) -> JobInfo:
        """Run Python script detached in background."""
        if job_name is None:
            job_name = generate_job_name()

        remote_script = f"/tmp/rex-{job_name}.py"
        remote_log = f"/tmp/rex-{job_name}.log"

        # Copy script via scp
        import subprocess

        subprocess.run(
            ["scp", "-q", str(script_path), f"{self.ssh.target}:{remote_script}"],
            check=True,
        )

        # Build command
        env_prefix = ""
        if ctx.gpus:
            env_prefix += f"export CUDA_VISIBLE_DEVICES={ctx.gpus}; "
        if ctx.run_dir:
            env_prefix += f"cd {ctx.run_dir}; "

        args_str = " ".join(f"'{a}'" for a in args) if args else ""
        python_cmd = f"{ctx.python} -u {remote_script}"
        if args_str:
            python_cmd += f" {args_str}"

        # Wrapper with exit code reporting
        wrapper = (
            f"{{ {env_prefix}{python_cmd}; code=$?; "
            f'if [ $code -gt 128 ]; then echo "[rex] Killed by signal $((code-128))" >&2; '
            f'elif [ $code -ne 0 ]; then echo "[rex] Exit code: $code" >&2; fi; }}'
        )

        # Run detached with nohup
        cmd = (
            f"nohup bash -c '{wrapper}' > {remote_log} 2>&1 & "
            f"pid=$!; disown $pid 2>/dev/null; sleep 0.5; echo $pid"
        )

        code, stdout, _ = self.ssh.exec(cmd)
        pid = int(stdout.strip()) if stdout.strip() else None

        success(f"Detached: {job_name} (PID {pid})")
        return JobInfo(
            job_id=job_name,
            log_path=remote_log,
            is_slurm=False,
            pid=pid,
        )

    def exec_foreground(self, ctx: ExecutionContext, cmd: str) -> int:
        """Execute shell command in foreground."""
        # Build prefix
        prefix = ""
        if ctx.modules:
            prefix += f"module load {' '.join(ctx.modules)} && "
        if ctx.gpus:
            prefix += f"export CUDA_VISIBLE_DEVICES={ctx.gpus} && "
        if ctx.code_dir:
            prefix += f"source {ctx.code_dir}/.venv/bin/activate && "
        if ctx.run_dir:
            prefix += f"cd {ctx.run_dir} && "

        full_cmd = prefix + cmd

        # Build script for login shell execution
        script = ScriptBuilder().shebang(login=True).run_command(full_cmd).build()

        return self.ssh.exec_script_streaming(script, login_shell=True, tty=True)

    def exec_detached(
        self, ctx: ExecutionContext, cmd: str, job_name: str | None = None
    ) -> JobInfo:
        """Execute shell command detached."""
        if job_name is None:
            job_name = generate_job_name()

        remote_log = f"/tmp/rex-{job_name}.log"

        # Build prefix
        prefix = ""
        if ctx.modules:
            prefix += f"module load {' '.join(ctx.modules)} && "
        if ctx.gpus:
            prefix += f"export CUDA_VISIBLE_DEVICES={ctx.gpus} && "
        if ctx.code_dir:
            prefix += f"source {ctx.code_dir}/.venv/bin/activate && "
        if ctx.run_dir:
            prefix += f"cd {ctx.run_dir} && "

        full_cmd = prefix + cmd

        # Escape single quotes
        escaped_cmd = full_cmd.replace("'", "'\\''")

        # Run detached with login shell
        ssh_cmd = (
            f"nohup bash -l -c '{escaped_cmd}' > {remote_log} 2>&1 & "
            f"pid=$!; disown $pid 2>/dev/null; sleep 0.5; echo $pid"
        )

        code, stdout, _ = self.ssh.exec(ssh_cmd)
        pid = int(stdout.strip()) if stdout.strip() else None

        success(f"Detached: {job_name} (PID {pid})")
        return JobInfo(
            job_id=job_name,
            log_path=remote_log,
            is_slurm=False,
            pid=pid,
        )

    def list_jobs(self, target: str) -> list[JobStatus]:
        """List all rex jobs on remote."""
        script = '''
for log in /tmp/rex-*.log ~/.rex/rex-*.log; do
    [ -f "$log" ] || continue
    job=$(basename "$log" .log | sed "s/rex-//")
    pattern="rex-${job}[.]py"
    pid=$(pgrep -f "$pattern" 2>/dev/null | head -1)
    if [ -n "$pid" ]; then
        status="running"
    else
        status="done"
        pid="-"
    fi
    # Get first line of script as description
    desc=""
    script_tmp="/tmp/rex-${job}.py"
    if [ -f "$script_tmp" ]; then
        desc=$(head -1 "$script_tmp" | sed "s/^#[[:space:]]*//" | cut -c1-40)
    fi
    printf "%s\\t%s\\t%s\\t%s\\n" "$job" "$status" "$pid" "$desc"
done
'''
        code, stdout, _ = self.ssh.exec(f"bash -c {_shell_quote(script)}")

        jobs = []
        for line in stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                job_id, status, pid_str = parts[0], parts[1], parts[2]
                desc = parts[3] if len(parts) > 3 else None
                pid = int(pid_str) if pid_str != "-" else None
                jobs.append(JobStatus(
                    job_id=job_id,
                    status=status,
                    pid=pid,
                    description=desc,
                ))
        return jobs

    def get_status(self, target: str, job_id: str) -> JobStatus:
        """Get status of specific job."""
        pattern = job_pattern(job_id)
        code, stdout, _ = self.ssh.exec(f"pgrep -f '{pattern}' | head -1")

        if code != 0:
            return JobStatus(job_id=job_id, status="unknown")

        pid = int(stdout.strip()) if stdout.strip() else None
        status = "running" if pid else "done"
        return JobStatus(job_id=job_id, status=status, pid=pid)

    def get_log_path(self, target: str, job_id: str) -> str | None:
        """Get log file path for a job."""
        cmd = (
            f'if [ -f ~/.rex/rex-{job_id}.log ]; then echo ~/.rex/rex-{job_id}.log; '
            f'elif [ -f /tmp/rex-{job_id}.log ]; then echo /tmp/rex-{job_id}.log; fi'
        )
        code, stdout, _ = self.ssh.exec(cmd)
        return stdout.strip() if stdout.strip() else None

    def kill_job(self, target: str, job_id: str) -> bool:
        """Kill a running job."""
        pattern = job_pattern(job_id)
        cmd = (
            f"pid=$(pgrep -f '{pattern}' | head -1); "
            f'if [ -n "$pid" ]; then kill "$pid" 2>/dev/null && echo killed || echo failed; '
            f'else echo not_running; fi'
        )
        code, stdout, _ = self.ssh.exec(cmd)
        result = stdout.strip()

        if result == "killed":
            success(f"Killed job {job_id}")
            return True
        elif result == "not_running":
            warn(f"Job {job_id} is not running")
            return False
        else:
            warn(f"Failed to kill job {job_id}")
            return False

    def watch_job(
        self, target: str, job_id: str, poll_interval: int = 5
    ) -> JobResult:
        """Wait for job to complete."""
        max_failures = 3
        failures = 0

        while True:
            status = self.get_status(target, job_id)

            if status.status == "unknown":
                failures += 1
                if failures >= max_failures:
                    warn(f"Lost connection after {max_failures} attempts")
                    return JobResult(job_id=job_id, status="unknown", exit_code=1)
                warn(f"Connection failed (attempt {failures}/{max_failures}), retrying...")
                time.sleep(poll_interval)
                continue

            failures = 0

            if status.status == "done":
                success(f"Job {job_id} completed")
                return JobResult(job_id=job_id, status="done", exit_code=0)

            time.sleep(poll_interval)


def _shell_quote(s: str) -> str:
    """Quote string for shell."""
    escaped = s.replace("'", "'\\''")
    return f"'{escaped}'"
