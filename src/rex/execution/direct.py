"""Direct SSH execution (non-SLURM)."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from rex.execution.base import ExecutionContext, JobInfo, JobResult, JobStatus
from rex.execution.script import ScriptBuilder, build_context_commands, get_log_path as _get_log_path
from rex.output import success, warn
from rex.ssh.executor import SSHExecutor
from rex.utils import generate_script_id, job_pattern, shell_quote


def _run_detached_nohup(
    ssh: SSHExecutor,
    bash_cmd: str,
    log_path: str,
    job_name: str,
    *,
    login_shell: bool = False,
    show_status: bool = False,
) -> JobInfo:
    """Run command detached via nohup and return JobInfo.

    Args:
        ssh: SSH executor.
        bash_cmd: Command to run (will be passed to bash -c).
        log_path: Remote path for output log.
        job_name: Job identifier.
        login_shell: Use bash -l (login shell).
        show_status: Print status command hint.
    """
    shell_flag = "-l -c" if login_shell else "-c"
    nohup_cmd = (
        f"nohup bash {shell_flag} '{bash_cmd}' > {log_path} 2>&1 & "
        f"pid=$!; disown $pid 2>/dev/null; sleep 0.5; echo $pid"
    )

    _, stdout, _ = ssh.exec(nohup_cmd)
    pid = int(stdout.strip()) if stdout.strip() else None

    target = ssh.target
    success(f"Detached: {job_name} (PID {pid})")
    if show_status:
        print(f"Status: rex {target} --status {job_name}")
    print(f"Log:    rex {target} --log {job_name}")
    print(f"Kill:   rex {target} --kill {job_name}")

    return JobInfo(
        job_id=job_name,
        log_path=log_path,
        is_slurm=False,
        pid=pid,
    )


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
        subprocess.run(
            ["ssh", *self.ssh._opts, self.ssh.target, f"cat > {remote_py}"],
            input=script_content.encode(),
            check=True,
        )

        # Build wrapper script
        builder = ScriptBuilder().shebang(login=True)
        builder.apply_context(ctx)
        builder.run_python(ctx.python, remote_py, args)

        wrapper = builder.build()

        # Execute via script method
        exit_code = self.ssh.exec_script_streaming(wrapper, tty=None)

        # Cleanup
        self.ssh.exec(f"rm -f {remote_py} {remote_sh}")

        return exit_code

    def run_detached(
        self,
        ctx: ExecutionContext,
        script_path: Path,
        args: list[str],
        job_name: str,
    ) -> JobInfo:
        """Run Python script detached in background."""
        remote_script = f"/tmp/rex-{job_name}.py"
        remote_log = f"/tmp/rex-{job_name}.log"

        # Copy script via scp
        subprocess.run(
            ["scp", "-q", str(script_path), f"{self.ssh.target}:{remote_script}"],
            check=True,
        )

        # Build command
        context_cmds = build_context_commands(ctx)
        env_prefix = "; ".join(context_cmds) + "; " if context_cmds else ""

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

        return _run_detached_nohup(
            self.ssh, wrapper, remote_log, job_name, show_status=True
        )

    def exec_foreground(self, ctx: ExecutionContext, cmd: str) -> int:
        """Execute shell command in foreground."""
        # Check for heredoc delimiter collision
        if "\nREXCMD\n" in f"\n{cmd}\n":
            from rex.output import error
            error("Command contains 'REXCMD' as a line, which conflicts with internal delimiter")
            return 1

        script_id = generate_script_id()
        remote_cmd = f"/tmp/rex-exec-{script_id}.sh"

        # Build setup prefix
        context_cmds = build_context_commands(ctx)
        prefix = "\n".join(context_cmds) + "\n" if context_cmds else ""

        # Write command to temp file using heredoc (preserves all quoting)
        # The quoted 'REXCMD' prevents any shell interpretation
        write_script = f"""cat > {remote_cmd} << 'REXCMD'
#!/bin/bash -l
{prefix}{cmd}
REXCMD
chmod +x {remote_cmd}"""

        # Write the command file
        code, _, stderr = self.ssh.exec(write_script)
        if code != 0:
            from rex.output import warn
            warn(f"Failed to write command script: {stderr}")
            return code

        # Execute and cleanup
        return self.ssh.exec_streaming(
            f"{remote_cmd}; _e=$?; rm -f {remote_cmd}; exit $_e",
            tty=None,
        )

    def exec_detached(
        self, ctx: ExecutionContext, cmd: str, job_name: str
    ) -> JobInfo:
        """Execute shell command detached."""
        remote_log = f"/tmp/rex-{job_name}.log"

        # Build prefix
        context_cmds = build_context_commands(ctx)
        prefix = " && ".join(context_cmds) + " && " if context_cmds else ""

        full_cmd = prefix + cmd

        # Escape single quotes for bash -c
        escaped_cmd = full_cmd.replace("'", "'\\''")

        return _run_detached_nohup(
            self.ssh, escaped_cmd, remote_log, job_name, login_shell=True
        )

    def list_jobs(self, since_minutes: int = 0) -> list[JobStatus]:
        """List all rex jobs on remote."""
        # Note: since_minutes not supported for direct execution (no job history)
        script = '''
for log in /tmp/rex-*.log ~/.rex/rex-*.log; do
    [ -f "$log" ] || continue
    job=$(basename "$log" .log | sed "s/rex-//")
    pattern="rex-${job}[.]py"
    pid=$(pgrep -f "$pattern" 2>/dev/null | head -1)
    if [ -n "$pid" ]; then
        status="running"
    else
        status="completed"
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
        code, stdout, _ = self.ssh.exec(f"bash -c {shell_quote(script)}")

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
                    hostname=self.ssh.target,
                ))
        return jobs

    def get_status(self, job_id: str) -> JobStatus:
        """Get status of specific job."""
        pattern = job_pattern(job_id)
        code, stdout, _ = self.ssh.exec(f"pgrep -f '{pattern}' | head -1")

        if code != 0:
            return JobStatus(job_id=job_id, status="unknown")

        pid = int(stdout.strip()) if stdout.strip() else None
        status = "running" if pid else "completed"
        return JobStatus(job_id=job_id, status=status, pid=pid)

    def get_log_path(self, job_id: str) -> str | None:
        """Get log file path for a job."""
        return _get_log_path(self.ssh, job_id)

    def kill_job(self, job_id: str) -> bool:
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

    def watch_job(self, job_id: str, poll_interval: int = 5) -> JobResult:
        """Wait for job to complete."""
        max_failures = 3
        failures = 0

        while True:
            status = self.get_status(job_id)

            if status.status == "unknown":
                failures += 1
                if failures >= max_failures:
                    warn(f"Lost connection after {max_failures} attempts")
                    return JobResult(job_id=job_id, status="unknown", exit_code=1)
                warn(f"Connection failed (attempt {failures}/{max_failures}), retrying...")
                time.sleep(poll_interval)
                continue

            failures = 0

            if status.status == "completed":
                success(f"Job {job_id} completed")
                return JobResult(job_id=job_id, status="completed", exit_code=0)

            time.sleep(poll_interval)
