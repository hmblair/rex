"""Integration test fixtures using Docker SSH server."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from rex.ssh.executor import SSHExecutor


def docker_available() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        # Check both return code and that we can actually connect
        if result.returncode != 0:
            return False
        # Also verify we don't have connection errors in stderr
        if b"Cannot connect" in result.stderr or b"connect:" in result.stderr:
            return False
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


# Cache the result at module load time
_DOCKER_AVAILABLE = docker_available()


def find_free_port() -> int:
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_ssh(host: str, port: int, timeout: float = 30) -> bool:
    """Wait for SSH server to accept connections."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.5)
    return False


# Skip all tests in this module if Docker is not available
pytestmark = pytest.mark.skipif(
    not _DOCKER_AVAILABLE,
    reason="Docker not available"
)


@pytest.fixture(scope="session")
def docker_ssh_image():
    """Build the test SSH server image."""
    image_name = "rex-test-ssh:latest"
    dockerfile_dir = Path(__file__).parent

    result = subprocess.run(
        ["docker", "build", "-t", image_name, str(dockerfile_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(f"Failed to build Docker image: {result.stderr}")

    yield image_name

    # Cleanup image after all tests
    subprocess.run(["docker", "rmi", "-f", image_name], capture_output=True)


@pytest.fixture(scope="function")
def ssh_server(docker_ssh_image, tmp_path):
    """Start SSH server container and return SSHExecutor connected to it.

    Uses SSH key authentication generated per-test.
    """
    container_name = f"rex-test-{os.getpid()}-{time.time_ns()}"
    port = find_free_port()

    # Generate temporary SSH key pair
    key_path = tmp_path / "test_key"
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", "", "-q"],
        check=True,
    )
    pub_key = (tmp_path / "test_key.pub").read_text().strip()

    # Start container
    result = subprocess.run(
        [
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{port}:22",
            docker_ssh_image,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(f"Failed to start container: {result.stderr}")

    try:
        # Wait for SSH to be ready
        if not wait_for_ssh("127.0.0.1", port):
            logs = subprocess.run(
                ["docker", "logs", container_name],
                capture_output=True, text=True
            )
            pytest.fail(f"SSH server not ready. Logs: {logs.stdout}{logs.stderr}")

        # Add public key to container
        subprocess.run(
            ["docker", "exec", container_name, "mkdir", "-p", "/home/test/.ssh"],
            check=True,
        )
        subprocess.run(
            ["docker", "exec", container_name, "sh", "-c",
             f'echo "{pub_key}" > /home/test/.ssh/authorized_keys'],
            check=True,
        )
        subprocess.run(
            ["docker", "exec", container_name, "chown", "-R", "test:test", "/home/test/.ssh"],
            check=True,
        )
        subprocess.run(
            ["docker", "exec", container_name, "chmod", "600", "/home/test/.ssh/authorized_keys"],
            check=True,
        )

        # Create SSHExecutor with custom options for test server
        ssh = SSHExecutor(
            target=f"test@127.0.0.1",
            opts=[
                "-p", str(port),
                "-i", str(key_path),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
            ],
        )

        # Verify connection works
        code, stdout, stderr = ssh.exec("echo ready")
        if code != 0 or "ready" not in stdout:
            pytest.fail(f"SSH connection failed: {stderr}")

        # Store port and container info for tests that need it
        ssh._test_port = port
        ssh._test_container = container_name
        ssh._test_key = str(key_path)

        yield ssh

    finally:
        # Cleanup container
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
        )
