"""Tests for config merging logic."""

import argparse
import pytest
from pathlib import Path

from rex.cli import merge_configs, resolve_paths
from rex.config.global_config import HostConfig
from rex.config.project import ProjectConfig


def make_args(**kwargs) -> argparse.Namespace:
    """Create args namespace with defaults."""
    defaults = {
        "partition": None,
        "gres": None,
        "time": None,
        "cpus": None,
        "mem": None,
        "constraint": None,
        "prefer": None,
        "gpu": False,
        "cpu": False,
        "modules": [],
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def make_project(tmp_path: Path, **kwargs) -> ProjectConfig:
    """Create a ProjectConfig with given values."""
    defaults = {
        "root": tmp_path,
        "name": "test-project",
        "code_dir": None,
        "run_dir": None,
        "modules": None,
        "cpu_partition": None,
        "gpu_partition": None,
        "gres": None,
        "time": None,
        "cpus": None,
        "mem": None,
        "constraint": None,
        "prefer": None,
        "default_gpu": None,
        "env": {},
    }
    defaults.update(kwargs)
    return ProjectConfig(**defaults)


class TestMergeConfigs:
    """Tests for merge_configs function."""

    def test_cli_overrides_all(self, tmp_path):
        """CLI arguments override project and host config."""
        args = make_args(
            partition="cli-partition",
            gres="gpu:2",
            time="2:00:00",
            modules=["cli-module"],
        )
        project = make_project(
            tmp_path,
            gpu_partition="proj-gpu",
            gres="gpu:1",
            time="1:00:00",
            modules=["proj-module"],
        )
        host_config = HostConfig(
            gpu_partition="host-gpu",
            gres="gpu:4",
            time="4:00:00",
            modules=["host-module"],
        )

        slurm_opts, modules, use_gpu, env = merge_configs(args, project, host_config)

        assert slurm_opts["partition"] == "cli-partition"
        assert slurm_opts["gres"] == "gpu:2"
        assert slurm_opts["time"] == "2:00:00"
        assert modules == ["cli-module"]

    def test_project_overrides_host(self, tmp_path):
        """Project config overrides host config."""
        args = make_args()
        project = make_project(
            tmp_path,
            gpu_partition="proj-gpu",
            gres="gpu:1",
            time="1:00:00",
            modules=["proj-module"],
            default_gpu=True,
        )
        host_config = HostConfig(
            gpu_partition="host-gpu",
            gres="gpu:4",
            time="4:00:00",
            modules=["host-module"],
        )

        slurm_opts, modules, use_gpu, env = merge_configs(args, project, host_config)

        assert slurm_opts["partition"] == "proj-gpu"
        assert slurm_opts["gres"] == "gpu:1"
        assert slurm_opts["time"] == "1:00:00"
        assert modules == ["proj-module"]
        assert use_gpu is True

    def test_host_defaults(self, tmp_path):
        """Host config used when project doesn't override."""
        args = make_args()
        project = make_project(tmp_path)  # Minimal project
        host_config = HostConfig(
            cpu_partition="host-cpu",
            gpu_partition="host-gpu",
            time="4:00:00",
            modules=["host-module"],
        )

        slurm_opts, modules, use_gpu, env = merge_configs(args, project, host_config)

        assert slurm_opts["partition"] == "host-cpu"
        assert slurm_opts["time"] == "4:00:00"
        assert modules == ["host-module"]
        assert use_gpu is False

    def test_gpu_flag_selects_gpu_partition(self, tmp_path):
        """--gpu flag selects GPU partition."""
        args = make_args(gpu=True)
        project = make_project(tmp_path)
        host_config = HostConfig(
            cpu_partition="cpu",
            gpu_partition="gpu",
            gres="gpu:1",
            prefer="GPU_SKU:H100",
        )

        slurm_opts, modules, use_gpu, env = merge_configs(args, project, host_config)

        assert slurm_opts["partition"] == "gpu"
        assert slurm_opts["gres"] == "gpu:1"
        assert slurm_opts["prefer"] == "GPU_SKU:H100"
        assert use_gpu is True

    def test_cpu_flag_selects_cpu_partition(self, tmp_path):
        """--cpu flag selects CPU partition."""
        args = make_args(cpu=True)
        project = make_project(tmp_path, default_gpu=True, gpu_partition="gpu")
        host_config = HostConfig(
            cpu_partition="cpu",
            gpu_partition="gpu",
        )

        slurm_opts, modules, use_gpu, env = merge_configs(args, project, host_config)

        assert slurm_opts["partition"] == "cpu"
        assert use_gpu is False

    def test_default_gpu_uses_gpu_partition(self, tmp_path):
        """default_gpu=true uses GPU partition by default."""
        args = make_args()
        project = make_project(tmp_path)
        host_config = HostConfig(
            cpu_partition="cpu",
            gpu_partition="gpu",
            gres="gpu:1",
            default_gpu=True,
        )

        slurm_opts, modules, use_gpu, env = merge_configs(args, project, host_config)

        assert slurm_opts["partition"] == "gpu"
        assert slurm_opts["gres"] == "gpu:1"
        assert use_gpu is True

    def test_project_default_gpu_overrides_host(self, tmp_path):
        """Project default_gpu overrides host default_gpu."""
        args = make_args()
        project = make_project(tmp_path, default_gpu=False, cpu_partition="proj-cpu")
        host_config = HostConfig(
            cpu_partition="host-cpu",
            gpu_partition="host-gpu",
            default_gpu=True,
        )

        slurm_opts, modules, use_gpu, env = merge_configs(args, project, host_config)

        assert slurm_opts["partition"] == "proj-cpu"
        assert use_gpu is False

    def test_gres_only_applied_when_using_gpu(self, tmp_path):
        """Host gres only applied when using GPU partition."""
        args = make_args()
        project = make_project(tmp_path)
        host_config = HostConfig(
            cpu_partition="cpu",
            gpu_partition="gpu",
            gres="gpu:1",
        )

        slurm_opts, modules, use_gpu, env = merge_configs(args, project, host_config)

        # Using CPU partition, so host gres should not apply
        assert slurm_opts["partition"] == "cpu"
        assert slurm_opts["gres"] is None
        assert use_gpu is False

    def test_env_merged(self, tmp_path):
        """Environment variables are merged (host < project)."""
        args = make_args()
        project = make_project(
            tmp_path,
            env={"PROJ_VAR": "proj", "SHARED": "from_proj"},
        )
        host_config = HostConfig(
            env={"HOST_VAR": "host", "SHARED": "from_host"},
        )

        slurm_opts, modules, use_gpu, env = merge_configs(args, project, host_config)

        assert env["HOST_VAR"] == "host"
        assert env["PROJ_VAR"] == "proj"
        assert env["SHARED"] == "from_proj"  # Project overrides

    def test_no_project(self):
        """Works without project config."""
        args = make_args()
        host_config = HostConfig(
            cpu_partition="cpu",
            time="1:00:00",
            modules=["mod"],
        )

        slurm_opts, modules, use_gpu, env = merge_configs(args, None, host_config)

        assert slurm_opts["partition"] == "cpu"
        assert slurm_opts["time"] == "1:00:00"
        assert modules == ["mod"]

    def test_no_host_config(self, tmp_path):
        """Works without host config."""
        args = make_args()
        project = make_project(
            tmp_path,
            cpu_partition="proj-cpu",
            modules=["proj-mod"],
        )

        slurm_opts, modules, use_gpu, env = merge_configs(args, project, None)

        assert slurm_opts["partition"] == "proj-cpu"
        assert modules == ["proj-mod"]


class TestResolvePaths:
    """Tests for resolve_paths function."""

    def test_no_project(self):
        """Returns None when no project."""
        code_dir, run_dir = resolve_paths(None, None)
        assert code_dir is None
        assert run_dir is None

    def test_project_code_dir_used_directly(self, tmp_path):
        """Project code_dir used as-is when specified."""
        project = make_project(tmp_path, code_dir="/custom/code")
        host_config = HostConfig(code_dir="/host/base")

        code_dir, run_dir = resolve_paths(project, host_config)

        assert code_dir == "/custom/code"

    def test_host_code_dir_with_project_name(self, tmp_path):
        """Host code_dir + project name when project doesn't specify."""
        project = make_project(tmp_path, name="my-project")
        host_config = HostConfig(code_dir="/host/base")

        code_dir, run_dir = resolve_paths(project, host_config)

        assert code_dir == "/host/base/my-project"

    def test_project_run_dir_used_directly(self, tmp_path):
        """Project run_dir used as-is when specified."""
        project = make_project(tmp_path, run_dir="/custom/run")
        host_config = HostConfig(run_dir="/host/scratch")

        code_dir, run_dir = resolve_paths(project, host_config)

        assert run_dir == "/custom/run"

    def test_host_run_dir_with_project_name(self, tmp_path):
        """Host run_dir + project name when project doesn't specify."""
        project = make_project(tmp_path, name="my-project")
        host_config = HostConfig(run_dir="/host/scratch")

        code_dir, run_dir = resolve_paths(project, host_config)

        assert run_dir == "/host/scratch/my-project"

    def test_both_paths_resolved(self, tmp_path):
        """Both paths resolved together."""
        project = make_project(tmp_path, name="flash-eq")
        host_config = HostConfig(
            code_dir="/home/groups/rhiju/hmblair",
            run_dir="/scratch/users/hmblair",
        )

        code_dir, run_dir = resolve_paths(project, host_config)

        assert code_dir == "/home/groups/rhiju/hmblair/flash-eq"
        assert run_dir == "/scratch/users/hmblair/flash-eq"

    def test_no_host_config(self, tmp_path):
        """Returns None paths when no host config and project doesn't specify."""
        project = make_project(tmp_path)

        code_dir, run_dir = resolve_paths(project, None)

        assert code_dir is None
        assert run_dir is None
