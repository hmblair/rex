"""Tests for config merging logic."""

import argparse
import pytest
from pathlib import Path

from rex.cli import merge_configs, resolve_config, resolve_paths
from rex.config.global_config import HostConfig
from rex.config.project import ProjectConfig
from rex.config.resolved import ResolvedConfig
from rex.execution.base import ExecutionContext
from rex.execution.slurm import SlurmOptions


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
        "python": "python3",
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


class TestBuildCodeDirResolution:
    """Test that build uses resolved code_dir from host config."""

    def test_build_uses_host_code_dir(self, tmp_path, mocker):
        """Build works when code_dir comes from host config, not project."""
        # Project has NO code_dir
        project = make_project(tmp_path, name="my-project", code_dir=None)

        # Host config HAS code_dir
        host_config = HostConfig(code_dir="/host/base")

        # Resolve config (as CLI does)
        args = make_args()
        config = resolve_config(args, project, host_config)

        # Mock SSH to avoid real connection
        mock_ssh = mocker.Mock()
        mock_ssh.exec.return_value = (0, "12345", "")
        mock_ssh._opts = []
        mock_ssh.target = "test-host"

        mocker.patch("subprocess.run")

        from rex.commands.build import build

        result = build(mock_ssh, config)

        assert result == 0
        assert config.execution.code_dir == "/host/base/my-project"

    def test_build_uses_resolved_modules(self, tmp_path, mocker):
        """Build uses modules from resolved config, not raw project."""
        # Project has NO modules
        project = make_project(tmp_path, name="my-project", modules=None)

        # Host config HAS modules
        host_config = HostConfig(
            code_dir="/host/base",
            modules=["cuda/12.0", "python/3.11"],
        )

        # Resolve config
        args = make_args()
        config = resolve_config(args, project, host_config)

        # Verify modules are resolved from host config
        assert config.execution.modules == ["cuda/12.0", "python/3.11"]

        # Mock SSH
        mock_ssh = mocker.Mock()
        mock_ssh.exec.return_value = (0, "12345", "")
        mock_ssh._opts = []
        mock_ssh.target = "test-host"

        mock_run = mocker.patch("subprocess.run")

        from rex.commands.build import build

        result = build(mock_ssh, config)

        assert result == 0

        # Verify the script content includes module load
        call_args = mock_run.call_args
        script_content = call_args.kwargs["input"].decode()
        assert "module load cuda/12.0 python/3.11" in script_content


class TestResolveConfig:
    """Tests for resolve_config function and ResolvedConfig composition."""

    def test_returns_resolved_config(self, tmp_path):
        """resolve_config returns a ResolvedConfig instance."""
        args = make_args()
        project = make_project(tmp_path)
        host_config = HostConfig()

        config = resolve_config(args, project, host_config)

        assert isinstance(config, ResolvedConfig)

    def test_execution_context_composed(self, tmp_path):
        """ResolvedConfig contains ExecutionContext."""
        args = make_args()
        project = make_project(tmp_path)
        host_config = HostConfig()

        config = resolve_config(args, project, host_config)

        assert isinstance(config.execution, ExecutionContext)

    def test_slurm_options_composed(self, tmp_path):
        """ResolvedConfig contains SlurmOptions."""
        args = make_args()
        project = make_project(tmp_path)
        host_config = HostConfig()

        config = resolve_config(args, project, host_config)

        assert isinstance(config.slurm, SlurmOptions)

    def test_identity_fields_from_project(self, tmp_path):
        """name and root come from project."""
        args = make_args()
        project = make_project(tmp_path, name="my-project")
        host_config = HostConfig()

        config = resolve_config(args, project, host_config)

        assert config.name == "my-project"
        assert config.root == tmp_path

    def test_identity_fields_none_without_project(self):
        """name and root are None without project."""
        args = make_args()

        config = resolve_config(args, None, None)

        assert config.name is None
        assert config.root is None

    def test_execution_python_from_args(self, tmp_path):
        """ExecutionContext.python comes from args."""
        args = make_args(python="python3.11")
        project = make_project(tmp_path)

        config = resolve_config(args, project, None)

        assert config.execution.python == "python3.11"

    def test_execution_python_default(self, tmp_path):
        """ExecutionContext.python defaults to python3."""
        args = make_args()  # python="python3" is default
        project = make_project(tmp_path)

        config = resolve_config(args, project, None)

        assert config.execution.python == "python3"

    def test_execution_modules_from_cli(self, tmp_path):
        """CLI modules override project and host."""
        args = make_args(modules=["cli-mod"])
        project = make_project(tmp_path, modules=["proj-mod"])
        host_config = HostConfig(modules=["host-mod"])

        config = resolve_config(args, project, host_config)

        assert config.execution.modules == ["cli-mod"]

    def test_execution_modules_from_project(self, tmp_path):
        """Project modules used when CLI doesn't specify."""
        args = make_args(modules=[])
        project = make_project(tmp_path, modules=["proj-mod"])
        host_config = HostConfig(modules=["host-mod"])

        config = resolve_config(args, project, host_config)

        assert config.execution.modules == ["proj-mod"]

    def test_execution_modules_from_host(self, tmp_path):
        """Host modules used as fallback."""
        args = make_args(modules=[])
        project = make_project(tmp_path, modules=None)
        host_config = HostConfig(modules=["host-mod"])

        config = resolve_config(args, project, host_config)

        assert config.execution.modules == ["host-mod"]

    def test_execution_modules_empty_default(self, tmp_path):
        """Modules default to empty list."""
        args = make_args(modules=[])
        project = make_project(tmp_path, modules=None)

        config = resolve_config(args, project, None)

        assert config.execution.modules == []

    def test_execution_code_dir_resolved(self, tmp_path):
        """code_dir is resolved from host + project name."""
        args = make_args()
        project = make_project(tmp_path, name="my-proj")
        host_config = HostConfig(code_dir="/base")

        config = resolve_config(args, project, host_config)

        assert config.execution.code_dir == "/base/my-proj"

    def test_execution_run_dir_resolved(self, tmp_path):
        """run_dir is resolved from host + project name."""
        args = make_args()
        project = make_project(tmp_path, name="my-proj")
        host_config = HostConfig(run_dir="/scratch")

        config = resolve_config(args, project, host_config)

        assert config.execution.run_dir == "/scratch/my-proj"

    def test_execution_env_merged(self, tmp_path):
        """Environment variables merged into ExecutionContext."""
        args = make_args()
        project = make_project(tmp_path, env={"PROJ": "val1", "SHARED": "proj"})
        host_config = HostConfig(env={"HOST": "val2", "SHARED": "host"})

        config = resolve_config(args, project, host_config)

        assert config.execution.env["PROJ"] == "val1"
        assert config.execution.env["HOST"] == "val2"
        assert config.execution.env["SHARED"] == "proj"  # Project wins

    def test_execution_env_empty_default(self, tmp_path):
        """Env defaults to empty dict."""
        args = make_args()
        project = make_project(tmp_path, env={})

        config = resolve_config(args, project, None)

        assert config.execution.env == {}

    def test_slurm_partition_from_cli(self, tmp_path):
        """CLI partition overrides all."""
        args = make_args(partition="cli-part")
        project = make_project(tmp_path, cpu_partition="proj-part")
        host_config = HostConfig(cpu_partition="host-part")

        config = resolve_config(args, project, host_config)

        assert config.slurm.partition == "cli-part"

    def test_slurm_gres_from_cli(self, tmp_path):
        """CLI gres overrides all."""
        args = make_args(gres="gpu:2")
        project = make_project(tmp_path, gres="gpu:1")

        config = resolve_config(args, project, None)

        assert config.slurm.gres == "gpu:2"

    def test_slurm_time_from_project(self, tmp_path):
        """Project time used when CLI doesn't specify."""
        args = make_args()
        project = make_project(tmp_path, time="2:00:00")
        host_config = HostConfig(time="4:00:00")

        config = resolve_config(args, project, host_config)

        assert config.slurm.time == "2:00:00"

    def test_slurm_cpus_from_host(self, tmp_path):
        """Host cpus used as fallback."""
        args = make_args()
        project = make_project(tmp_path)
        host_config = HostConfig(cpus=8)

        config = resolve_config(args, project, host_config)

        assert config.slurm.cpus == 8

    def test_slurm_mem_resolved(self, tmp_path):
        """Memory resolved through priority chain."""
        args = make_args(mem="16G")

        config = resolve_config(args, None, None)

        assert config.slurm.mem == "16G"

    def test_slurm_constraint_resolved(self, tmp_path):
        """Constraint resolved through priority chain."""
        args = make_args()
        project = make_project(tmp_path, constraint="skylake")

        config = resolve_config(args, project, None)

        assert config.slurm.constraint == "skylake"

    def test_slurm_prefer_resolved(self, tmp_path):
        """Prefer resolved through priority chain."""
        args = make_args(gpu=True)
        host_config = HostConfig(
            gpu_partition="gpu",
            prefer="GPU_SKU:H100",
        )

        config = resolve_config(args, None, host_config)

        assert config.slurm.prefer == "GPU_SKU:H100"

    def test_all_none_inputs(self):
        """Works with no project and no host config."""
        args = make_args()

        config = resolve_config(args, None, None)

        assert config.name is None
        assert config.root is None
        assert config.execution.python == "python3"
        assert config.execution.modules == []
        assert config.execution.code_dir is None
        assert config.execution.run_dir is None
        assert config.slurm.partition is None

    def test_full_config_composition(self, tmp_path):
        """Full integration: all fields populated correctly."""
        args = make_args(
            python="python3.10",
            modules=["cuda/12"],
            partition="gpu-h100",
            gres="gpu:4",
            time="8:00:00",
            cpus=16,
            mem="64G",
        )
        project = make_project(
            tmp_path,
            name="ml-project",
            env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
        )
        host_config = HostConfig(
            code_dir="/home/user/code",
            run_dir="/scratch/user",
        )

        config = resolve_config(args, project, host_config)

        # Identity
        assert config.name == "ml-project"
        assert config.root == tmp_path

        # Execution
        assert config.execution.python == "python3.10"
        assert config.execution.modules == ["cuda/12"]
        assert config.execution.code_dir == "/home/user/code/ml-project"
        assert config.execution.run_dir == "/scratch/user/ml-project"
        assert config.execution.env == {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}

        # SLURM
        assert config.slurm.partition == "gpu-h100"
        assert config.slurm.gres == "gpu:4"
        assert config.slurm.time == "8:00:00"
        assert config.slurm.cpus == 16
        assert config.slurm.mem == "64G"
