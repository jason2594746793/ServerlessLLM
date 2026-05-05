# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
"""Command builders for ServerlessLLM inference backends."""

from typing import Tuple

from sllm.database import Deployment

import os

# Detect environment
if os.path.exists("/opt/venvs"):
    VENV_VLLM = "/opt/venvs/vllm"
    VENV_SGLANG = "/opt/venvs/sglang"
    VENV_SLLM_STORE = "/opt/venvs/sllm-store"
else:
    # Local development fallback
    # Assuming standard structure: root/.venv
    # We need absolute path for Pylet
    _cwd = os.getcwd()
    _venv_path = os.path.join(_cwd, ".venv")
    if not os.path.exists(_venv_path):
         # Try to find it if we are in a subdir
         _venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.venv"))

    VENV_VLLM = _venv_path
    VENV_SGLANG = _venv_path
    VENV_SLLM_STORE = _venv_path


def check_backend_available(backend: str) -> None:
    """
    Check if a backend is available (importable).

    Raises ImportError with helpful message if not installed.
    Note: This checks the current Python environment. On distributed setups,
    backends run in separate venvs on worker nodes.

    Args:
        backend: Backend name ("vllm" or "sglang")

    Raises:
        ImportError: If backend is not installed
        ValueError: If backend is unknown
    """
    if backend == "vllm":
        try:
            import vllm  # noqa: F401
        except ImportError:
            raise ImportError(
                "vLLM is required but not installed. "
                "Install it with: pip install vllm"
            ) from None
    elif backend == "sglang":
        try:
            import sglang  # noqa: F401
        except ImportError:
            raise ImportError(
                "SGLang is required but not installed. "
                "Install it with: pip install sglang"
            ) from None
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Supported backends: vllm, sglang"
        )


def _has_sllm_store_shards(model_dir: str, tp: int) -> bool:
    """True iff the model directory has sllm-store shards for the given TP.

    The sllm-store native format saves one ``rank_<i>/tensor.data_0`` per TP
    rank. vLLM's ``ServerlessLLMLoader`` (registered via the patch in
    ``sllm_store/vllm_patch/sllm_load.patch``) expects all ranks to be
    present. If even one shard is missing — e.g. a partial save was
    interrupted, or the model was originally saved with a different TP —
    we fall back to the HF safetensors loader rather than crashing in
    ``load_dict()``.
    """
    if not os.path.isdir(model_dir):
        return False
    for r in range(tp):
        if not os.path.isfile(
            os.path.join(model_dir, f"rank_{r}", "tensor.data_0")
        ):
            return False
    return True


def build_vllm_command(
    deployment: Deployment, storage_path: str = "/models"
) -> Tuple[str, str]:
    """Build vLLM serve command for Pylet submission."""
    config = deployment.backend_config or {}
    tp = config.get("tensor_parallel_size", 1)
    max_model_len = config.get("max_model_len")
    gpu_memory_utilization = config.get("gpu_memory_utilization")
    dtype = config.get("dtype")
    trust_remote_code = config.get("trust_remote_code", False)
    enforce_eager = config.get("enforce_eager", False)

    # Use sllm-store fast checkpoint loader when shards are on disk for
    # this TP. This activates the ServerlessLLMLoader registered by the
    # vLLM patch (sllm_store/vllm_patch/sllm_load.patch) which mmaps
    # ``rank_<i>/tensor.data_0`` instead of reading HF safetensors.
    # The loader requires:
    #   1. Model arg = local directory path (asserted via os.path.isdir).
    #   2. STORAGE_PATH env var set so it can derive the relative model id.
    # Reconciler sets STORAGE_PATH; we rewrite the model arg here.
    model_dir = os.path.join(storage_path, deployment.model_name)
    use_sllm_load = _has_sllm_store_shards(model_dir, tp)
    model_arg = model_dir if use_sllm_load else deployment.model_name

    cmd_parts = [
        "vllm serve",
        model_arg,
        f"--served-model-name {deployment.model_name}",
        "--port $PORT",
        "--host 0.0.0.0",
        f"--tensor-parallel-size {tp}",
        "--enable-prefix-caching",
    ]

    if use_sllm_load:
        cmd_parts.append("--load-format serverless_llm")

    # Default max_model_len to 4096 if not specified.
    # Many newer models (e.g. Qwen3-8B) default to 40960+ tokens,
    # which exceeds single-GPU VRAM on A5000 (24GB).
    if not max_model_len:
        max_model_len = 4096
    cmd_parts.append(f"--max-model-len {max_model_len}")

    if gpu_memory_utilization:
        cmd_parts.append(f"--gpu-memory-utilization {gpu_memory_utilization}")

    if dtype:
        cmd_parts.append(f"--dtype {dtype}")

    if trust_remote_code:
        cmd_parts.append("--trust-remote-code")

    if enforce_eager:
        cmd_parts.append("--enforce-eager")

    extra_args = config.get("extra_args", [])
    if extra_args:
        if isinstance(extra_args, list):
            cmd_parts.extend(extra_args)
        elif isinstance(extra_args, str):
            cmd_parts.append(extra_args)

    return " ".join(cmd_parts), VENV_VLLM


def build_sglang_command(
    deployment: Deployment, storage_path: str = "/models"
) -> Tuple[str, str]:
    """Build SGLang launch_server command for Pylet submission."""
    config = deployment.backend_config or {}
    tp = config.get("tensor_parallel_size", 1)

    cmd_parts = [
        "python -m sglang.launch_server",
        f"--model-path {deployment.model_name}",
        f"--served-model-name {deployment.model_name}",
        "--port $PORT",
        "--host 0.0.0.0",
        f"--tp {tp}",
    ]

    if config.get("mem_fraction_static"):
        cmd_parts.append(
            f"--mem-fraction-static {config['mem_fraction_static']}"
        )

    if config.get("dtype"):
        cmd_parts.append(f"--dtype {config['dtype']}")

    if config.get("trust_remote_code"):
        cmd_parts.append("--trust-remote-code")

    extra_args = config.get("extra_args", [])
    if extra_args:
        if isinstance(extra_args, list):
            cmd_parts.extend(extra_args)
        elif isinstance(extra_args, str):
            cmd_parts.append(extra_args)

    return " ".join(cmd_parts), VENV_SGLANG


BUILDERS = {
    "vllm": build_vllm_command,
    "sglang": build_sglang_command,
}


def build_instance_command(
    deployment: Deployment, storage_path: str = "/models"
) -> Tuple[str, str]:
    """Build command for a deployment instance."""
    builder = BUILDERS.get(deployment.backend)
    if not builder:
        raise ValueError(f"Unknown backend: {deployment.backend}")
    return builder(deployment, storage_path)
