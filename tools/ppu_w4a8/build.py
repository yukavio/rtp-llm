"""Build the pinned W4A8 dependency beside the existing DeepGEMM package.

Run with the target runtime's Python, Torch and SDK. Only a fresh work
copy and a fresh output directory are written; no global package is replaced.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

COMMIT = "209dd9fa0c85982971f6657a001f97d26602d0c6"


def bridge_cuda_torch(root):
    """Avoid including native PPU and CUDA half types in the same host TU."""
    shutil.copyfile(
        Path(__file__).with_name("cuda_torch_bridge.hpp"),
        root / "csrc/rtp_cuda_torch_bridge.hpp",
    )
    replacements = {
        "#include <ATen/cuda/CUDAContext.h>": '#include "rtp_cuda_torch_bridge.hpp"',
        "#include <ATen/cuda/CUDADataType.h>": '#include "rtp_cuda_torch_bridge.hpp"',
        "at::cuda::getCurrentCUDAStream": "c10::cuda::getCurrentCUDAStream",
        "at::cuda::current_device": "c10::cuda::current_device",
        "at::cuda::getCurrentDeviceProperties": "dg_cuda_torch::getCurrentDeviceProperties",
        "at::cuda::getDeviceProperties": "dg_cuda_torch::getDeviceProperties",
        "at::cuda::ScalarTypeToCudaDataType": "dg_cuda_torch::scalarType",
        "#define PYTORCH_SUPPORTS_GET_ACBLASLT_HANDLE (TORCH_VERSION_MAJOR > 2 or (TORCH_VERSION_MAJOR == 2 and TORCH_VERSION_MINOR >= 3))": "#define PYTORCH_SUPPORTS_GET_ACBLASLT_HANDLE 0",
    }
    for path in (root / "csrc").rglob("*.hpp"):
        if path.name.startswith("._"):
            continue
        original = text = path.read_text()
        for old, new in replacements.items():
            text = text.replace(old, new)
        if text != original:
            path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cuda-abi-torch", action="store_true")
    args = parser.parse_args()
    source, work, output = (p.resolve() for p in (args.source, args.work, args.output))
    if work.exists() or output.exists():
        parser.error("work and output must be fresh directories")
    if any(
        a == b or a in b.parents
        for a, b in ((source, work), (source, output), (work, output), (output, work))
    ):
        parser.error("source, work and output must be separate directories")
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != COMMIT:
        parser.error(f"expected DeepGEMM commit {COMMIT}, got {actual}")
    subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD"], check=True)
    submodules = subprocess.check_output(
        ["git", "-C", str(source), "submodule", "status", "--recursive"], text=True
    )
    if any(line[:1] != " " for line in submodules.splitlines()):
        parser.error("initialize the pinned submodules before building")
    shutil.copytree(
        source,
        work,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            "build", "dist", "*.egg-info", "__pycache__", "._*"
        ),
    )
    env = os.environ.copy()
    includes = []
    if args.cuda_abi_torch:
        bridge_cuda_torch(work)
        sdk = Path(env.get("PPU_SDK", "/usr/local/PPU_SDK"))
        cuda = Path(env.get("CUDA_HOME", str(sdk / "CUDA_SDK")))
        includes = [str(work / "csrc"), str(cuda / "targets/x86_64-linux/include")]
    if env.get("CPLUS_INCLUDE_PATH"):
        includes.append(env["CPLUS_INCLUDE_PATH"])
    if includes:
        env["CPLUS_INCLUDE_PATH"] = os.pathsep.join(includes)
    subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel"], cwd=work, env=env, check=True
    )
    packages = list((work / "build").glob("lib.*/deep_gemm"))
    if len(packages) != 1:
        raise RuntimeError(f"expected one built package, found {packages}")
    output.mkdir(parents=True)
    shutil.copytree(packages[0], output / "deep_gemm_w4a8")
    shutil.copyfile(work / "LICENSE", output / "deep_gemm_w4a8/LICENSE")
    # Import outside the source checkout, which otherwise shadows old deep_gemm.
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(output), env.get("PYTHONPATH")])
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch, deep_gemm, deep_gemm_w4a8; "
            "print(torch.__version__, deep_gemm.__file__, deep_gemm_w4a8.__file__)",
        ],
        cwd=output,
        env=env,
        check=True,
    )
    import torch

    (output / "build.json").write_text(
        json.dumps(
            {
                "commit": COMMIT,
                "torch": torch.__version__,
                "python": sys.version,
                "cuda_abi_bridge": args.cuda_abi_torch,
                "submodules": submodules,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Add {output} to PYTHONPATH for DSV4_PPU_MOE_ACTIVATION=fp8")


if __name__ == "__main__":
    main()
