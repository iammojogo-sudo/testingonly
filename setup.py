"""
Hunyuan3D-2mv - Modly extension setup script.

Called by Modly at install time:
    python setup.py <json_args>

json_args keys:
    python_exe  - path to Modly's embedded Python
    ext_dir     - absolute path to this extension directory
    gpu_sm      - GPU compute capability as integer (e.g. 89 for RTX 4050)
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def pip(venv, *args):
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe)] + list(args), check=True)


def python_exe_in_venv(venv):
    is_win = platform.system() == "Windows"
    return venv / ("Scripts/python.exe" if is_win else "bin/python")


def setup(python_exe, ext_dir, gpu_sm):
    venv = ext_dir / "venv"
    is_win = platform.system() == "Windows"

    print("[setup] Creating venv at %s ..." % venv)
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ------------------------------------------------------------------ #
    # PyTorch
    # ------------------------------------------------------------------ #
    if gpu_sm >= 100:
        torch_index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs = ["torch>=2.7.0", "torchvision>=0.22.0", "torchaudio>=2.7.0"]
        print("[setup] SM %d (Blackwell) -> PyTorch 2.7 + CUDA 12.8" % gpu_sm)
    elif gpu_sm >= 70:
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
        print("[setup] SM %d -> PyTorch 2.5.1 + CUDA 12.4" % gpu_sm)
    else:
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
        print("[setup] SM %d (legacy) -> PyTorch 2.5.1 + CUDA 11.8" % gpu_sm)

    print("[setup] Installing PyTorch...")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # ------------------------------------------------------------------ #
    # xformers
    # ------------------------------------------------------------------ #
    print("[setup] Installing xformers...")
    if gpu_sm >= 70:
        pip(venv, "install", "xformers==0.0.28.post3", "--index-url", torch_index)
    else:
        pip(venv, "install", "xformers==0.0.28.post2", "--index-url",
            "https://download.pytorch.org/whl/cu118")

    # ------------------------------------------------------------------ #
    # Clone Hunyuan3D-2 repo and install hy3dgen package
    # ------------------------------------------------------------------ #
    repo_dir = ext_dir / "Hunyuan3D-2"
    if not repo_dir.exists():
        print("[setup] Cloning Hunyuan3D-2 repo...")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git",
             str(repo_dir)],
            check=True
        )
    else:
        print("[setup] Repo already exists, skipping clone.")

    print("[setup] Installing hy3dgen package...")
    venv_python = python_exe_in_venv(venv)
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-e", str(repo_dir)],
        check=True
    )

    # ------------------------------------------------------------------ #
    # Core dependencies
    # ------------------------------------------------------------------ #
    print("[setup] Installing core dependencies...")
    pip(venv, "install",
        "transformers==4.40.2",
        "diffusers==0.27.2",
        "huggingface_hub==0.23.5",
        "accelerate",
        "omegaconf",
        "einops",
        "Pillow",
        "numpy",
        "scipy",
        "trimesh",
        "pymeshlab",
        "pygltflib",
        "opencv-python-headless",
        "tqdm",
        "safetensors",
        "rembg",
        "onnxruntime",
    )

    # ------------------------------------------------------------------ #
    # Paint pipeline extra dependencies
    # ------------------------------------------------------------------ #
    print("[setup] Installing paint pipeline dependencies...")
    pip(venv, "install",
        "ninja",           # speeds up C++ extension compilation
        "jinja2",
        "matplotlib",
        "scikit-image",
        "open3d",          # used by hy3dgen texture baking
    )

    # ------------------------------------------------------------------ #
    # onnxruntime-gpu if supported
    # ------------------------------------------------------------------ #
    if gpu_sm >= 70:
        print("[setup] Installing onnxruntime-gpu...")
        try:
            pip(venv, "install", "onnxruntime-gpu")
        except subprocess.CalledProcessError:
            print("[setup] onnxruntime-gpu failed, falling back to cpu.")
            pip(venv, "install", "onnxruntime")

    # ------------------------------------------------------------------ #
    # Compile custom_rasterizer  (required for texture paint pipeline)
    # ------------------------------------------------------------------ #
    rasterizer_dir = repo_dir / "hy3dgen" / "texgen" / "custom_rasterizer"
    if rasterizer_dir.exists():
        print("[setup] Compiling custom_rasterizer (this may take a few minutes)...")
        env = os.environ.copy()

        if is_win:
            # On Windows we need MSVC on the PATH.
            # Try to locate it via vswhere — ships with every VS / Build Tools install.
            vswhere = Path(
                r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
            )
            if vswhere.exists():
                result = subprocess.run(
                    [str(vswhere), "-latest", "-property", "installationPath"],
                    capture_output=True, text=True
                )
                vs_path = result.stdout.strip()
                if vs_path:
                    vcvars = (
                        Path(vs_path)
                        / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                    )
                    if vcvars.exists():
                        print("[setup] Found MSVC at: %s" % vs_path)
                        # Run vcvars64.bat then build in a single shell call
                        build_cmd = (
                            'call "%s" && "%s" setup.py build_ext --inplace'
                            % (vcvars, venv_python)
                        )
                        subprocess.run(
                            ["cmd", "/c", build_cmd],
                            cwd=str(rasterizer_dir),
                            env=env,
                            check=True
                        )
                    else:
                        print("[setup] WARNING: vcvars64.bat not found — "
                              "custom_rasterizer compilation skipped. "
                              "Install Visual Studio Build Tools to enable texturing.")
                else:
                    print("[setup] WARNING: vswhere returned no path — "
                          "custom_rasterizer compilation skipped.")
            else:
                print("[setup] WARNING: vswhere.exe not found at expected location. "
                      "custom_rasterizer compilation skipped. "
                      "Install Visual Studio Build Tools 2022 to enable texturing.")
        else:
            # Linux / macOS — gcc should be available
            subprocess.run(
                [str(venv_python), "setup.py", "build_ext", "--inplace"],
                cwd=str(rasterizer_dir),
                env=env,
                check=True
            )
        print("[setup] custom_rasterizer compilation done.")
    else:
        print("[setup] WARNING: custom_rasterizer directory not found at %s. "
              "The paint node will not work without it." % rasterizer_dir)

    print("[setup] Done. Venv ready at: %s" % venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe=sys.argv[1],
            ext_dir=Path(sys.argv[2]),
            gpu_sm=int(sys.argv[3]),
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe=args["python_exe"],
            ext_dir=Path(args["ext_dir"]),
            gpu_sm=int(args["gpu_sm"]),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":89}\'')
        sys.exit(1)
