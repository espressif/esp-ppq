## ESP-PPQ Quantization Tool

ESP-PPQ is a quantization tool based on PPQ, and its [source code](https://github.com/espressif/esp-ppq) is fully open-sourced. Built upon [PPQ](https://github.com/OpenPPL/ppq), ESP-PPQ adds Espressif-customized quantizers and exporters, allowing users to select quantization rules compatible with ESP-DL for different chips and export standardized model files that can be directly loaded by ESP-DL. ESP-PPQ is fully compatible with all PPQ APIs and quantization scripts.

For more details on quantization principles, please refer to the [PPQ documentation and videos](https://github.com/OpenPPL/ppq). For instructions on using ESP-PPQ, see [How to quantize model](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_quantize_model.html).

### Installation (安装方法)

1. Install CUDA from [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)

2. Install Complier

```bash
apt-get install ninja-build # for debian/ubuntu user
yum install ninja-build # for redhat/centos user
```

For Windows User:

  (1) Download ninja.exe from [https://github.com/ninja-build/ninja/releases](https://github.com/ninja-build/ninja/releases), add it to Windows PATH.

  (2) Install Visual Studio 2019 from [https://visualstudio.microsoft.com](https://visualstudio.microsoft.com/zh-hans/).

  (3) Add your C++ compiler to Windows PATH Environment, if you are using Visual Studio, it should be like "C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx86\x86"

  (4) Update PyTorch version to 2.0.0+.

3. Install PPQ

Method 1: Install the package using pip
```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install esp-ppq
```

Method 2: Install from source with pip to stay synchronized with the master branch
```
   git clone https://github.com/espressif/esp-ppq.git
   cd esp-ppq
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install -e .
```

Method 3: Install the package using uv
```
   uv pip install "esp-ppq[cpu]" --torch-backend=cpu
   # GPU
   # uv pip install "esp-ppq[cpu]" --torch-backend=cu124
   # AMD GPU
   # uv pip install "esp-ppq[cpu]" --torch-backend=rocm6.2
   # Intel XPU
   # uv pip install "esp-ppq[cpu]" --torch-backend=xpu
```

Method 4: Install from source using uv to stay in sync with the master branch
```
   git clone https://github.com/espressif/esp-ppq.git
   cd esp-ppq
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   uv pip install -e .
```

Method 5: Use esp-ppq with docker:
```
docker build -t esp-ppq:your_tag https://github.com/espressif/esp-ppq.git
```
**Note**
* The example code installs the Linux PyTorch CPU version. Please install the appropriate PyTorch version based on your actual needs.
* If installing the package with uv, simply modify the ``--torch-backend`` parameter, which will override the PyTorch URLs index configured in the project.

### License

This project is distributed under the [Apache License, Version 2.0](LICENSE).
