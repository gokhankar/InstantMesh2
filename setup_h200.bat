@echo off
REM InstantMesh H200 Quick Start Script for Windows
REM Bu script H200 GPU için optimize edilmiş InstantMesh kurulumunu yapar

echo ========================================
echo InstantMesh H200 Optimized Setup
echo ========================================
echo.

REM Sistem bilgisi
echo System Information:
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo.

REM Conda ortamı oluştur
echo Creating conda environment...
conda create -n instantmesh python=3.10 -y

REM Ortamı aktifleştir
call conda activate instantmesh

REM Pip güncelle
echo Upgrading pip...
python -m pip install --upgrade pip

REM Ninja kur
echo Installing Ninja...
conda install -y Ninja

REM CUDA kur
echo Installing CUDA 12.1...
conda install -y cuda -c nvidia/label/cuda-12.1.0

REM PyTorch kur
echo Installing PyTorch (CUDA 12.1)...
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

REM xformers kur
echo Installing xformers...
pip install xformers==0.0.22.post7

REM Diğer bağımlılıklar
echo Installing other dependencies...
pip install -r requirements.txt

REM Klasörler oluştur
echo Creating directories...
if not exist "ckpts" mkdir ckpts
if not exist "outputs" mkdir outputs
if not exist "examples" mkdir examples

REM CUDA testi
echo.
echo Testing CUDA...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start the application:
echo   conda activate instantmesh
echo   python app_h200_optimized.py
echo.
echo For more information, see H200_SETUP.md
echo.

pause
