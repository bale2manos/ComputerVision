param(
    [string]$Python = "py -3.10",
    [string]$VenvDir = ".venv-cv",
    [switch]$CpuTorch
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

Write-Host "Creating unified environment at $VenvDir" -ForegroundColor Cyan
if (Test-Path $VenvDir) {
    Write-Host "Removing existing $VenvDir" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $VenvDir
}

$pythonParts = $Python.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
$pythonExe = $pythonParts[0]
$pythonArgs = @()
if ($pythonParts.Length -gt 1) { $pythonArgs = $pythonParts[1..($pythonParts.Length - 1)] }
& $pythonExe @pythonArgs -m venv $VenvDir

$VenvPython = Join-Path $Root "$VenvDir\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip setuptools wheel

Write-Host "Installing PyTorch" -ForegroundColor Cyan
if ($CpuTorch) {
    & $VenvPython -m pip install torch torchvision torchaudio
}
else {
    & $VenvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
}

Write-Host "Installing unified requirements" -ForegroundColor Cyan
& $VenvPython -m pip install -r requirements-unified.txt

if (Test-Path "external\PaddleOCR") {
    Write-Host "Installing local PaddleOCR clone editable" -ForegroundColor Cyan
    & $VenvPython -m pip install -e external\PaddleOCR
}
else {
    Write-Host "external\PaddleOCR not found; installing paddleocr package from pip" -ForegroundColor Yellow
    & $VenvPython -m pip install paddleocr
}

Write-Host "Checking imports" -ForegroundColor Cyan
& $VenvPython -c "import sys; print(sys.executable); import cv2; print('cv2', cv2.__version__); import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); import ultralytics; print('ultralytics ok'); import easyocr; print('easyocr ok'); import paddle; print('paddle', paddle.__version__); from tools.infer.predict_rec import TextRecognizer; print('paddleocr TextRecognizer ok')"

Write-Host "`nDONE" -ForegroundColor Green
Write-Host "Activate with:"
Write-Host "  .\$VenvDir\Scripts\activate"
Write-Host "Use python: $VenvPython"
