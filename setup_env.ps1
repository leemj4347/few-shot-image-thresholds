

param([string]$Base = '')
$ErrorActionPreference = 'Stop'
if (-not $Base) {
  $PythonCommand = Get-Command python -CommandType Application -ErrorAction Stop
  $Base = $PythonCommand.Source
}
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $Root '.venv'
$Py = Join-Path $VenvDir 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Base)) { throw "Python executable not found: $Base" }
try { nvidia-smi --query-gpu=name,driver_version --format=csv,noheader } catch { Write-Host 'nvidia-smi failed; check the GPU driver' -ForegroundColor Yellow }
if (-not (Test-Path -LiteralPath $Py)) { & $Base -m venv $VenvDir }
& $Py -m pip install --upgrade pip
& $Py -m pip install numpy==2.5.3 pandas==3.0.6 scipy==1.18.1 pyarrow==26.0.0 opencv-python==5.0.0.93 pillow==12.3.0 ImageHash==4.3.2 PyWavelets==1.10.0 tqdm==4.70.1
if ($LASTEXITCODE -ne 0) { throw 'Package installation failed' }
$ErrorActionPreference = 'Continue'


$check = "import torch; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'available', ok, torch.cuda.get_device_name(0) if ok else ''); raise SystemExit(0 if ok else 3)"
$ok = $false
foreach ($cu in @('cu126','cu128','cu129','cu130')) {
  Write-Host "Installing torch 2.14.1 ($cu)"
  & $Py -m pip uninstall -y torch | Out-Null
  & $Py -m pip install torch==2.14.1 --index-url "https://download.pytorch.org/whl/$cu"
  if ($LASTEXITCODE -ne 0) { Write-Host "  Package unavailable: $cu" ; continue }
  & $Py -c $check
  if ($LASTEXITCODE -eq 0) { $ok = $true; break }
  Write-Host "  CUDA unavailable with this driver: $cu" -ForegroundColor Yellow
}
if (-not $ok) {
  & $Py -m pip uninstall -y torch | Out-Null
  Write-Host 'CUDA-enabled PyTorch installation failed. Check Python, the GPU driver and package errors.' -ForegroundColor Red
  exit 3
}
Write-Host 'Environment ready' -ForegroundColor Green
