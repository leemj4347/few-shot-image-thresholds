

param([string]$Base = '')
$ErrorActionPreference = 'Stop'
if (-not $Base) {
  $PythonCommand = Get-Command python -CommandType Application -ErrorAction Stop
  $Base = $PythonCommand.Source
}
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $Root '.venv'
$Py = Join-Path $VenvDir 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Base)) { throw "기준 Python 없음: $Base" }
try { nvidia-smi --query-gpu=name,driver_version --format=csv,noheader } catch { Write-Host 'nvidia-smi 확인 실패(드라이버 확인 필요)' -ForegroundColor Yellow }
if (-not (Test-Path -LiteralPath $Py)) { & $Base -m venv $VenvDir }
& $Py -m pip install --upgrade pip
& $Py -m pip install numpy==2.5.3 pandas==3.0.6 scipy==1.18.1 pyarrow==26.0.0 opencv-python==5.0.0.93 pillow==12.3.0 ImageHash==4.3.2 PyWavelets==1.10.0 tqdm==4.70.1
if ($LASTEXITCODE -ne 0) { throw '기본 패키지 설치 실패' }
$ErrorActionPreference = 'Continue'


$check = "import torch; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'available', ok, torch.cuda.get_device_name(0) if ok else ''); raise SystemExit(0 if ok else 3)"
$ok = $false
foreach ($cu in @('cu126','cu128','cu129','cu130')) {
  Write-Host "torch 2.14.1 ($cu) 설치 시도"
  & $Py -m pip uninstall -y torch | Out-Null
  & $Py -m pip install torch==2.14.1 --index-url "https://download.pytorch.org/whl/$cu"
  if ($LASTEXITCODE -ne 0) { Write-Host "  $cu 판 없음" ; continue }
  & $Py -c $check
  if ($LASTEXITCODE -eq 0) { $ok = $true; break }
  Write-Host "  $cu 판은 이 드라이버에서 GPU를 못 씀" -ForegroundColor Yellow
}
if (-not $ok) {
  & $Py -m pip uninstall -y torch | Out-Null
  Write-Host 'CUDA 사용이 가능한 PyTorch를 설치하지 못했습니다. Python 버전, GPU 드라이버와 패키지 설치 오류를 확인하세요.' -ForegroundColor Red
  exit 3
}
Write-Host '실행 환경 준비 완료' -ForegroundColor Green
