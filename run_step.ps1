param(
 [Parameter(Mandatory=$true)][ValidateSet('hashes','tasks','main','hash_thresholds','support_baselines','edit_detection','statistics','residuals','phash_build','phash','sscd_download','sscd_embeddings','sscd_build','sscd','disc21_prepare','disc21','coverage')][string]$Step,
 [switch]$Smoke,
 [string]$Part = '',
 [int]$Workers = 0,
 [string]$RunRoot = ''
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Py)) { throw "환경 없음: $Py  (먼저 setup_env.ps1 실행)" }
$env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'; $env:PYTHONDONTWRITEBYTECODE = '1'; $env:SSCD_DEVICE = 'cuda'
if ($RunRoot) { $RunRoot = [IO.Path]::GetFullPath((Join-Path $Root $RunRoot)); $env:RUN_ROOT = $RunRoot } else { Remove-Item Env:RUN_ROOT -ErrorAction SilentlyContinue }

if ($Workers -le 0) { $Workers = [Math]::Max(1, [int][Math]::Floor([Environment]::ProcessorCount * 0.7)) }
$env:WORKERS = "$Workers"
$env:OMP_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'
try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch { }
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$map = @{ hashes='experiments/compute_hashes.py'; tasks='experiments/build_tasks.py'; main='experiments/train_evaluate.py';
          hash_thresholds='experiments/hash_thresholds.py'; support_baselines='experiments/support_baselines.py'; edit_detection='experiments/edit_detection.py'; statistics='experiments/statistical_analysis.py'; residuals='experiments/residual_analysis.py';
          phash_build='experiments/phash.py'; phash='experiments/phash.py';
          sscd_download='experiments/download_sscd.py'; sscd_embeddings='experiments/embed_sscd.py'; sscd_build='experiments/build_sscd_tasks.py'; sscd='experiments/evaluate_sscd.py';
          disc21_prepare='experiments/prepare_disc21.py'; disc21='experiments/evaluate_disc21.py'; coverage='experiments/coverage_sweep.py' }
$extra = @()
if ($Step -eq 'phash_build') { $extra += @('--part','build') }
if ($Step -eq 'phash')   { $extra += @('--part','run') }
if ($Part) { $extra += @('--part',$Part) }
if ($Smoke) { $extra += '--smoke' }
Push-Location $Root
$ErrorActionPreference = 'Continue'
try {
  & $Py -u -m ("experiments."+[IO.Path]::GetFileNameWithoutExtension($map[$Step])) @extra
  $code = $LASTEXITCODE
} finally { Pop-Location }
if ($code -ne 0) { exit $code }
