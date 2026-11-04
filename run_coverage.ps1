param(
 [switch]$Smoke,
 [string]$Part = 'all'
)


$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Py)) { throw "Python environment missing: $Py" }
if (-not (Test-Path -LiteralPath (Join-Path $Root 'experiments/coverage_sweep.py'))) { throw "coverage_sweep.py not found" }
$env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'; $env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'
try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch { }
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$extra = @('--part', $Part)
if ($Smoke) { $extra += '--smoke' }
Push-Location $Root
$ErrorActionPreference = 'Continue'
try {
  & $Py -u -m experiments.coverage_sweep @extra
  $code = $LASTEXITCODE
} finally { Pop-Location }
if ($code -ne 0) { exit $code }
