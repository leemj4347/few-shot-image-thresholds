param([string]$Run='', [string]$Data='')
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $MyInvocation.MyCommand.Path
if(-not $Run){$Run='runs/numerical_'+(Get-Date -Format 'yyyyMMdd_HHmmss')}
$Out=if([IO.Path]::IsPathRooted($Run)){[IO.Path]::GetFullPath($Run)}else{[IO.Path]::GetFullPath((Join-Path $Root $Run))}
$Py=Join-Path $Root '.venv/Scripts/python.exe'
$env:PYTHONDONTWRITEBYTECODE='1';$env:PYTHONIOENCODING='utf-8';$env:PYTHONUNBUFFERED='1'
$CallArgs=@((Join-Path $Root 'analysis/run_analysis.py'),'--output',$Out,'--experiment-python',$Py)
if ($Data) { $CallArgs += @('--data', [IO.Path]::GetFullPath($Data)) }
& $Py @CallArgs
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
