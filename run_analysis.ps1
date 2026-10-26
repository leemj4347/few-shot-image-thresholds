param([string]$Run='')
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $MyInvocation.MyCommand.Path
if(-not $Run){$Run='runs/numerical_'+(Get-Date -Format 'yyyyMMdd_HHmmss')}
$Out=if([IO.Path]::IsPathRooted($Run)){[IO.Path]::GetFullPath($Run)}else{[IO.Path]::GetFullPath((Join-Path $Root $Run))}
$Allowed=[IO.Path]::GetFullPath((Join-Path $Root 'runs'))+[IO.Path]::DirectorySeparatorChar
if(-not $Out.StartsWith($Allowed,[StringComparison]::OrdinalIgnoreCase)){throw 'Output must be inside experiment/runs'}
$Py=Join-Path $Root '.venv/Scripts/python.exe'
$env:PYTHONDONTWRITEBYTECODE='1';$env:PYTHONIOENCODING='utf-8';$env:PYTHONUNBUFFERED='1'
$CallArgs=@((Join-Path $Root 'analysis/run_analysis.py'),'--output',$Out,'--experiment-python',$Py)
& $Py @CallArgs
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
