param(
 [string]$From = '',
 [int]$Workers = 0
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$order = @('hashes','tasks','main','hash_thresholds','support_baselines','edit_detection','statistics','residuals','calibration_size','data_overlap','phash_build','phash','sscd_download','sscd_embeddings','sscd_build','sscd','disc21_prepare','disc21','coverage')
if ($From) { $i = [Array]::IndexOf($order, $From); if ($i -lt 0) { throw "알 수 없는 단계: $From" }; $order = $order[$i..($order.Count-1)] }
foreach ($s in $order) {
  & (Join-Path $Root 'run_step.ps1') -Step $s -Workers $Workers
  if ($LASTEXITCODE -ne 0) { Write-Host "중단: $s 본 실행 실패" -ForegroundColor Red; exit $LASTEXITCODE }
}


& (Join-Path $Root "run_analysis.ps1")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

