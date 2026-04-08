param(
  [string]$Source = "E:\ai_models\data\Godzilla-Piano-MIDI-Dataset-CC-BY-NC-SA\MIDIs",
  [string]$OutputRoot = "C:\pulse88_tokenize_100k_local",
  [int]$EndIndex = 100000,
  [int]$Workers = 0,
  [int]$OutputShardSize = 100000,
  [int]$MinTokenLength = 192,
  [int]$CheckpointEvery = 2000,
  [int]$ProgressEvery = 500,
  [bool]$CompressOutput = $true,
  [bool]$StartOver = $false,
  [switch]$AllowMixedInstruments,
  [switch]$DryRun
)

$repoRoot = $PSScriptRoot
if (-not (Test-Path $repoRoot)) {
  Write-Error "Could not resolve script directory."
  exit 1
}

if (-not (Test-Path $Source)) {
  Write-Error "Source path does not exist: $Source"
  exit 1
}

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
  $pythonExe = $venvPython
} else {
  $pythonExe = "python"
}

$tokenizeScript = Join-Path $repoRoot "scripts\tokenize_godzilla_local.py"
if (-not (Test-Path $tokenizeScript)) {
  Write-Error "Tokenizer script not found: $tokenizeScript"
  exit 1
}

# This keeps all files in a single shard folder: data/00000.
# Using --end-index instead of --max-files keeps the tokenizer in parallel mode.
$args = @(
  $tokenizeScript,
  "--source", $Source,
  "--output-root", $OutputRoot,
  "--start-index", "0",
  "--end-index", "$EndIndex",
  "--workers", "$Workers",
  "--output-shard-size", "$OutputShardSize",
  "--min-token-length", "$MinTokenLength",
  "--checkpoint-every", "$CheckpointEvery",
  "--progress-every", "$ProgressEvery"
)

if ($CompressOutput) {
  $args += "--compress-output"
}
if ($StartOver) {
  $args += "--start-over"
}
if ($AllowMixedInstruments) {
  $args += "--allow-mixed-instruments"
}

Write-Output "Local-only tokenization run"
Write-Output "  source: $Source"
Write-Output "  output_root: $OutputRoot"
Write-Output "  end_index: $EndIndex"
Write-Output "  workers: $Workers"
Write-Output "  output_shard_size: $OutputShardSize"
Write-Output "  compress_output: $CompressOutput"
Write-Output "  start_over: $StartOver"
Write-Output "  python: $pythonExe"

if ($DryRun) {
  Write-Output "Dry run only. Command would be:"
  Write-Output "& $pythonExe $($args -join ' ')"
  exit 0
}

& $pythonExe @args
if ($LASTEXITCODE -ne 0) {
  Read-Host "Tokenization failed. Press Enter to close"
  exit $LASTEXITCODE
}

Write-Output "Done. Local tokenization completed."
