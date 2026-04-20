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

$launcher = Join-Path $PSScriptRoot "scripts\tokenize_launcher.ps1"
if (-not (Test-Path $launcher)) {
  Write-Error "Shared launcher not found: $launcher"
  exit 1
}

$invokeArgs = @{
  Mode            = "local"
  Source          = $Source
  OutputRoot      = $OutputRoot
  StartIndex      = 0
  EndIndex        = $EndIndex
  Workers         = $Workers
  OutputShardSize = $OutputShardSize
  MinTokenLength  = $MinTokenLength
  CheckpointEvery = $CheckpointEvery
  ProgressEvery   = $ProgressEvery
  CompressOutput  = $CompressOutput
  StartOver       = $StartOver
}

if ($AllowMixedInstruments) {
  $invokeArgs["AllowMixedInstruments"] = $true
}
if ($DryRun) {
  $invokeArgs["DryRun"] = $true
}

& $launcher @invokeArgs
exit $LASTEXITCODE
