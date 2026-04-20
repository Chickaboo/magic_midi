param(
  [string]$Source = "E:\ai_models\data\Godzilla-Piano-MIDI-Dataset-CC-BY-NC-SA\MIDIs",
  [string]$FlashRoot = "C:\pulse88_tokenize_500k_work",
  [int]$StartIndex = 0,
  [int]$TargetPieces = 500000,
  [int]$ChunkMembers = 50000,
  [int]$Workers = 0,
  [int]$OutputShardSize = 5000,
  [int]$PositionsPerBar = 31,
  [int]$MinTokenLength = 192,
  [int]$CheckpointEvery = 2000,
  [int]$ProgressEvery = 500,
  [bool]$CompressOutput = $true,
  [switch]$StartFresh,
  [bool]$UploadToHF = $false,
  [int]$RetryDelaySeconds = 30,
  [int]$MaxAttempts = 0,
  [string]$CombinedManifestName = "manifest_500k.json",
  [switch]$AllowMixedInstruments,
  [switch]$DryRun
)

$launcher = Join-Path $PSScriptRoot "scripts\tokenize_launcher.ps1"
if (-not (Test-Path $launcher)) {
  Write-Error "Shared launcher not found: $launcher"
  exit 1
}

function Get-NextIndex {
  param([string]$Root)

  try {
    $statePath = Join-Path $Root "_controller\state.json"
    if (-not (Test-Path $statePath)) {
      return -1
    }
    $state = Get-Content -Raw -Path $statePath | ConvertFrom-Json
    return [int]$state.next_index
  } catch {
    return -1
  }
}

function Write-CombinedManifest {
  param(
    [string]$Root,
    [string]$ManifestFileName
  )

  $workDir = Join-Path $Root "work"
  $controllerDir = Join-Path $Root "_controller"
  if (-not (Test-Path $controllerDir)) {
    New-Item -ItemType Directory -Path $controllerDir -Force | Out-Null
  }

  $outPath = Join-Path $controllerDir $ManifestFileName
  $allRows = @()

  if (Test-Path $workDir) {
    $chunkDirs = Get-ChildItem -Path $workDir -Directory -Filter "chunk_*" | Sort-Object Name
    foreach ($chunk in $chunkDirs) {
      $chunkManifest = Join-Path $chunk.FullName "metadata\manifest.json"
      if (-not (Test-Path $chunkManifest)) {
        continue
      }

      try {
        $rows = Get-Content -Raw -Path $chunkManifest | ConvertFrom-Json
      } catch {
        Write-Warning "Failed to read chunk manifest: $chunkManifest"
        continue
      }

      if ($null -eq $rows) {
        continue
      }

      if ($rows -is [System.Array]) {
        $allRows += $rows
      } else {
        $allRows += @($rows)
      }
    }
  }

  $sortedRows = $allRows | Sort-Object { [int]$_.index }
  $sortedRows | ConvertTo-Json -Depth 6 | Set-Content -Path $outPath -Encoding UTF8

  return @{
    Path  = $outPath
    Count = @($sortedRows).Count
  }
}

$attempt = 0
while ($true) {
  $attempt += 1
  $startFreshThisAttempt = ($attempt -eq 1) -and [bool]$StartFresh.IsPresent

  $invokeArgs = @{
    Mode                    = "controller"
    Source                  = $Source
    FlashRoot               = $FlashRoot
    StartIndex              = $StartIndex
    TargetPieces            = $TargetPieces
    ChunkMembers            = $ChunkMembers
    Workers                 = $Workers
    OutputShardSize         = $OutputShardSize
    MinTokenLength          = $MinTokenLength
    CheckpointEvery         = $CheckpointEvery
    ProgressEvery           = $ProgressEvery
    PositionsPerBar         = $PositionsPerBar
    CompressOutput          = $CompressOutput
    StartFresh              = $startFreshThisAttempt
    UploadToHF              = $UploadToHF
    KeepLocal               = $true
    IncludeStructuralPrefix = $true
    PromptOnFailure         = $false
  }

  if ($AllowMixedInstruments) {
    $invokeArgs["AllowMixedInstruments"] = $true
  }
  if ($DryRun) {
    $invokeArgs["DryRun"] = $true
  }

  Write-Output ("[{0}] Tokenization attempt {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $attempt)
  & $launcher @invokeArgs
  $exitCode = $LASTEXITCODE

  if ($exitCode -eq 0) {
    if (-not $DryRun) {
      $manifestInfo = Write-CombinedManifest -Root $FlashRoot -ManifestFileName $CombinedManifestName
      Write-Output ("Combined manifest written: {0}" -f $manifestInfo.Path)
      Write-Output ("Combined manifest entries: {0}" -f $manifestInfo.Count)
    }
    Write-Output "Tokenization controller completed successfully."
    exit 0
  }

  if ($DryRun) {
    exit $exitCode
  }

  $nextIndex = Get-NextIndex -Root $FlashRoot
  if ($nextIndex -ge 0) {
    Write-Warning "Controller failed with exit code $exitCode. Resume index is $nextIndex. Retrying from saved state..."
  } else {
    Write-Warning "Controller failed with exit code $exitCode. No state index found yet. Retrying..."
  }

  if ([int]$MaxAttempts -gt 0 -and $attempt -ge [int]$MaxAttempts) {
    Write-Error "Reached MaxAttempts=$MaxAttempts. Stopping."
    exit $exitCode
  }

  $delay = [int]([Math]::Max(1, $RetryDelaySeconds))
  Write-Output "Waiting $delay seconds before retry..."
  Start-Sleep -Seconds $delay
}
