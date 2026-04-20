param(
  [ValidateSet("local", "controller")]
  [string]$Mode,

  [string]$Source,
  [string]$OutputRoot,
  [string]$FlashRoot,

  [int]$StartIndex = 0,
  [int]$EndIndex = -1,
  [int]$TargetPieces = 0,
  [int]$ChunkMembers = 50000,
  [int]$Workers = 0,
  [int]$OutputShardSize = 5000,
  [int]$MinTokenLength = 192,
  [int]$CheckpointEvery = 2000,
  [int]$ProgressEvery = 500,
  [int]$PositionsPerBar = 31,

  [string]$RepoId = "",
  [string]$UploadPrefix = "tokenized/chunks",
  [string]$UploadBackend = "large-folder",
  [int]$UploadWorkers = 8,
  [int]$UploadAttempts = 12,
  [int]$UploadBackoffSeconds = 3,
  [int]$MinUploadWorkers = 2,

  [bool]$CompressOutput = $true,
  [bool]$StartOver = $false,
  [bool]$StartFresh = $false,
  [bool]$UploadToHF = $false,
  [bool]$KeepLocal = $true,
  [bool]$EnableHfTransfer = $false,
  [bool]$PromptOnFailure = $false,

  [string]$HfToken = "",
  [switch]$AllowMixedInstruments,
  [switch]$IncludeStructuralPrefix,
  [switch]$DryRun
)

function Resolve-PythonExe {
  param([string]$RepoRoot)

  $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
  if (Test-Path $venvPython) {
    return $venvPython
  }
  return "python"
}

function Ensure-HfTransfer {
  param([string]$PythonExe)

  & $PythonExe -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('hf_transfer') else 1)"
  if ($LASTEXITCODE -ne 0) {
    & $PythonExe -m pip install -q hf_transfer
  }
}

function Ensure-SourceExists {
  param([string]$PathValue)

  if (-not (Test-Path $PathValue)) {
    Write-Error "Source path does not exist: $PathValue"
    exit 1
  }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not (Test-Path $repoRoot)) {
  Write-Error "Could not resolve repository root."
  exit 1
}

$modeNormalized = "$(($Mode | ForEach-Object { $_.Trim().ToLowerInvariant() }))"
if ($modeNormalized -notin @("local", "controller")) {
  Write-Error "Mode must be 'local' or 'controller'."
  exit 1
}

if ([string]::IsNullOrWhiteSpace($Source)) {
  Write-Error "Source must be provided."
  exit 1
}
Ensure-SourceExists -PathValue $Source

$pythonExe = Resolve-PythonExe -RepoRoot $repoRoot

if (-not [string]::IsNullOrWhiteSpace($HfToken)) {
  $env:HF_TOKEN = $HfToken
}

if ($EnableHfTransfer) {
  $env:HF_HUB_ENABLE_HF_TRANSFER = "1"
  Ensure-HfTransfer -PythonExe $pythonExe
}

if ($modeNormalized -eq "local") {
  if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    Write-Error "OutputRoot must be provided for local mode."
    exit 1
  }

  $tokenizeScript = Join-Path $repoRoot "scripts\tokenize_godzilla_local.py"
  if (-not (Test-Path $tokenizeScript)) {
    Write-Error "Tokenizer script not found: $tokenizeScript"
    exit 1
  }

  $effectiveEndIndex = if ($EndIndex -gt 0) {
    [int]$EndIndex
  } elseif ($TargetPieces -gt 0) {
    [int]($StartIndex + $TargetPieces)
  } else {
    0
  }
  if ($effectiveEndIndex -le $StartIndex) {
    Write-Error "EndIndex must be greater than StartIndex (or provide TargetPieces > 0)."
    exit 1
  }

  $args = @(
    $tokenizeScript,
    "--source", $Source,
    "--output-root", $OutputRoot,
    "--start-index", "$StartIndex",
    "--end-index", "$effectiveEndIndex",
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
  if ($IncludeStructuralPrefix) {
    $args += "--include-structural-prefix"
  }

  Write-Output "Local tokenization run"
  Write-Output "  source: $Source"
  Write-Output "  output_root: $OutputRoot"
  Write-Output "  start_index: $StartIndex"
  Write-Output "  end_index: $effectiveEndIndex"
  Write-Output "  workers: $Workers"
  Write-Output "  output_shard_size: $OutputShardSize"
  Write-Output "  include_structural_prefix: $($IncludeStructuralPrefix.IsPresent)"
  Write-Output "  python: $pythonExe"

  if ($DryRun) {
    Write-Output "Dry run only. Command would be:"
    Write-Output "& $pythonExe $($args -join ' ')"
    exit 0
  }

  & $pythonExe @args
  if ($LASTEXITCODE -ne 0) {
    if ($PromptOnFailure) {
      Read-Host "Tokenization failed. Press Enter to close"
    }
    exit $LASTEXITCODE
  }

  Write-Output "Done. Local tokenization completed."
  exit 0
}

if ([string]::IsNullOrWhiteSpace($FlashRoot)) {
  Write-Error "FlashRoot must be provided for controller mode."
  exit 1
}

$controllerScript = Join-Path $repoRoot "scripts\tokenize_upload_hf_batches.py"
if (-not (Test-Path $controllerScript)) {
  Write-Error "Controller script not found: $controllerScript"
  exit 1
}

$effectiveEndIndex = if ($EndIndex -gt 0) {
  [int]$EndIndex
} elseif ($TargetPieces -gt 0) {
  [int]($StartIndex + $TargetPieces)
} else {
  -1
}
if (($effectiveEndIndex -ge 0) -and ($effectiveEndIndex -le $StartIndex)) {
  Write-Error "EndIndex must be greater than StartIndex (or provide TargetPieces > 0)."
  exit 1
}

$workPath = Join-Path $FlashRoot "work"
$controllerPath = Join-Path $FlashRoot "_controller"
if ($StartFresh) {
  if (Test-Path $workPath) {
    Remove-Item $workPath -Recurse -Force -ErrorAction SilentlyContinue
  }
  if (Test-Path $controllerPath) {
    Remove-Item $controllerPath -Recurse -Force -ErrorAction SilentlyContinue
  }
}

$args = @(
  $controllerScript,
  "--source", $Source,
  "--flash-root", $FlashRoot,
  "--start-index", "$StartIndex",
  "--end-index", "$effectiveEndIndex",
  "--chunk-members", "$ChunkMembers",
  "--workers", "$Workers",
  "--output-shard-size", "$OutputShardSize",
  "--min-token-length", "$MinTokenLength",
  "--checkpoint-every", "$CheckpointEvery",
  "--progress-every", "$ProgressEvery",
  "--positions-per-bar", "$PositionsPerBar"
)

if ($IncludeStructuralPrefix) {
  $args += "--include-structural-prefix"
}
if ($CompressOutput) {
  $args += "--compress-output"
}
if ($AllowMixedInstruments) {
  $args += "--allow-mixed-instruments"
}
if ($StartFresh) {
  $args += "--reset-state"
}

if ($UploadToHF) {
  if ([string]::IsNullOrWhiteSpace($RepoId)) {
    Write-Error "RepoId is required when UploadToHF is true."
    exit 1
  }

  $args += @(
    "--repo-id", $RepoId,
    "--upload-prefix", $UploadPrefix,
    "--upload-backend", $UploadBackend,
    "--upload-workers", "$UploadWorkers",
    "--upload-attempts", "$UploadAttempts",
    "--upload-backoff-seconds", "$UploadBackoffSeconds",
    "--min-upload-workers", "$MinUploadWorkers"
  )

  if ($KeepLocal) {
    $args += "--keep-local"
  }
} else {
  $args += "--skip-upload"
  if ($KeepLocal) {
    $args += "--keep-local"
  }
}

Write-Output "Controller tokenization run"
Write-Output "  source: $Source"
Write-Output "  flash_root: $FlashRoot"
Write-Output "  start_index: $StartIndex"
Write-Output "  end_index: $effectiveEndIndex"
Write-Output "  target_pieces: $TargetPieces"
Write-Output "  chunk_members: $ChunkMembers"
Write-Output "  workers: $Workers"
Write-Output "  output_shard_size: $OutputShardSize"
Write-Output "  positions_per_bar: $PositionsPerBar"
Write-Output "  include_structural_prefix: $($IncludeStructuralPrefix.IsPresent)"
Write-Output "  upload_to_hf: $UploadToHF"
Write-Output "  repo_id: $RepoId"
Write-Output "  compress_output: $CompressOutput"
Write-Output "  start_fresh: $StartFresh"
Write-Output "  python: $pythonExe"

if ($DryRun) {
  Write-Output "Dry run only. Command would be:"
  Write-Output "& $pythonExe $($args -join ' ')"
  exit 0
}

& $pythonExe @args
if ($LASTEXITCODE -ne 0) {
  if ($PromptOnFailure) {
    Read-Host "Tokenization failed. Press Enter to close"
  }
  exit $LASTEXITCODE
}

Write-Output "Done. Controller tokenization completed."