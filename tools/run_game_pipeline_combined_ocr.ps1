param(
    [Parameter(Mandatory=$true)]
    [string]$Video,

    [string]$RunName = "",
    [string]$OutputRoot = "runs\pipeline",

    [string]$VenvPython = ".\venv\Scripts\python.exe",
    [string]$PaddlePython = ".\paddle_venv\Scripts\python.exe",

    [string]$PaddleModelDir = "runs\jersey_ocr_paddle\inference",
    [string]$PaddleCharDict = "datasets\jersey_ocr_paddle\digit_dict.txt",

    [string]$Device = "0",
    [switch]$NoPaddle,
    [switch]$NoDebugPossession,
    [switch]$FixOpenCV,
    [switch]$ReuseCalibration,
    [switch]$SkipCalibration
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

function Resolve-ExistingPath([string]$PathText, [string]$Label) {
    if (!(Test-Path $PathText)) {
        throw "$Label not found: $PathText"
    }
    return (Resolve-Path $PathText).Path
}

function Get-VideoStem([string]$VideoPath) {
    return [System.IO.Path]::GetFileNameWithoutExtension($VideoPath)
}

function Get-LatestBestPt([string[]]$Roots) {
    $files = @()
    foreach ($root in $Roots) {
        if (Test-Path $root) {
            $files += Get-ChildItem -Path $root -Filter best.pt -Recurse -ErrorAction SilentlyContinue
        }
    }
    if ($files.Count -eq 0) { return $null }
    return ($files | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}

function Test-OpenCVGui([string]$PythonExe) {
    # Do not pass multiline Python through -c; PowerShell can strip quotes from
    # variable-expanded native args. Write a tiny temporary script instead.
    $tmp = Join-Path $env:TEMP ("cv_gui_check_" + [System.Guid]::NewGuid().ToString("N") + ".py")
    $code = @'
import cv2
info = cv2.getBuildInformation()
no_gui_markers = (
    "GUI:                           NONE",
    "GUI:                            NONE",
    "GUI: NONE",
)
print("NO_GUI" if any(marker in info for marker in no_gui_markers) else "GUI_OK")
'@
    try {
        Set-Content -Path $tmp -Value $code -Encoding UTF8
        $result = & $PythonExe $tmp 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host ($result -join "`n") -ForegroundColor Yellow
            return $false
        }
        return (($result -join "`n") -match "GUI_OK")
    }
    finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
}

function Repair-OpenCV([string]$PythonExe) {
    Write-Host "[preflight] Repairing OpenCV GUI package in selected Python env..." -ForegroundColor Yellow
    & $PythonExe -m pip uninstall -y opencv-python-headless opencv-contrib-python-headless | Out-Host
    & $PythonExe -m pip install --upgrade --force-reinstall opencv-python opencv-contrib-python | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "OpenCV reinstall failed." }
}

$VenvPython = Resolve-ExistingPath $VenvPython "Main/unified venv python"
$Video = Resolve-ExistingPath $Video "Video"

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "$(Get-VideoStem $Video)_game"
}

$RunDir = Join-Path (Join-Path $Root $OutputRoot) $RunName
$CalibrationPath = Join-Path $RunDir "court_calibration.json"

if (!$SkipCalibration -and !(Test-Path $CalibrationPath)) {
    if (!(Test-OpenCVGui $VenvPython)) {
        if ($FixOpenCV) {
            Repair-OpenCV $VenvPython
            if (!(Test-OpenCVGui $VenvPython)) {
                throw "OpenCV still has no GUI support after reinstall. Close/reopen PowerShell and retry."
            }
        }
        else {
            throw @"
OpenCV in the selected Python env has no GUI/highgui support, so court calibration cannot open the click window.

Selected Python:
  $VenvPython

Run:

  .\tools\run_game_pipeline_combined_ocr.ps1 -Video "$Video" -RunName "$RunName" -VenvPython "$VenvPython" -PaddlePython "$PaddlePython" -FixOpenCV

or manually:

  $VenvPython -m pip uninstall -y opencv-python-headless opencv-contrib-python-headless
  $VenvPython -m pip install --upgrade --force-reinstall opencv-python opencv-contrib-python

If you already have a valid calibration at:
  $CalibrationPath
run with:
  -ReuseCalibration
"@
        }
    }
}

$DebugArgs = @()
if (!$NoDebugPossession) { $DebugArgs += "--debug-possession" }

$CalibrationArgs = @()
if ($ReuseCalibration) { $CalibrationArgs += "--reuse-calibration" }
if ($SkipCalibration) { $CalibrationArgs += "--skip-calibration" }

Write-Host "`n=== STEP 1/4: base game pipeline + EasyOCR ===" -ForegroundColor Cyan
& $VenvPython "tools\run_game_pipeline.py" `
    --video $Video `
    --output-root $OutputRoot `
    --run-name $RunName `
    --ball-model auto `
    --role-model auto `
    --possession-model auto `
    --with-ocr `
    --ocr-device cuda `
    --save-crops `
    --device $Device `
    @CalibrationArgs `
    @DebugArgs
if ($LASTEXITCODE -ne 0) { throw "Base game pipeline failed." }

$EasyReport = Join-Path $RunDir "jerseys\jersey_numbers.json"
$PaddleReport = Join-Path $RunDir "jerseys_paddle_v3\jersey_numbers.json"
$CombinedReport = Join-Path $RunDir "jerseys_combined\jersey_numbers.json"
$FinalCombinedVideo = Join-Path $RunDir "final_combined_ocr.mp4"

if ($NoPaddle) {
    Write-Host "`n[info] --NoPaddle set. Final output is EasyOCR render:" -ForegroundColor Yellow
    Write-Host (Join-Path $RunDir "final_game_pipeline.mp4")
    exit 0
}

if (!(Test-Path $PaddlePython)) {
    Write-Host "`n[warn] Paddle python not found. Skipping adapted OCR:" -ForegroundColor Yellow
    Write-Host "       $PaddlePython"
    Write-Host "Final output is EasyOCR render: $(Join-Path $RunDir 'final_game_pipeline.mp4')"
    exit 0
}

if (!(Test-Path $PaddleModelDir) -or !(Test-Path $PaddleCharDict)) {
    Write-Host "`n[warn] Paddle OCR model/dict not found. Skipping adapted OCR." -ForegroundColor Yellow
    Write-Host "       model: $PaddleModelDir"
    Write-Host "       dict : $PaddleCharDict"
    Write-Host "Final output is EasyOCR render: $(Join-Path $RunDir 'final_game_pipeline.mp4')"
    exit 0
}

Write-Host "`n=== STEP 2/4: adapted PaddleOCR jersey extraction ===" -ForegroundColor Cyan
& $PaddlePython "tools\extract_jersey_numbers_paddle_v3.py" `
    --video $Video `
    --tracks (Join-Path $RunDir "tracks.json") `
    --player-summary (Join-Path $RunDir "player_summary.json") `
    --output-dir (Join-Path $RunDir "jerseys_paddle_v3") `
    --ocr-engine paddle `
    --paddle-rec-model-dir $PaddleModelDir `
    --paddle-rec-char-dict $PaddleCharDict `
    --paddle-min-confidence 0.20 `
    --max-crops-per-player 80 `
    --sample-step 5 `
    --save-crops
if ($LASTEXITCODE -ne 0) { throw "Adapted PaddleOCR extraction failed." }

if (!(Test-Path $PaddleReport)) {
    throw "Paddle report was not generated: $PaddleReport"
}

Write-Host "`n=== STEP 3/4: combine EasyOCR + PaddleOCR reports ===" -ForegroundColor Cyan
& $VenvPython "tools\combine_jersey_ocr_reports.py" `
    --easyocr $EasyReport `
    --paddle $PaddleReport `
    --output $CombinedReport
if ($LASTEXITCODE -ne 0) { throw "OCR report combination failed." }

Write-Host "`n=== STEP 4/4: final render with combined OCR ===" -ForegroundColor Cyan
$RoleModel = Get-LatestBestPt @("runs\classify\runs\person_roles", "runs\person_roles", "runs\classify\person_roles")
$PossessionModel = Get-LatestBestPt @("runs\classify\runs\possession_cls", "runs\possession_cls", "runs\classify\possession_cls")

$RenderArgs = @(
    "tools\render_possession_with_model.py",
    "--video", $Video,
    "--tracks", (Join-Path $RunDir "tracks.json"),
    "--calibration", (Join-Path $RunDir "court_calibration.json"),
    "--events", (Join-Path $RunDir "events.json"),
    "--team-calibration", (Join-Path $RunDir "team_calibration.json"),
    "--jersey-numbers", $CombinedReport,
    "--output", $FinalCombinedVideo,
    "--output-possession", (Join-Path $RunDir "possession_timeline_combined_ocr.json")
)
if ($RoleModel) { $RenderArgs += @("--role-model", $RoleModel) }
if ($PossessionModel) { $RenderArgs += @("--possession-model", $PossessionModel) }
if (!$NoDebugPossession) { $RenderArgs += "--debug-possession" }

& $VenvPython @RenderArgs
if ($LASTEXITCODE -ne 0) { throw "Final combined render failed." }

Write-Host "`nDONE" -ForegroundColor Green
Write-Host "Run folder: $RunDir"
Write-Host "Final video: $FinalCombinedVideo"
Write-Host "Combined OCR: $CombinedReport"
