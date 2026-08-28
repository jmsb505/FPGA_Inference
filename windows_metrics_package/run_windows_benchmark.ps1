param(
    [string]$RepoRoot = "",
    [string]$CondaEnv = "gen",
    [string]$Devices = "cpu,cuda",
    [string]$PairLimit = "",
    [int]$PowerInferenceRepetitions = 25,
    [int]$PowerPostprocessRepetitions = 1,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($RepoRoot -eq "") {
    $RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
}

$Runner = Join-Path $ScriptDir "run_windows_benchmark.py"
$ArgsList = @(
    "--repo-root", $RepoRoot,
    "--devices", $Devices,
    "--power-inference-repetitions", $PowerInferenceRepetitions,
    "--power-postprocess-repetitions", $PowerPostprocessRepetitions
)

if ($PairLimit -ne "") {
    $ArgsList += @("--pair-limit", $PairLimit)
}
if ($CheckOnly) {
    $ArgsList += @("--check-only")
}

if ($CondaEnv -ne "") {
    conda run -n $CondaEnv python $Runner @ArgsList
} else {
    python $Runner @ArgsList
}
