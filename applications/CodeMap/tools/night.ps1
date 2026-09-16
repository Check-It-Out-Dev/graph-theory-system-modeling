# The night, end to end, on the owner's box (PowerShell 7). Registered once with:
#   schtasks /Create /TN "CodeMap night" /SC DAILY /ST 03:00 /TR "pwsh -NoProfile -File C:\Users\Norbert\IdeaProjects\graph-theory-system-modeling\applications\CodeMap\tools\night.ps1"
# Steps: personas (Haiku first, caps) → judge → calibration → quality (+ Grafana push) → commit the
# artifacts on the working branch → push (so nightly.yml rebuilds the public quality page).
# Every model call is `claude -p` on the subscription; ANTHROPIC_API_KEY is never set here.
param(
    [string]$Date = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd"),
    [switch]$NoPush,
    [switch]$HaikuOnly,
    [int]$MaxCredits = 3000,
    [double]$BaselineShare = 0.5,
    [int]$BankPass = 24
)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
$R = Split-Path -Parent $PSScriptRoot           # applications/CodeMap
$Repo = Split-Path -Parent (Split-Path -Parent $R)
Set-Location $R

function Load-Env($path) {
    if (-not (Test-Path $path)) { throw "missing $path" }
    Get-Content $path | Where-Object { $_ -match '^\s*[A-Za-z_][A-Za-z0-9_]*=' } | ForEach-Object {
        $k, $v = $_ -split '=', 2
        [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim().Trim('"'), "Process")
    }
}
Load-Env "$HOME\.codemap-remote.env"
Load-Env "$HOME\.grafana-cloud.env"

$log = "eval\humans\runs\$Date.night.log"
New-Item -ItemType Directory -Force -Path "eval\humans\runs", "eval\judge\runs", "eval\quality\runs" | Out-Null
"[$(Get-Date -Format s)] night $Date starts" | Tee-Object -FilePath $log -Append

# 1. the personas (the runner syncs the miss backlog and refuses to overrun the caps)
$args = @("eval\humans\run_night.py", "--date", $Date, "--max-credits", "$MaxCredits", "--baseline-share", "$BaselineShare")
if ($HaikuOnly) { $args += "--haiku-only" }
python @args 2>&1 | Tee-Object -FilePath $log -Append

# 1b. the bank pass: exact bank questions and probes as the judge user, so the night's own κ rests on enough oracle rows
if ($BankPass -gt 0) {
    python eval\judge\bank_pass.py --url $env:CODEMAP_URL --token $env:CODEMAP_TOKEN --user judge --n $BankPass --probes 6 --seed ([int]$Date.Replace("-", "") % 1000) 2>&1 | Tee-Object -FilePath $log -Append
}

# 2. the night's events from the VPS (the server's artifact of record), then the judge
$hdr = @{ Authorization = "Bearer $env:CODEMAP_TOKEN"; "X-CodeMap-Admin" = $env:CODEMAP_ADMIN_TOKEN }
# the night label may sit ahead of the clock: the events window starts where the personas started
$since = "${Date}T00:00:00Z"
$summaryPath = "eval\humans\runs\$Date.summary.json"
if (Test-Path $summaryPath) { $started = (Get-Content $summaryPath -Raw | ConvertFrom-Json).started; if ($started) { $since = $started } }
$ev = Invoke-RestMethod -Uri "$env:CODEMAP_URL/admin/events?since=$since&limit=5000" -Headers $hdr
$ev.events | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 12 } | Set-Content -Encoding utf8 "eval\judge\runs\events-$Date.jsonl"
"events fetched: $($ev.count)" | Tee-Object -FilePath $log -Append
python eval\judge\judge.py --events "eval\judge\runs\events-$Date.jsonl" --out "eval\judge\runs\$Date.json" --backend claude --modal 2>&1 | Tee-Object -FilePath $log -Append
python eval\judge\calibrate.py --run "eval\judge\runs\$Date.json" 2>&1 | Tee-Object -FilePath $log -Append

# 3. quality: one artifact per night, pushed as codemap_quality_* gauges
python eval\quality\quality.py --date $Date --events "eval\judge\runs\events-$Date.jsonl" --judge "eval\judge\runs\$Date.json" `
    --humans "eval\humans\runs\$Date.jsonl" --out "eval\quality\runs\$Date.json" --push 2>&1 | Tee-Object -FilePath $log -Append

# 3b. the pair campaign across nights (the README's gain condition counts pairs cumulatively)
python eval\quality\campaign.py 2>&1 | Tee-Object -FilePath $log -Append

# 4. the artifacts become history (the public quality page reads them)
git -C $Repo add "applications/CodeMap/eval/humans/runs" "applications/CodeMap/eval/judge/runs" "applications/CodeMap/eval/quality/runs" "applications/CodeMap/graph/delta/backlog.jsonl" 2>$null
$msg = "Night ${Date}: personas, judge, quality"
git -C $Repo commit -q -m $msg
if (-not $NoPush) { git -C $Repo push -q origin HEAD }
"[$(Get-Date -Format s)] night $Date done" | Tee-Object -FilePath $log -Append
