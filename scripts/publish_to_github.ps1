param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteUrl,

    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if (-not (Test-Path ".git")) {
    throw "This directory is not a git repository: $repoRoot"
}

$dirty = git status --short
if ($dirty) {
    throw "Working tree is not clean. Commit or discard changes before publishing.`n$dirty"
}

$currentBranch = git branch --show-current
if ($currentBranch -ne $Branch) {
    git branch -M $Branch
}

$existingRemote = git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0) {
    git remote set-url origin $RemoteUrl
} else {
    git remote add origin $RemoteUrl
}

git push -u origin $Branch
