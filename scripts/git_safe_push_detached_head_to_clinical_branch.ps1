$ErrorActionPreference = "Stop"

$RemoteName = "origin"
$TargetBranch = "clinical-knowledge-architecture"
$RemoteRef = "$RemoteName/$TargetBranch"
$ExpectedRemoteFragment = "courseeast-jpg/MedAI"

function Run-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Args
    )
    & git @Args
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Args -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Git-Output {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Args
    )
    $output = & git @Args
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Args -join ' ') failed with exit code $LASTEXITCODE"
    }
    return (($output | Out-String).Trim())
}

$status = Git-Output @("status", "--short")
$localHead = Git-Output @("rev-parse", "HEAD")
$remoteUrls = Git-Output @("remote", "get-url", "--all", $RemoteName)

Run-Git @("fetch", $RemoteName)
$remoteHeadBefore = Git-Output @("rev-parse", $RemoteRef)

Write-Host "current HEAD: $localHead"
Write-Host "$RemoteRef HEAD before push: $remoteHeadBefore"
Write-Host "git status --short:"
if ($status) {
    Write-Host $status
} else {
    Write-Host "<clean>"
}

if ($status) {
    throw "Refusing to push: git status is not clean."
}

if ($remoteUrls -notlike "*$ExpectedRemoteFragment*") {
    throw "Refusing to push: remote '$RemoteName' does not contain $ExpectedRemoteFragment."
}

& git merge-base --is-ancestor $RemoteRef HEAD
if ($LASTEXITCODE -ne 0) {
    throw "Refusing to push: $RemoteRef is not an ancestor of HEAD."
}

if ($localHead -eq $remoteHeadBefore) {
    throw "already pushed"
}

Run-Git @("push", $RemoteName, "HEAD:$TargetBranch")
Run-Git @("fetch", $RemoteName)
$remoteHeadAfter = Git-Output @("rev-parse", $RemoteRef)

if ($remoteHeadAfter -ne $localHead) {
    throw "Push verification failed: $RemoteRef is $remoteHeadAfter, expected $localHead."
}

Write-Host "push succeeded: $RemoteRef now equals $localHead"
