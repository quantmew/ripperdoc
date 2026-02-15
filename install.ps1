[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidatePattern('^(latest|stable|v?[0-9]+\.[0-9]+\.[0-9]+([-.][A-Za-z0-9._-]+)?)$')]
    [string]$Version = "latest",

    [string]$BinDir = "$env:LOCALAPPDATA\Ripperdoc\bin",

    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = 'quantmew/ripperdoc'
$apiBase = "https://api.github.com/repos/$repo"
$releaseBase = "https://github.com/$repo/releases/download"
$appBase = Join-Path $env:LOCALAPPDATA 'Ripperdoc'
$versionsDir = Join-Path $appBase 'versions'
$stateFile = Join-Path $appBase 'version'

function Resolve-Version {
    param([string]$InputVersion)

    if ($InputVersion -in @('latest', 'stable')) {
        try {
            return (Invoke-RestMethod -Uri "$apiBase/releases/latest").tag_name
        } catch {
            throw "Failed to query latest version: $($_.Exception.Message)"
        }
    }

    if ($InputVersion -like 'v*') {
        return $InputVersion
    }

    return "v$InputVersion"
}

function Get-Platform {
    if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') {
        return 'windows-arm64'
    }
    if ($env:PROCESSOR_ARCHITECTURE -in @('AMD64', 'X64') -or $env:PROCESSOR_ARCHITEW6432 -in @('AMD64', 'X64')) {
        return 'windows-x86_64'
    }
    throw "Unsupported architecture: $env:PROCESSOR_ARCHITECTURE"
}

function Get-InstalledVersion {
    if (Test-Path $stateFile) {
        return (Get-Content -Raw $stateFile).Trim()
    }
    return ''
}

function Ensure-UserPath {
    param([string]$PathToAdd)

    $normalized = $PathToAdd.TrimEnd('\')
    $current = [Environment]::GetEnvironmentVariable('Path', 'User')
    $entries = @()
    if (-not [string]::IsNullOrWhiteSpace($current)) {
        $entries = $current -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    }

    $exists = $entries | Where-Object { $_.TrimEnd('\').ToLowerInvariant() -eq $normalized.ToLowerInvariant() }
    if ($exists.Count -gt 0) {
        return
    }

    $entries += $normalized
    [Environment]::SetEnvironmentVariable('Path', ($entries | Select-Object -Unique) -join ';', 'User')
    if (";$env:PATH;" -notlike "*;$normalized;*") {
        $env:PATH = "$env:PATH;$normalized"
    }
}

function Set-EntryLink {
    param(
        [string]$LinkPath,
        [string]$TargetPath
    )

    if (Test-Path $LinkPath) {
        Remove-Item -Force $LinkPath
    }

    try {
        New-Item -ItemType SymbolicLink -Path $LinkPath -Target $TargetPath -ErrorAction Stop | Out-Null
        return
    } catch {
        # fallback to hardlink then copy
    }

    try {
        New-Item -ItemType HardLink -Path $LinkPath -Target $TargetPath -ErrorAction Stop | Out-Null
        return
    } catch {
        Copy-Item -Force $TargetPath $LinkPath
    }
}

function Remove-EntryLink {
    param([string]$LinkPath)

    if (Test-Path $LinkPath) {
        Remove-Item -Force $LinkPath
    }
}

$resolvedVersion = Resolve-Version $Version
if ($resolvedVersion -notmatch '^v[0-9]+\.[0-9]+\.[0-9]+([-.][A-Za-z0-9._-]+)?$') {
    throw \"Invalid version: $resolvedVersion\"
}

$normalizedBinDir = $BinDir.TrimEnd('\')
New-Item -ItemType Directory -Force -Path $versionsDir | Out-Null
New-Item -ItemType Directory -Force -Path $normalizedBinDir | Out-Null

if ($Uninstall) {
    $entryPath = Join-Path $normalizedBinDir 'ripperdoc.exe'
    Remove-EntryLink -LinkPath $entryPath

    $installedVersion = Get-InstalledVersion
    if ($installedVersion) {
        $installedDir = Join-Path $versionsDir $installedVersion
        if (Test-Path $installedDir) {
            Remove-Item -Recurse -Force $installedDir
        }
        Remove-Item -Force $stateFile -ErrorAction Ignore
        Write-Host \"Uninstalled ripperdoc $installedVersion\"
    } else {
        Write-Host 'No installed version metadata found; removed launcher only.'
    }
    exit 0
}

$platform = Get-Platform
$binaryName = 'ripperdoc.exe'
$archiveName = \"ripperdoc-$resolvedVersion-$platform.zip\"
$downloadUrl = \"$releaseBase/$resolvedVersion/$archiveName\"

$versionDir = Join-Path $versionsDir $resolvedVersion
New-Item -ItemType Directory -Force -Path $versionDir | Out-Null
$versionBinary = Join-Path $versionDir $binaryName
if (Test-Path $versionBinary) { Remove-Item -Force $versionBinary }

$tempRoot = Join-Path $env:TEMP \"ripperdoc-install-$(New-Guid)\"
$archivePath = Join-Path $tempRoot $archiveName
$extractDir = Join-Path $tempRoot 'payload'

try {
    New-Item -ItemType Directory -Path $tempRoot | Out-Null
    Invoke-WebRequest -Uri $downloadUrl -OutFile $archivePath
    Expand-Archive -Path $archivePath -DestinationPath $extractDir -Force

    $binary = Get-ChildItem -Path $extractDir -Recurse -Filter $binaryName | Select-Object -First 1
    if (-not $binary) {
        throw \"Downloaded archive does not contain $binaryName\"
    }

    Copy-Item -Force $binary.FullName $versionBinary
    $entryPath = Join-Path $normalizedBinDir $binaryName
    Set-EntryLink -LinkPath $entryPath -TargetPath $versionBinary
    Ensure-UserPath $normalizedBinDir

    New-Item -ItemType Directory -Force -Path $appBase | Out-Null
    Set-Content -Path $stateFile -Value $resolvedVersion -NoNewline

    Write-Host \"Installed ripperdoc $resolvedVersion to $versionDir\"
    Write-Host \"Launcher: $entryPath\"
} finally {
    if (Test-Path $tempRoot) {
        Remove-Item -Recurse -Force $tempRoot
    }
}
