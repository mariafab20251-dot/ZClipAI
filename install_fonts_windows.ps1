<#
Installs all bundled caption fonts (assets/fonts/*.ttf, *.otf) into the current
user's Windows font store so every app on the system can use them.

Per-user install: no admin required, writes to
  %LOCALAPPDATA%\Microsoft\Windows\Fonts
and registers each face under
  HKCU:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts

Re-running is safe (idempotent) — already-installed fonts are skipped.
#>

$ErrorActionPreference = 'Stop'

$srcDir = Join-Path $PSScriptRoot 'assets\fonts'
if (-not (Test-Path $srcDir)) {
    Write-Error "Fonts directory not found: $srcDir"
    exit 1
}

$userFonts = Join-Path $env:LOCALAPPDATA 'Microsoft\Windows\Fonts'
if (-not (Test-Path $userFonts)) {
    New-Item -ItemType Directory -Path $userFonts -Force | Out-Null
}
$regKey = 'HKCU:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts'
if (-not (Test-Path $regKey)) {
    New-Item -Path $regKey -Force | Out-Null
}

# Read the font's real family name from its 'name' table via System.Drawing.
Add-Type -AssemblyName System.Drawing

function Get-FontFamilyName([string]$path) {
    try {
        $pfc = New-Object System.Drawing.Text.PrivateFontCollection
        $pfc.AddFontFile($path)
        $name = $pfc.Families[0].Name
        $pfc.Dispose()
        return $name
    } catch {
        # Fall back to the file's base name if the face can't be parsed.
        return [System.IO.Path]::GetFileNameWithoutExtension($path)
    }
}

$installed = 0
$skipped = 0

Get-ChildItem -Path $srcDir -Include *.ttf, *.otf -File -Recurse | ForEach-Object {
    $file = $_
    $ext = $file.Extension.ToLower()
    $suffix = if ($ext -eq '.otf') { '(OpenType)' } else { '(TrueType)' }
    $family = Get-FontFamilyName $file.FullName
    $regValue = "$family $suffix"

    $dest = Join-Path $userFonts $file.Name

    $existingVal = (Get-ItemProperty -Path $regKey -Name $regValue -ErrorAction SilentlyContinue).$regValue
    if ((Test-Path $dest) -and $existingVal) {
        Write-Host "  skip  $family" -ForegroundColor DarkGray
        $script:skipped++
        return
    }

    Copy-Item -Path $file.FullName -Destination $dest -Force
    # Store just the filename; Windows resolves it inside the user Fonts dir.
    New-ItemProperty -Path $regKey -Name $regValue -Value $file.Name -PropertyType String -Force | Out-Null
    Write-Host "  ok    $family" -ForegroundColor Green
    $script:installed++
}

Write-Host ""
Write-Host "Fonts installed: $installed, already present: $skipped" -ForegroundColor Cyan
Write-Host "New fonts are available to apps launched from now on." -ForegroundColor Cyan
