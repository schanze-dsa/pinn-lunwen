param(
    [string]$LensFile = "",
    [string]$OutputDir = ".\outputs\zemax_bridge",
    [string]$GridSagFile = "",
    [int]$GridSagSurface = -1,
    [string]$SampleRelativePath = "Sequential\Objectives\Double Gauss 28 degree field.zos",
    [int]$MtfMaxFrequency = 50
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

function Invoke-WindowsPowerShellIfNeeded {
    if ($PSVersionTable.PSEdition -eq "Desktop") {
        return
    }

    $winPs = Join-Path $env:WINDIR "System32\WindowsPowerShell\v1.0\powershell.exe"
    $args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath)
    foreach ($entry in $MyInvocation.BoundParameters.GetEnumerator()) {
        $args += "-$($entry.Key)"
        $args += [string]$entry.Value
    }
    & $winPs @args
    exit $LASTEXITCODE
}

function Initialize-ZemaxApi {
    $zemaxRoot = $null
    try {
        $zemaxRoot = (Get-ItemProperty -Path "HKCU:\Software\Zemax" -ErrorAction Stop).ZemaxRoot
    } catch {
        throw "Cannot read HKCU:\Software\Zemax\ZemaxRoot. Open OpticStudio once or repair the installation."
    }

    $netHelper = Join-Path $zemaxRoot "ZOS-API\Libraries\ZOSAPI_NetHelper.dll"
    if (!(Test-Path -LiteralPath $netHelper)) {
        throw "Cannot find ZOSAPI_NetHelper.dll at $netHelper"
    }

    Add-Type -Path $netHelper
    $ok = [ZOSAPI_NetHelper.ZOSAPI_Initializer]::Initialize()
    if (!$ok) {
        throw "ZOSAPI_NetHelper failed to locate OpticStudio."
    }
    $zemaxDir = [ZOSAPI_NetHelper.ZOSAPI_Initializer]::GetZemaxDirectory()
    Add-Type -Path (Join-Path $zemaxDir "ZOSAPI_Interfaces.dll")
    Add-Type -Path (Join-Path $zemaxDir "ZOSAPI.dll")

    return $zemaxDir
}

function Write-CsvRows {
    param(
        [string]$Path,
        [string[]]$Header,
        [object[]]$Rows
    )
    $parent = Split-Path -Parent $Path
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add(($Header -join ","))
    foreach ($row in $Rows) {
        $lines.Add(($row -join ","))
    }
    [System.IO.File]::WriteAllLines($Path, $lines.ToArray(), [System.Text.Encoding]::UTF8)
}

function Export-SpotMetrics {
    param(
        [object]$System,
        [string]$OutputCsv
    )
    $spot = $System.Analyses.New_Analysis([ZOSAPI.Analysis.AnalysisIDM]::StandardSpot)
    $settings = $spot.GetSettings()
    $settings.Field.SetFieldNumber(0)
    $settings.Wavelength.SetWavelengthNumber(0)
    $settings.ReferTo = [ZOSAPI.Analysis.Settings.Spot.Reference]::Centroid
    $spot.ApplyAndWaitForCompletion()
    $results = $spot.GetResults()

    $rows = @()
    $nFields = [int]$System.SystemData.Fields.NumberOfFields
    $nWaves = [int]$System.SystemData.Wavelengths.NumberOfWavelengths
    for ($field = 1; $field -le $nFields; $field++) {
        for ($wave = 1; $wave -le $nWaves; $wave++) {
            $rms = $results.SpotData.GetRMSSpotSizeFor($field, $wave)
            $geo = $results.SpotData.GetGeoSpotSizeFor($field, $wave)
            $rows += ,@($field, $wave, ("{0:R}" -f $rms), ("{0:R}" -f $geo))
        }
    }
    Write-CsvRows -Path $OutputCsv -Header @("field", "wave", "rms_spot_radius", "geo_spot_radius") -Rows $rows
    $spot.Close()
    return $rows
}

function Export-MtfData {
    param(
        [object]$System,
        [string]$OutputCsv,
        [int]$MaxFrequency
    )
    $mtf = $System.Analyses.New_FftMtf()
    $settings = $mtf.GetSettings()
    $settings.MaximumFrequency = $MaxFrequency
    $settings.SampleSize = [ZOSAPI.Analysis.SampleSizes]::S_256x256
    $mtf.ApplyAndWaitForCompletion()
    $results = $mtf.GetResults()

    $rows = @()
    $seriesCount = [int]$results.NumberOfDataSeries
    for ($seriesIndex = 0; $seriesIndex -lt $seriesCount; $seriesIndex++) {
        $series = $results.GetDataSeries($seriesIndex)
        $xCount = [int]$series.XData.Length
        for ($k = 0; $k -lt $xCount; $k++) {
            $x = $series.XData.GetValueAt($k)
            for ($j = 0; $j -lt [int]$series.NumSeries; $j++) {
                $y = $series.YData.GetValueAt($k, $j)
                $rows += ,@($seriesIndex, $j, ("{0:R}" -f $x), ("{0:R}" -f $y))
            }
        }
    }
    Write-CsvRows -Path $OutputCsv -Header @("series", "subseries", "frequency", "mtf") -Rows $rows
    $mtf.Close()
    return $rows
}

function Attach-GridSagIfRequested {
    param(
        [object]$Application,
        [object]$System,
        [string]$GridFile,
        [int]$SurfaceNumber
    )
    if ([string]::IsNullOrWhiteSpace($GridFile)) {
        return $null
    }
    if (!(Test-Path -LiteralPath $GridFile)) {
        throw "GridSagFile does not exist: $GridFile"
    }

    $targetName = [System.IO.Path]::GetFileName($GridFile)

    if ($SurfaceNumber -ge 0) {
        $surface = $System.LDE.GetSurfaceAt($SurfaceNumber)
        if ($surface.Type -ne [ZOSAPI.Editors.LDE.SurfaceType]::GridSag) {
            $typeSettings = $surface.GetSurfaceTypeSettings([ZOSAPI.Editors.LDE.SurfaceType]::GridSag)
            if (!$typeSettings.IsValid) {
                throw "GridSag surface settings are invalid for surface $SurfaceNumber"
            }
            [void]$surface.ChangeType($typeSettings)
        }

        $gridDir = $surface.ImportData.DefaultImportDirectory
        New-Item -ItemType Directory -Force -Path $gridDir | Out-Null
        $targetPath = Join-Path $gridDir $targetName
        Copy-Item -LiteralPath $GridFile -Destination $targetPath -Force
        $imported = $surface.ImportData.ImportDataFile($targetPath)
        if (!$imported -or !$surface.ImportData.IsValid) {
            throw "Failed to import Grid Sag data file on surface $SurfaceNumber`: $targetPath"
        }
    } else {
        $gridDir = Join-Path $Application.ZemaxDataDir "Objects\Grid Files"
        New-Item -ItemType Directory -Force -Path $gridDir | Out-Null
        $targetPath = Join-Path $gridDir $targetName
        Copy-Item -LiteralPath $GridFile -Destination $targetPath -Force
    }

    return @{
        source = (Resolve-Path -LiteralPath $GridFile).Path
        copied_to = $targetPath
        surface = $SurfaceNumber
    }
}

Invoke-WindowsPowerShellIfNeeded

$outRoot = Join-Path (Resolve-Path -LiteralPath ".").Path $OutputDir
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

$zemaxDir = Initialize-ZemaxApi
$connection = New-Object ZOSAPI.ZOSAPI_Connection
$application = $connection.CreateNewApplication()
if ($null -eq $application) {
    throw "CreateNewApplication returned null."
}
if (!$application.IsValidLicenseForAPI) {
    throw "ZOS-API license is not valid: $($application.LicenseStatus)"
}

try {
    $system = $application.PrimarySystem
    if ([string]::IsNullOrWhiteSpace($LensFile)) {
        $LensFile = Join-Path $application.SamplesDir $SampleRelativePath
    }
    if (!(Test-Path -LiteralPath $LensFile)) {
        throw "Lens file does not exist: $LensFile"
    }
    $system.LoadFile($LensFile, $false)

    $gridInfo = Attach-GridSagIfRequested `
        -Application $application `
        -System $system `
        -GridFile $GridSagFile `
        -SurfaceNumber $GridSagSurface

    $spotCsv = Join-Path $outRoot "spot_metrics.csv"
    $mtfCsv = Join-Path $outRoot "fft_mtf.csv"
    $spotRows = Export-SpotMetrics -System $system -OutputCsv $spotCsv
    $mtfRows = Export-MtfData -System $system -OutputCsv $mtfCsv -MaxFrequency $MtfMaxFrequency

    $copyPath = Join-Path $outRoot "evaluated_system.zos"
    $system.SaveAs($copyPath)

    $summary = [ordered]@{
        ok = $true
        zemax_dir = $zemaxDir
        zemax_data_dir = $application.ZemaxDataDir
        lens_file = (Resolve-Path -LiteralPath $LensFile).Path
        evaluated_system = $copyPath
        grid_sag = $gridInfo
        spot_csv = $spotCsv
        mtf_csv = $mtfCsv
        spot_rows = $spotRows.Count
        mtf_rows = $mtfRows.Count
        license_status = [string]$application.LicenseStatus
    }
    $summaryPath = Join-Path $outRoot "summary.json"
    $summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
    Write-Host "Zemax optical evaluation completed."
    Write-Host "Summary: $summaryPath"
    Write-Host "Spot CSV: $spotCsv"
    Write-Host "MTF CSV: $mtfCsv"
} finally {
    if ($null -ne $application) {
        $application.CloseApplication()
    }
}
