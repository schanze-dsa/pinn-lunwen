param(
    [string]$LensFile = "",
    [string]$OutputDir = ".\outputs\zemax_bridge\energy_concentration",
    [double]$EncircledRadiusUm = 8.021,
    [double]$EnsquaredHalfWidthUm = 8.0,
    [double]$RadiusMaximumUm = 20.0,
    [int]$GridSagSurface = 5,
    [string]$SampleSize = "S_512x512",
    [string]$BaselineName = "baseline",
    [string]$FullCaseName = "",
    [string]$AnnularCaseName = "",
    [string]$FullGridSagFile = ".\outputs\zemax_bridge\pinn_case_01_123_s3_grid_sag.dat",
    [string]$AnnularGridSagFile = ".\outputs\zemax_bridge\pinn_case_01_123_s3_annular_grid_sag.dat"
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

    $surface = $System.LDE.GetSurfaceAt($SurfaceNumber)
    if ($surface.Type -ne [ZOSAPI.Editors.LDE.SurfaceType]::GridSag) {
        $typeSettings = $surface.GetSurfaceTypeSettings([ZOSAPI.Editors.LDE.SurfaceType]::GridSag)
        if (!$typeSettings.IsValid) {
            throw "GridSag surface settings are invalid for surface $SurfaceNumber"
        }
        [void]$surface.ChangeType($typeSettings)
    }

    $targetName = [System.IO.Path]::GetFileName($GridFile)
    $gridDir = $surface.ImportData.DefaultImportDirectory
    New-Item -ItemType Directory -Force -Path $gridDir | Out-Null
    $targetPath = Join-Path $gridDir $targetName
    Copy-Item -LiteralPath $GridFile -Destination $targetPath -Force
    $imported = $surface.ImportData.ImportDataFile($targetPath)
    if (!$imported -or !$surface.ImportData.IsValid) {
        throw "Failed to import Grid Sag data file on surface $SurfaceNumber`: $targetPath"
    }

    return @{
        source = (Resolve-Path -LiteralPath $GridFile).Path
        copied_to = $targetPath
        surface = $SurfaceNumber
    }
}

function Convert-ToInvariant {
    param([double]$Value)
    return $Value.ToString("R", [System.Globalization.CultureInfo]::InvariantCulture)
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
        $escaped = @()
        foreach ($cell in $row) {
            $s = [string]$cell
            if ($s.Contains(",") -or $s.Contains('"') -or $s.Contains("`n")) {
                $s = '"' + $s.Replace('"', '""') + '"'
            }
            $escaped += $s
        }
        $lines.Add(($escaped -join ","))
    }
    [System.IO.File]::WriteAllLines($Path, $lines.ToArray(), [System.Text.Encoding]::UTF8)
}

function Get-InterpolatedValue {
    param(
        [double[]]$X,
        [double[]]$Y,
        [double]$Target
    )
    if ($X.Length -ne $Y.Length -or $X.Length -eq 0) {
        throw "Cannot interpolate empty or mismatched series."
    }
    if ($Target -le $X[0]) {
        return $Y[0]
    }
    $last = $X.Length - 1
    if ($Target -ge $X[$last]) {
        return $Y[$last]
    }
    for ($i = 1; $i -lt $X.Length; $i++) {
        if ($X[$i] -ge $Target) {
            $x0 = $X[$i - 1]
            $x1 = $X[$i]
            $y0 = $Y[$i - 1]
            $y1 = $Y[$i]
            if ([Math]::Abs($x1 - $x0) -lt 1.0e-15) {
                return $y1
            }
            $t = ($Target - $x0) / ($x1 - $x0)
            return $y0 + $t * ($y1 - $y0)
        }
    }
    return $Y[$last]
}

function Export-DiffractionEnergy {
    param(
        [object]$System,
        [string]$CaseName,
        [string]$MetricName,
        [string]$EnergyType,
        [double]$TargetDistanceUm,
        [double]$RadiusMaxUm,
        [string]$SampleSizeName,
        [string]$OutputRoot
    )

    $analysis = $System.Analyses.New_DiffractionEncircledEnergy()
    $settings = $analysis.GetSettings()
    $settings.RadiusMaximum = $RadiusMaxUm
    $settings.SampleSize = [ZOSAPI.Analysis.SampleSizes]::$SampleSizeName
    $settings.ReferTo = [ZOSAPI.Analysis.Settings.EncircledEnergy.ReferToTypes]::Centroid
    $settings.Type = [ZOSAPI.Analysis.Settings.EncircledEnergy.EncircledEnergyTypes]::$EnergyType
    $settings.ShowDiffractionLimit = $true

    $rawText = Join-Path $OutputRoot "$CaseName`_$MetricName`_raw.txt"
    $curveCsv = Join-Path $OutputRoot "$CaseName`_$MetricName`_curve.csv"

    $analysis.ApplyAndWaitForCompletion()
    $results = $analysis.GetResults()
    $results.GetTextFile($rawText)

    $rows = New-Object System.Collections.Generic.List[object]
    $summaryRows = New-Object System.Collections.Generic.List[object]
    $seriesCount = [int]$results.NumberOfDataSeries
    for ($seriesIndex = 0; $seriesIndex -lt $seriesCount; $seriesIndex++) {
        $series = $results.GetDataSeries($seriesIndex)
        $xValues = New-Object double[] ([int]$series.XData.Length)
        $yValues = New-Object double[] ([int]$series.XData.Length)
        for ($k = 0; $k -lt [int]$series.XData.Length; $k++) {
            $x = [double]$series.XData.GetValueAt($k)
            $y = [double]$series.YData.GetValueAt($k, 0)
            $xValues[$k] = $x
            $yValues[$k] = $y
            [void]$rows.Add([pscustomobject]@{
                case = $CaseName
                metric = $MetricName
                series = $seriesIndex
                distance_um = $x
                energy_fraction = $y
            })
        }
        $targetValue = Get-InterpolatedValue -X $xValues -Y $yValues -Target $TargetDistanceUm
        $seriesRole = "actual"
        if ($seriesCount -gt 1 -and $seriesIndex -eq 0) {
            $seriesRole = "diffraction_limit"
        }
        $summaryObject = [pscustomobject]@{
            case = $CaseName
            metric = $MetricName
            series = $seriesIndex
            series_role = $seriesRole
            target_distance_um = $TargetDistanceUm
            energy_fraction = $targetValue
            energy_percent = 100.0 * $targetValue
            raw_text = $rawText
            curve_csv = $curveCsv
        }
        [void]$summaryRows.Add($summaryObject)
        if ($null -ne $script:EnergySummaryRows) {
            [void]$script:EnergySummaryRows.Add(@(
                $CaseName,
                $MetricName,
                $seriesIndex,
                $seriesRole,
                (Convert-ToInvariant $TargetDistanceUm),
                (Convert-ToInvariant $targetValue),
                (Convert-ToInvariant (100.0 * $targetValue)),
                $rawText,
                $curveCsv
            ))
        }
    }

    $rows | Export-Csv -LiteralPath $curveCsv -NoTypeInformation -Encoding UTF8
    $analysis.Close()
    return $summaryRows.ToArray()
}

function Run-Case {
    param(
        [object]$Application,
        [string]$LensPath,
        [string]$CaseName,
        [string]$GridFile,
        [int]$SurfaceNumber,
        [string]$OutputRoot,
        [double]$EncRadiusUm,
        [double]$EnsHalfWidthUm,
        [double]$RadiusMaxUm,
        [string]$SampleSizeName
    )
    $system = $Application.PrimarySystem
    $system.LoadFile($LensPath, $false)

    $gridInfo = Attach-GridSagIfRequested `
        -Application $Application `
        -System $system `
        -GridFile $GridFile `
        -SurfaceNumber $SurfaceNumber

    $rows = @()
    $rows += Export-DiffractionEnergy `
        -System $system `
        -CaseName $CaseName `
        -MetricName "encircled_r8p021um" `
        -EnergyType "Encircled" `
        -TargetDistanceUm $EncRadiusUm `
        -RadiusMaxUm $RadiusMaxUm `
        -SampleSizeName $SampleSizeName `
        -OutputRoot $OutputRoot
    $rows += Export-DiffractionEnergy `
        -System $system `
        -CaseName $CaseName `
        -MetricName "ensquared_halfwidth8um" `
        -EnergyType "Ensquared" `
        -TargetDistanceUm $EnsHalfWidthUm `
        -RadiusMaxUm $RadiusMaxUm `
        -SampleSizeName $SampleSizeName `
        -OutputRoot $OutputRoot

    $copyPath = Join-Path $OutputRoot "$CaseName`_evaluated_system.zos"
    $system.SaveAs($copyPath)
    return $rows
}

Invoke-WindowsPowerShellIfNeeded

$repoRoot = (Resolve-Path -LiteralPath ".").Path
$outRoot = Join-Path $repoRoot $OutputDir
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

if ([string]::IsNullOrWhiteSpace($LensFile)) {
    throw "LensFile is required. Provide a Zemax .zmx file path."
}
$lensPath = (Resolve-Path -LiteralPath $LensFile).Path
$fullGrid = ""
if (![string]::IsNullOrWhiteSpace($FullGridSagFile)) {
    $fullGrid = (Resolve-Path -LiteralPath $FullGridSagFile).Path
}
$annularGrid = ""
if (![string]::IsNullOrWhiteSpace($AnnularGridSagFile)) {
    $annularGrid = (Resolve-Path -LiteralPath $AnnularGridSagFile).Path
}

$script:EnergySummaryRows = New-Object System.Collections.Generic.List[object]

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
    if ([string]::IsNullOrWhiteSpace($FullCaseName)) {
        $FullCaseName = "pinn_full_grid_sag_s$GridSagSurface"
    }
    if ([string]::IsNullOrWhiteSpace($AnnularCaseName)) {
        $AnnularCaseName = "pinn_annular_grid_sag_s$GridSagSurface"
    }
    $cases = @(
        @{ name = $BaselineName; grid = "" },
        @{ name = $FullCaseName; grid = $fullGrid },
        @{ name = $AnnularCaseName; grid = $annularGrid }
    )

    foreach ($case in $cases) {
        if (![string]::IsNullOrWhiteSpace($case.grid) -or $case.name -eq $BaselineName) {
            Write-Host "Running energy concentration case: $($case.name)"
            [void](Run-Case `
                -Application $application `
                -LensPath $lensPath `
                -CaseName $case.name `
                -GridFile $case.grid `
                -SurfaceNumber $GridSagSurface `
                -OutputRoot $outRoot `
                -EncRadiusUm $EncircledRadiusUm `
                -EnsHalfWidthUm $EnsquaredHalfWidthUm `
                -RadiusMaxUm $RadiusMaximumUm `
                -SampleSizeName $SampleSize)
        }
    }

    $summaryCsv = Join-Path $outRoot "energy_summary.csv"
    Write-CsvRows -Path $summaryCsv `
        -Header @("case", "metric", "series", "series_role", "target_distance_um", "energy_fraction", "energy_percent", "raw_text", "curve_csv") `
        -Rows $script:EnergySummaryRows.ToArray()

    $summaryJson = Join-Path $outRoot "summary.json"
    [ordered]@{
        ok = $true
        lens_file = $lensPath
        output_dir = $outRoot
        summary_csv = $summaryCsv
        zemax_dir = $zemaxDir
        zemax_data_dir = $application.ZemaxDataDir
        license_status = [string]$application.LicenseStatus
        sample_size = $SampleSize
        encircled_radius_um = $EncircledRadiusUm
        ensquared_half_width_um = $EnsquaredHalfWidthUm
        radius_maximum_um = $RadiusMaximumUm
        grid_sag_surface = $GridSagSurface
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryJson -Encoding UTF8

    Write-Host "Energy concentration evaluation completed."
    Write-Host "Summary CSV: $summaryCsv"
    Write-Host "Summary JSON: $summaryJson"
} finally {
    if ($null -ne $application) {
        $application.CloseApplication()
    }
}
