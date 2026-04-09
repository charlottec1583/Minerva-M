param(
    [string]$MetricsCsv = "presentation_charts/ck_comparison_5models_final/ck_comparison_metrics.csv",
    [string]$OutputPng = "presentation_charts/ck_comparison_5models_final/ck_comparison.png",
    [string]$Title = "Cumulative Success Rate by Stage (C_k)"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

function Get-ColorByModelName {
    param([string]$ModelName, [int]$FallbackIndex = 0)

    $name = $ModelName.ToLowerInvariant()
    if ($name.Contains("qwen")) { return [System.Drawing.ColorTranslator]::FromHtml("#ea5a5a") }
    if ($name.Contains("deepseek")) { return [System.Drawing.ColorTranslator]::FromHtml("#7fbf7b") }
    if ($name.Contains("gemini")) { return [System.Drawing.ColorTranslator]::FromHtml("#f5c04a") }
    if ($name.Contains("gpt-3.5") -or $name.Contains("gpt 3.5") -or $name.Contains("turbo")) {
        return [System.Drawing.ColorTranslator]::FromHtml("#c18a67")
    }
    if ($name.Contains("gpt-5.1") -or $name.Contains("gpt 5.1")) {
        return [System.Drawing.ColorTranslator]::FromHtml("#b3b3b3")
    }

    $palette = @("#ea5a5a", "#7fbf7b", "#f5c04a", "#c18a67", "#b3b3b3")
    return [System.Drawing.ColorTranslator]::FromHtml($palette[$FallbackIndex % $palette.Count])
}

function New-RoundedRectPath {
    param(
        [float]$X,
        [float]$Y,
        [float]$Width,
        [float]$Height,
        [float]$Radius
    )

    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $diameter = $Radius * 2
    $path.AddArc($X, $Y, $diameter, $diameter, 180, 90)
    $path.AddArc($X + $Width - $diameter, $Y, $diameter, $diameter, 270, 90)
    $path.AddArc($X + $Width - $diameter, $Y + $Height - $diameter, $diameter, $diameter, 0, 90)
    $path.AddArc($X, $Y + $Height - $diameter, $diameter, $diameter, 90, 90)
    $path.CloseFigure()
    return $path
}

function Measure-CenteredX {
    param(
        [System.Drawing.Graphics]$Graphics,
        [string]$Text,
        [System.Drawing.Font]$Font,
        [float]$CenterX
    )

    $size = $Graphics.MeasureString($Text, $Font)
    return $CenterX - ($size.Width / 2.0)
}

$stageNames = @{
    1 = "Context Establishment"
    2 = "Relationship Building"
    3 = "Constraint Induction"
    4 = "Escalation"
}

$stageLabelOffsets = @{
    1 = @(
        @{ X = 48; Y = -30 },
        @{ X = 48; Y = -10 },
        @{ X = 48; Y = 10 },
        @{ X = 48; Y = 30 },
        @{ X = 48; Y = 50 }
    )
    2 = @(
        @{ X = -48; Y = -30 },
        @{ X = 48; Y = -10 },
        @{ X = -48; Y = 10 },
        @{ X = 48; Y = 30 },
        @{ X = 0; Y = 50 }
    )
    3 = @(
        @{ X = -56; Y = -36 },
        @{ X = 56; Y = -18 },
        @{ X = -56; Y = 0 },
        @{ X = 56; Y = 18 },
        @{ X = 0; Y = 40 }
    )
    4 = @(
        @{ X = -48; Y = -30 },
        @{ X = -48; Y = -10 },
        @{ X = -48; Y = 10 },
        @{ X = -48; Y = 30 },
        @{ X = -48; Y = 50 }
    )
}

$csvPath = [System.IO.Path]::GetFullPath($MetricsCsv)
$pngPath = [System.IO.Path]::GetFullPath($OutputPng)

if (-not (Test-Path -LiteralPath $csvPath)) {
    throw "Metrics CSV not found: $csvPath"
}

$csvHeaders = @(
    "target_model",
    "source_summary",
    "N",
    "ASR",
    "AUC_C",
    "C1",
    "C2",
    "C3",
    "C4",
    "Delta1",
    "Delta2",
    "Delta3",
    "Delta4_upper",
    "ratio_c3_c4",
    "delta4",
    "earliest_converge_stage",
    "decision"
)

$records = (Get-Content -LiteralPath $csvPath | Select-Object -Skip 1 | ConvertFrom-Csv -Header $csvHeaders) | Sort-Object target_model | ForEach-Object {
    [PSCustomObject]@{
        Model = $_.target_model
        C1 = [double]$_.C1
        C2 = [double]$_.C2
        C3 = [double]$_.C3
        C4 = [double]$_.C4
        Ratio = if ([string]::IsNullOrWhiteSpace($_.ratio_c3_c4)) { $null } else { [double]$_.ratio_c3_c4 }
        Delta4 = if ([string]::IsNullOrWhiteSpace($_.delta4)) { $null } else { [double]$_.delta4 }
        EarliestConvergeStage = if ([string]::IsNullOrWhiteSpace($_.earliest_converge_stage)) { $null } else { [int]$_.earliest_converge_stage }
        Decision = $_.decision
    }
}

$width = 1640
$height = 860
$cardInset = 32.0
$left = 100.0
$right = 450.0
$top = 145.0
$bottom = 170.0
$plotWidth = $width - $left - $right
$plotHeight = $height - $top - $bottom
$yMin = 0.25
$yMax = 1.0
$legendX = $left + $plotWidth + 28.0
$legendY = $top + 24.0

function Get-XOfStage {
    param([int]$Stage)
    return $left + (($Stage - 1) * ($plotWidth / 3.0))
}

function Get-YOfValue {
    param([double]$Value)
    $clamped = [Math]::Min([Math]::Max($Value, $yMin), $yMax)
    return $top + (($yMax - $clamped) / ($yMax - $yMin) * $plotHeight)
}

$bitmap = New-Object System.Drawing.Bitmap $width, $height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

$backgroundBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.ColorTranslator]::FromHtml("#f8fafc"))
$cardBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::White)
$cardPen = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml("#e5e7eb"), 1.2)
$gridPen = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml("#e2e8f0"), 1.0)
$axisPen = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml("#0f172a"), 2.0)
$tickPen = New-Object System.Drawing.Pen ([System.Drawing.ColorTranslator]::FromHtml("#0f172a"), 1.5)

$titleFont = New-Object System.Drawing.Font("Segoe UI", 28, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
$subtitleFont = New-Object System.Drawing.Font("Segoe UI", 15, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
$axisFont = New-Object System.Drawing.Font("Segoe UI", 15, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
$tickFont = New-Object System.Drawing.Font("Segoe UI", 13, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
$labelFont = New-Object System.Drawing.Font("Segoe UI", 12, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
$legendTitleFont = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
$legendFont = New-Object System.Drawing.Font("Segoe UI", 12, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)

$titleBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.ColorTranslator]::FromHtml("#1f2937"))
$subtitleBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.ColorTranslator]::FromHtml("#475569"))
$axisBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.ColorTranslator]::FromHtml("#111827"))
$tickBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.ColorTranslator]::FromHtml("#334155"))

$graphics.FillRectangle($backgroundBrush, 0, 0, $width, $height)
$cardPath = New-RoundedRectPath $cardInset $cardInset ($width - (2 * $cardInset)) ($height - (2 * $cardInset)) 20
$graphics.FillPath($cardBrush, $cardPath)
$graphics.DrawPath($cardPen, $cardPath)

$centerX = $width / 2.0
$titleX = Measure-CenteredX $graphics $Title $titleFont $centerX
$graphics.DrawString($Title, $titleFont, $titleBrush, $titleX, 35)

$subtitle = "Compare C1-C4 across target models to see whether the curves flatten after a specific stage."
$subtitleX = Measure-CenteredX $graphics $subtitle $subtitleFont $centerX
$graphics.DrawString($subtitle, $subtitleFont, $subtitleBrush, $subtitleX, 80)

foreach ($tickValue in @(0.25, 0.50, 0.75, 1.00)) {
    $y = Get-YOfValue $tickValue
    $graphics.DrawLine($gridPen, $left, $y, $left + $plotWidth, $y)
    $pct = [int][Math]::Round($tickValue * 100)
    $label = "$pct%"
    $size = $graphics.MeasureString($label, $tickFont)
    $graphics.DrawString($label, $tickFont, $tickBrush, $left - 12 - $size.Width, $y - ($size.Height / 2.0))
}

$graphics.DrawLine($axisPen, $left, $top, $left, $top + $plotHeight)
$graphics.DrawLine($axisPen, $left, $top + $plotHeight, $left + $plotWidth, $top + $plotHeight)

for ($stage = 1; $stage -le 4; $stage++) {
    $x = Get-XOfStage $stage
    $graphics.DrawLine($tickPen, $x, $top + $plotHeight, $x, $top + $plotHeight + 6)

    $stageLabel = "Stage ${stage}:"
    $stageSize = $graphics.MeasureString($stageLabel, $tickFont)
    $graphics.DrawString($stageLabel, $tickFont, $tickBrush, $x - ($stageSize.Width / 2.0), $top + $plotHeight + 12)

    $stageName = $stageNames[$stage]
    $stageNameSize = $graphics.MeasureString($stageName, $tickFont)
    $graphics.DrawString($stageName, $tickFont, $tickBrush, $x - ($stageNameSize.Width / 2.0), $top + $plotHeight + 30)
}

$xAxisLabel = "Stage k"
$xAxisX = Measure-CenteredX $graphics $xAxisLabel $axisFont ($left + ($plotWidth / 2.0))
$graphics.DrawString($xAxisLabel, $axisFont, $axisBrush, $xAxisX, $height - 92)

$state = $graphics.Save()
$graphics.TranslateTransform(32, $top + ($plotHeight / 2.0))
$graphics.RotateTransform(-90)
$yAxisLabel = "Cumulative Success Rate (C_k)"
$yAxisSize = $graphics.MeasureString($yAxisLabel, $axisFont)
$graphics.DrawString($yAxisLabel, $axisFont, $axisBrush, -($yAxisSize.Width / 2.0), -($yAxisSize.Height / 2.0))
$graphics.Restore($state)

$legendTitle = "Models"
$graphics.DrawString($legendTitle, $legendTitleFont, $titleBrush, $legendX, $legendY - 18)

for ($i = 0; $i -lt $records.Count; $i++) {
    $record = $records[$i]
    $color = Get-ColorByModelName $record.Model $i
    $linePen = New-Object System.Drawing.Pen ($color, 4.0)
    $linePen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
    $linePen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round
    $linePen.LineJoin = [System.Drawing.Drawing2D.LineJoin]::Round

    $pointBrush = New-Object System.Drawing.SolidBrush ($color)
    $whitePen = New-Object System.Drawing.Pen ([System.Drawing.Color]::White, 1.5)
    $connectorPen = New-Object System.Drawing.Pen ($color, 1.1)
    $connectorPen.Color = [System.Drawing.Color]::FromArgb(190, $color)
    $labelBorderPen = New-Object System.Drawing.Pen ($color, 1.2)

    $values = @($record.C1, $record.C2, $record.C3, $record.C4)
    $points = New-Object 'System.Collections.Generic.List[System.Drawing.PointF]'
    for ($stage = 1; $stage -le 4; $stage++) {
        $x = [float](Get-XOfStage $stage)
        $y = [float](Get-YOfValue $values[$stage - 1])
        $points.Add((New-Object System.Drawing.PointF($x, $y)))
    }
    $graphics.DrawLines($linePen, $points.ToArray())

    for ($stage = 1; $stage -le 4; $stage++) {
        $pointX = Get-XOfStage $stage
        $pointY = Get-YOfValue $values[$stage - 1]
        $graphics.FillEllipse($pointBrush, $pointX - 6.5, $pointY - 6.5, 13, 13)
        $graphics.DrawEllipse($whitePen, $pointX - 6.5, $pointY - 6.5, 13, 13)

        $offsetChoices = @($stageLabelOffsets[$stage])
        $offset = $offsetChoices[[Math]::Min($i, $offsetChoices.Count - 1)]
        $labelX = $pointX + [double]$offset.X
        $labelY = $pointY + [double]$offset.Y
        $graphics.DrawLine($connectorPen, $pointX, $pointY, $labelX, $labelY - 6)

        $boxWidth = 40.0
        $boxHeight = 20.0
        $boxPath = New-RoundedRectPath ($labelX - ($boxWidth / 2.0)) ($labelY - $boxHeight + 2.0) $boxWidth $boxHeight 8
        $graphics.FillPath($cardBrush, $boxPath)
        $graphics.DrawPath($labelBorderPen, $boxPath)

        $pctText = "{0:0}%" -f ($values[$stage - 1] * 100.0)
        $pctSize = $graphics.MeasureString($pctText, $labelFont)
        $textBrush = New-Object System.Drawing.SolidBrush ($color)
        $graphics.DrawString($pctText, $labelFont, $textBrush, $labelX - ($pctSize.Width / 2.0), $labelY - 18)
        $textBrush.Dispose()
        $boxPath.Dispose()
    }

    $entryY = $legendY + ($i * 74) + 24
    $graphics.DrawLine($linePen, $legendX, $entryY - 10, $legendX + 36, $entryY - 10)
    $graphics.FillEllipse($pointBrush, $legendX + 12.5, $entryY - 15.5, 11, 11)
    $graphics.DrawEllipse($whitePen, $legendX + 12.5, $entryY - 15.5, 11, 11)
    $graphics.DrawString($record.Model, $legendFont, $subtitleBrush, $legendX + 48, $entryY - 18)

    if ($null -ne $record.EarliestConvergeStage) {
        $decisionText = "first converge: Stage $($record.EarliestConvergeStage)"
    } else {
        $decisionText = "first converge: none"
    }
    $ratioText = if ($null -ne $record.Ratio) { "C3/C4={0:0.00}" -f $record.Ratio } else { "C3/C4=N/A" }
    $deltaText = if ($null -ne $record.Delta4) { "Delta4={0:0.00}" -f $record.Delta4 } else { "Delta4=N/A" }
    $graphics.DrawString("$ratioText, $deltaText", $legendFont, $subtitleBrush, $legendX + 48, $entryY)
    $graphics.DrawString($decisionText, $legendFont, $subtitleBrush, $legendX + 48, $entryY + 16)

    $linePen.Dispose()
    $pointBrush.Dispose()
    $whitePen.Dispose()
    $connectorPen.Dispose()
    $labelBorderPen.Dispose()
}

$footer = "Reading guide: earlier flattening suggests earlier saturation; the legend reports the first stage at which the remaining tail gain becomes negligible under the convergence rule."
$footerFormat = New-Object System.Drawing.StringFormat
$footerFormat.Alignment = [System.Drawing.StringAlignment]::Center
$footerRect = New-Object System.Drawing.RectangleF -ArgumentList 64.0, ($height - 58.0), ($width - 128.0), 24.0
$graphics.DrawString($footer, $subtitleFont, $subtitleBrush, $footerRect, $footerFormat)

[System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($pngPath)) | Out-Null
$bitmap.Save($pngPath, [System.Drawing.Imaging.ImageFormat]::Png)

$graphics.Dispose()
$bitmap.Dispose()
$backgroundBrush.Dispose()
$cardBrush.Dispose()
$cardPen.Dispose()
$gridPen.Dispose()
$axisPen.Dispose()
$tickPen.Dispose()
$titleFont.Dispose()
$subtitleFont.Dispose()
$axisFont.Dispose()
$tickFont.Dispose()
$labelFont.Dispose()
$legendTitleFont.Dispose()
$legendFont.Dispose()
$titleBrush.Dispose()
$subtitleBrush.Dispose()
$axisBrush.Dispose()
$tickBrush.Dispose()
$cardPath.Dispose()
$footerFormat.Dispose()

Write-Output "Generated: $pngPath"
