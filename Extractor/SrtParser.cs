using System.Text.RegularExpressions;
using Extractor.Structs;

namespace Extractor;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;

public partial class SrtParser
{
    public static List<DjiTelemetryData> Parse(string srtContent)
    {
        var telemetryList = new List<DjiTelemetryData>();

        // This single regex pattern captures all required fields from each SRT block.
        // It uses named capture groups for easy access.
        var pattern = SubtitleMetadataRegex();
        var matches = pattern.Matches(srtContent);

        const string timeFormat = @"hh\:mm\:ss\,fff";
        
        foreach (Match match in matches)
        {
            try
            {
                var telemetryData = new DjiTelemetryData
                {
                    StartTime = TimeSpan.ParseExact(match.Groups["startTime"].Value, timeFormat, CultureInfo.InvariantCulture),
                    EndTime = TimeSpan.ParseExact(match.Groups["endTime"].Value, timeFormat, CultureInfo.InvariantCulture),
                    
                    // Use InvariantCulture to ensure '.' is treated as the decimal separator
                    FrameCount = int.Parse(match.Groups["frame"].Value),
                    DiffTime = int.Parse(match.Groups["diff"].Value),
                    ISO = int.Parse(match.Groups["iso"].Value),
                    Shutter = match.Groups["shutter"].Value,
                    FNum = double.Parse(match.Groups["fnum"].Value, CultureInfo.InvariantCulture),
                    F_length = double.Parse(match.Groups["focal_len"].Value, CultureInfo.InvariantCulture),
                    Latitude = double.Parse(match.Groups["lat"].Value, CultureInfo.InvariantCulture),
                    Longitude = double.Parse(match.Groups["lon"].Value, CultureInfo.InvariantCulture),
                    RelativeAltitude = double.Parse(match.Groups["rel_alt"].Value, CultureInfo.InvariantCulture),
                    AbsoluteAltitude = double.Parse(match.Groups["abs_alt"].Value, CultureInfo.InvariantCulture)
                };
                telemetryList.Add(telemetryData);
            }
            catch (FormatException ex)
            {
                // Handle cases where a value might be malformed in the SRT file
                Console.WriteLine($"Skipping entry due to parsing error: {ex.Message}");
            }
        }

        return telemetryList;
    }

    
    [GeneratedRegex(
        "(?<startTime>\\d{2}:\\d{2}:\\d{2},\\d{3}) --> (?<endTime>\\d{2}:\\d{2}:\\d{2},\\d{3}).*?" + 
        "FrameCnt: (?<frame>\\d+), DiffTime: (?<diff>\\d+)ms.*?" + 
        "\\[iso: (?<iso>\\d+)\\] \\[shutter: (?<shutter>[\\d\\./]+)\\] \\[fnum: (?<fnum>[\\d\\.]+)\\].*?" + 
        "\\[focal_len: (?<focal_len>[\\d\\.]+)\\].*?" + 
        "\\[latitude: (?<lat>[\\d\\.-]+)\\] \\[longitude: (?<lon>[\\d\\.-]+)\\].*?" + 
        "\\[rel_alt: (?<rel_alt>[\\d\\.-]+) abs_alt: (?<abs_alt>[\\d\\.-]+)\\]", 
        RegexOptions.Singleline
    )]
    private static partial Regex SubtitleMetadataRegex();
}