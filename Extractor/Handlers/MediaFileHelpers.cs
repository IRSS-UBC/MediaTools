using System.Drawing;
using Extractor.Extensions;
using FFMediaToolkit.Decoding;
using Spectre.Console;
using Image = SixLabors.ImageSharp.Image;

namespace Extractor.Handlers;

public static class MediaFileHelpers
{
    public static void PrintFileInfo(this MediaFile file)
    {

        var info = file.Video.Info;

        AnsiConsole.Write(new TextPath(file.Info.FilePath));

        AnsiConsole.WriteLine(Resources.Resources.InfoLine1, info.NumberOfFrames ?? 0, info.FrameSize, info.IsVariableFrameRate, info.AvgFrameRate);
        AnsiConsole.WriteLine(Resources.Resources.InfoLine2, info.Rotation, info.IsInterlaced, info.PixelFormat, info.Duration, info.CodecName);


        AnsiConsole.Write(Resources.Resources.MetadataHeader);
        foreach (var pair in info.Metadata)
        {
            AnsiConsole.Write(Resources.Resources.MetadataPair, pair.Key, pair.Value);
            AnsiConsole.Write(@" ");
        }
    }
    

    public readonly struct TimestampedFrame : IDisposable
    {
        public Image Bitmap { get; }
        public TimeSpan Timestamp { get; }

        public TimestampedFrame(Image bitmap, TimeSpan timestamp)
        {
            Bitmap = bitmap;
            Timestamp = timestamp;
        }

        public void Dispose() => Bitmap.Dispose();
    }

    public static IEnumerable<Image> GetFramesOld(this MediaFile mediaFile, float dropRatio)
    {
        switch (dropRatio)
        {
            case < 0 or > 1:
                throw new ArgumentException(Resources.Resources.DropRatioRequirements, nameof(dropRatio));
            case >= 1:
                yield break;
        }

        var keepFrameNumber = dropRatio > 0 ? 1 / (1 - dropRatio) : 1;
        
        uint frameCounter = 0;
        while (mediaFile.Video.TryGetNextFrame(out var imageData))
        {
            frameCounter++;

            if (dropRatio < 1 && frameCounter % (int)keepFrameNumber != 0)
                continue;

            yield return imageData.ToBitmap();
        }
    }
    
    // Returns TimestampedFrame where Timestamp is the requested time (exact).
    public static IEnumerable<TimestampedFrame> GetFrames(this MediaFile mediaFile, float dropRatio)
    {
        if (dropRatio is < 0 or >= 1)
            throw new ArgumentException("Drop ratio must be between 0 (none) and 1 (drop all).", nameof(dropRatio));

        var videoInfo = mediaFile.Video.Info;
        double srcFps = videoInfo.AvgFrameRate;
        if (srcFps <= 0)
            srcFps = 30.0; // fallback

        double targetFps = srcFps * (1 - dropRatio);
        if (targetFps <= 0)
            yield break;

        var targetInterval = TimeSpan.FromSeconds(1.0 / targetFps);
        var duration = videoInfo.Duration;

        for (var t = TimeSpan.Zero; t < duration; t += targetInterval)
        {
            // Acquire the unmanaged frame
            var imageData = mediaFile.Video.GetFrame(t);

            // Convert immediately to a managed Bitmap
            var bitmap = imageData.ToBitmap();
            
            var frame = new TimestampedFrame(bitmap, t);
            


            // Now yield the safe managed object
            yield return frame;
        }
    }
}
