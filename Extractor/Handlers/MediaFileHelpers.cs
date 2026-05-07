using System.Drawing;
using Extractor.Extensions;
using FFMediaToolkit.Decoding;
using FFMediaToolkit.Graphics;
using Spectre.Console;
using Image = SixLabors.ImageSharp.Image;

namespace Extractor.Handlers;

public static class MediaFileHelpers
{
    public readonly struct TimestampedFrame(Image bitmap, TimeSpan timestamp) : IDisposable
    {
        public Image Bitmap { get; } = bitmap;
        public TimeSpan Timestamp { get; } = timestamp;

        public void Dispose() => Bitmap.Dispose();
    }

    extension(MediaFile mediaFile)
    {
        public IEnumerable<Image> GetFramesOld(float dropRatio)
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

        public IEnumerable<TimestampedFrame> GetFrames(float targetFps,
            double startSeconds = 0, double? stopSeconds = null)
        {
            var videoInfo = mediaFile.Video.Info;
   
            if (startSeconds < 0)
                throw new ArgumentException("Start position must be zero or greater.", nameof(startSeconds));

            if (stopSeconds <= startSeconds)
                throw new ArgumentException("End position must be strictly greater than start position.",
                    nameof(stopSeconds));

            var targetInterval = TimeSpan.FromSeconds(1.0 / targetFps);
            var duration = videoInfo.Duration;

            var startTime = TimeSpan.FromSeconds(startSeconds);

            var endTime = duration;
            if (stopSeconds.HasValue)
            {
                var requestedEndTime = TimeSpan.FromSeconds(stopSeconds.Value);
                endTime = requestedEndTime < duration ? requestedEndTime : duration;
            }

            if (startTime >= duration)
                yield break;

            for (var t = startTime; t < endTime; t += targetInterval)
            {
                ImageData imageData;
                try
                {
                    imageData = mediaFile.Video.GetFrame(t);
                }
                catch (EndOfStreamException)
                {
                  
                    yield break; 
                }

                var bitmap = imageData.ToBitmap();

                var frame = new TimestampedFrame(bitmap, t);

                yield return frame;
            }
        }

        public void PrintFileInfo()
        {
            var info = mediaFile.Video.Info;

            AnsiConsole.Write(new TextPath(mediaFile.Info.FilePath));

            AnsiConsole.WriteLine(Resources.Resources.InfoLine1, info.NumberOfFrames ?? 0, info.FrameSize,
                info.IsVariableFrameRate, info.AvgFrameRate);
            AnsiConsole.WriteLine(Resources.Resources.InfoLine2, info.Rotation, info.IsInterlaced, info.PixelFormat,
                info.Duration, info.CodecName);


            AnsiConsole.Write(Resources.Resources.MetadataHeader);
            foreach (var pair in info.Metadata)
            {
                AnsiConsole.Write(Resources.Resources.MetadataPair, pair.Key, pair.Value);
                AnsiConsole.Write(@" ");
            }
        }
    }
}
