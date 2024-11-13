using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Extractor.Commands;
using Extractor.enums;
using Spectre.Console;
using Spectre.Console.Advanced;
using TreeBasedCli;


namespace Extractor.Handlers;

public class
    NormalizeLuminanceCommandHandler : ILeafCommandHandler<NormalizeLuminanceCommand.NormalizeLuminanceArguments>
{
    public async Task HandleAsync(NormalizeLuminanceCommand.NormalizeLuminanceArguments arguments,
        LeafCommand executedCommand)
    {
        var inputDirectory = arguments.InputDir;
        var outputDirectory = arguments.OutputDir;

        bool globalAverage = arguments.GlobalAverage;

        int windowSize = arguments.windowSize;

        int maxConcurrentTasks = arguments.MaxConcurrentTasks;

        if (!Directory.Exists(inputDirectory))
        {
            AnsiConsole.Console.WriteAnsi($"[red]Error: Input directory {inputDirectory} does not exist.[/]");
            return;
        }

        double gamma = arguments.Gamma;

        double clipLimit = arguments.ClipLimit;

        int kernel = arguments.KernelSize;

        double headroom = arguments.Headroom;

        ColorSpace colorSpace = arguments.colorSpace;
        
        bool useCIE = colorSpace == ColorSpace.LAB;


        Directory.CreateDirectory(outputDirectory);


        // read all files in the input directory
        // for each file, read the image, convert to HSV, and calculate the average luminance

        var files = Constants.SupportedExtensions
            .SelectMany(ext => Directory.GetFiles(inputDirectory, ext, SearchOption.AllDirectories)).ToArray();

        var luminanceValues =
            await AnsiConsole.Progress()
                .StartAsync(async ctx =>
                {
                    using var semaphore = new SemaphoreSlim(maxConcurrentTasks);
                    var task = ctx.AddTask($"[yellow]Extracting luminance values from images in {colorSpace.ToString()} color space...[/]",
                        maxValue: files.Length);

                    var luminanceTasks = files.Select(async file =>
                    {
                        await semaphore.WaitAsync();
                        try
                        {
                            var luminance = await Task.Run(() =>
                                new ImageLuminanceRecord(file, CalculateAverageImageLuminance(file, useCIE)));
                            task.Increment(1);
                            return luminance;
                        }
                        finally
                        {
                            semaphore.Release();
                        }
                    }).ToArray();

                    return await Task.WhenAll(luminanceTasks);
                });


        var luminanceValuesList = luminanceValues.Where(record => record.Luminance[1] > 0).ToList();


        luminanceValuesList.Sort((a, b) => new NaturalSortComparer().Compare(a.FilePath, b.FilePath));


        var averageLuminance = new LuminanceValues();

        if (globalAverage)
        {
            var avgVariance = luminanceValuesList.Average(record => Math.Pow(record.Luminance.StdDev, 2));
            var avgStdDev = Math.Sqrt(avgVariance);

            averageLuminance = new LuminanceValues(
                luminanceValues.Average(record => record.Luminance.Min),
                luminanceValues.Average(record => record.Luminance.Max),
                luminanceValues.Average(record => record.Luminance.Mean),
                avgStdDev
            );
        }
        else
        {
            var rollingAverages = CalculateRollingAverage(luminanceValuesList, windowSize);

            for (var i = 0; i < luminanceValuesList.Count; i++)
            {
                luminanceValuesList[i] = new ImageLuminanceRecord(luminanceValuesList[i].FilePath, rollingAverages[i]);
            }
        }


        await AnsiConsole.Progress().StartAsync(async ctx =>
        {
            var task = ctx.AddTask("[yellow]Normalizing luminance and saving to disk...[/]", maxValue: files.Length);
            var normalizationTasks = luminanceValues.Select(async record =>
            {
                await Task.Run(() =>
                {
                    NormalizeImageExposure(record.FilePath, outputDirectory,
                        globalAverage ? averageLuminance : record.Luminance, useCIE, gamma, clipLimit, kernel, headroom);

                    return true;
                });
                task.Increment(1);
            }).ToArray();

            await Task.WhenAll(normalizationTasks);
        });
    }


    private static LuminanceValues CalculateAverageImageLuminance(string imagePath, bool useCie)
    {
        try
        {
            Mat image = CvInvoke.Imread(imagePath);

            Mat luminanceChannel;

            if (useCie)
            {
                // Convert to CIE L*a*b* color space and get the L* channel
                Mat labImage = new Mat();
                CvInvoke.CvtColor(image, labImage, ColorConversion.Bgr2Lab);
                Mat[] labChannels = labImage.Split();
                luminanceChannel = labChannels[0]; // L* channel

                labImage.Dispose();
            }
            else
            {
                // Convert to HSV color space and get the Value (V) channel
                Mat hsvImage = new Mat();
                CvInvoke.CvtColor(image, hsvImage, ColorConversion.Bgr2Hsv);
                Mat[] hsvChannels = hsvImage.Split();
                luminanceChannel = hsvChannels[2]; // V channel

                hsvImage.Dispose();
            }

            var luminance = GetCurrentLuminance(luminanceChannel);

            image.Dispose();
            luminanceChannel.Dispose();

            return luminance;
        }
        catch (Exception ex)
        {
            AnsiConsole.Console.WriteLine(
                $"[red]Warning: Error processing image at {imagePath}: {ex.Message}. Skipping this image.[/]");
            return new LuminanceValues();
        }
    }


    static bool NormalizeImageExposure(string imagePath, string outputPath, LuminanceValues targetLuminance, bool useCIE,
        double gamma = 0.5, double clipLimit = 10, int kernel = 8, double headroom = 0.2)
    {
        try
        {
            Mat image = CvInvoke.Imread(imagePath);

            Mat luminanceChannel;
            Mat colorSpaceImage = new Mat();

            if (useCIE)
            {
                // Convert to CIE L*a*b* and get the L* channel
                CvInvoke.CvtColor(image, colorSpaceImage, ColorConversion.Bgr2Lab);
                Mat[] labChannels = colorSpaceImage.Split();
                luminanceChannel = labChannels[0]; // L* channel
            }
            else
            {
                // Convert to HSV and get the V channel
                CvInvoke.CvtColor(image, colorSpaceImage, ColorConversion.Bgr2Hsv);
                Mat[] hsvChannels = colorSpaceImage.Split();
                luminanceChannel = hsvChannels[2]; // V channel
            }

            // Apply gamma correction, CLAHE, and normalization
            luminanceChannel = ApplyGammaCorrection(luminanceChannel, gamma, DepthType.Cv16U);
            System.Drawing.Size tileGridSize = new System.Drawing.Size(kernel, kernel);
            CvInvoke.CLAHE(luminanceChannel, clipLimit, tileGridSize, luminanceChannel);

            var currentLuminance = GetCurrentLuminance(luminanceChannel);
            luminanceChannel.ConvertTo(luminanceChannel, DepthType.Cv32F);
            luminanceChannel -= currentLuminance.Mean;
            const double epsilon = 1e-6;
            var scale = targetLuminance.StdDev / (currentLuminance.StdDev + epsilon);
            luminanceChannel *= scale;
            luminanceChannel += targetLuminance.Max;
            CvInvoke.Normalize(luminanceChannel, luminanceChannel, 0, 255, NormType.MinMax);
            luminanceChannel.ConvertTo(luminanceChannel, DepthType.Cv8U);

            // Merge back the channels and convert back to BGR
            if (useCIE)
            {
                CvInvoke.Merge(
                    new VectorOfMat(luminanceChannel, colorSpaceImage.Split()[1], colorSpaceImage.Split()[2]),
                    colorSpaceImage);
                CvInvoke.CvtColor(colorSpaceImage, image, ColorConversion.Lab2Bgr);
            }
            else
            {
                CvInvoke.Merge(
                    new VectorOfMat(colorSpaceImage.Split()[0], colorSpaceImage.Split()[1], luminanceChannel),
                    colorSpaceImage);
                CvInvoke.CvtColor(colorSpaceImage, image, ColorConversion.Hsv2Bgr);
            }

            var outputFilePath = Path.Combine(outputPath, Path.GetFileName(imagePath));
            CvInvoke.Imwrite(outputFilePath, image);

            return true;
        }
        catch (Exception ex)
        {
            AnsiConsole.Console.WriteLine(
                $"[red]Warning: Error processing image at {imagePath}: {ex.Message}. Skipping this image.[/]");
            return false;
        }
    }


    private static LuminanceValues GetCurrentLuminance(Mat valueChannel)
    {
        MCvScalar mean = new(0), stddev = new(0);
        CvInvoke.MeanStdDev(valueChannel, ref mean, ref stddev);


        double currentMin = 0, currentMax = 0;
        System.Drawing.Point minLoc = new(), maxLoc = new();
        CvInvoke.MinMaxLoc(valueChannel, ref currentMin, ref currentMax, ref minLoc, ref maxLoc);


        return new LuminanceValues(currentMin, currentMax, mean.V0, stddev.V0);
    }


    private static Mat ApplyGammaCorrection(Mat src, double gamma, DepthType depthType = DepthType.Cv8U)
    {
        Mat srcFloat = new Mat();
        Mat dst = new Mat();

        switch (depthType)
        {
            // Convert source image to a 32-bit float and normalize based on depth type
            case DepthType.Cv8U:
                src.ConvertTo(srcFloat, DepthType.Cv32F, 1.0 / 255.0); // Normalize 8-bit to [0, 1]
                break;
            case DepthType.Cv16U:
                src.ConvertTo(srcFloat, DepthType.Cv32F, 1.0 / 65535.0); // Normalize 16-bit to [0, 1]
                break;
            case DepthType.Cv32F:
                // If the image is already 32F, no scaling needed, just copy to srcFloat
                srcFloat = src.Clone();
                break;
            default:
                throw new ArgumentException("Unsupported depth type for gamma correction.");
        }

        // Apply gamma correction (element-wise power transformation)
        CvInvoke.Pow(srcFloat, gamma, srcFloat); // Applies the power transformation to each pixel

        switch (depthType)
        {
            // Convert back to the original depth type and scale accordingly
            case DepthType.Cv8U:
                srcFloat.ConvertTo(dst, DepthType.Cv8U, 255.0); // Scale back to [0, 255] for 8-bit
                break;
            case DepthType.Cv16U:
                srcFloat.ConvertTo(dst, DepthType.Cv16U, 65535.0); // Scale back to [0, 65535] for 16-bit
                break;
            case DepthType.Cv32F:
                dst = srcFloat.Clone(); // No scaling needed for 32F; just copy the result
                break;
            default:
                throw new ArgumentException("Unsupported depth type for gamma correction.");
        }

        return dst;
    }


    private static LuminanceValues[] CalculateRollingAverage(IReadOnlyList<ImageLuminanceRecord> luminanceValues,
        int windowSize)
    {
        var n = luminanceValues.Count;
        var rollingAverages = new LuminanceValues[n];

        // Precompute cumulative sums and cumulative sum of squares
        var cumSumMin = new double[n + 1];
        var cumSumMax = new double[n + 1];
        var cumSumMean = new double[n + 1];
        var cumSumMeanSquares = new double[n + 1];

        for (var i = 0; i < n; i++)
        {
            var currentLuminance = luminanceValues[i].Luminance;
            cumSumMin[i + 1] = cumSumMin[i] + currentLuminance.Min;
            cumSumMax[i + 1] = cumSumMax[i] + currentLuminance.Max;
            cumSumMean[i + 1] = cumSumMean[i] + currentLuminance.Mean;
            cumSumMeanSquares[i + 1] = cumSumMeanSquares[i] + currentLuminance.Mean * currentLuminance.Mean;
        }

        // Determine the start index for the fixed window at the end
        var fixedWindowStartIndex = Math.Max(0, n - windowSize);
        var fixedWindowSize = n - fixedWindowStartIndex;

        for (var i = 0; i < n; i++)
        {
            int windowStartIndex, windowEndIndex, effectiveWindowSize;

            if (i + windowSize <= n)
            {
                // Normal case: window moves forward
                windowStartIndex = i;
                windowEndIndex = i + windowSize;
                effectiveWindowSize = windowSize;
            }
            else
            {
                // Near the end: use fixed window
                windowStartIndex = fixedWindowStartIndex;
                windowEndIndex = n;
                effectiveWindowSize = fixedWindowSize;
            }

            var sumMin = cumSumMin[windowEndIndex] - cumSumMin[windowStartIndex];
            var sumMax = cumSumMax[windowEndIndex] - cumSumMax[windowStartIndex];
            var sumMean = cumSumMean[windowEndIndex] - cumSumMean[windowStartIndex];
            var sumMeanSquares = cumSumMeanSquares[windowEndIndex] - cumSumMeanSquares[windowStartIndex];

            var avgMin = sumMin / effectiveWindowSize;
            var avgMax = sumMax / effectiveWindowSize;
            var avgMean = sumMean / effectiveWindowSize;

            var variance = (sumMeanSquares / effectiveWindowSize) - (avgMean * avgMean);
            variance = Math.Max(variance, 0);
            var stdDev = Math.Sqrt(variance);

            rollingAverages[i] = new LuminanceValues(avgMin, avgMax, avgMean, stdDev);
        }

        return rollingAverages;
    }
}

public struct ImageLuminanceRecord(string filePath, LuminanceValues luminance)
{
    public string FilePath { get; set; } = filePath;
    public LuminanceValues Luminance { get; set; } = luminance;
}

public struct LuminanceValues(double min, double max, double mean, double stdDev)
{
    public double Min { get; init; } = min;
    public double Max { get; init; } = max;
    public double Mean { get; init; } = mean;

    public double StdDev { get; init; } = stdDev;

    public double this[int i]
    {
        get
        {
            return i switch
            {
                0 => Min,
                1 => Max,
                2 => Mean,
                3 => StdDev,
                _ => 0
            };
        }
    }
}