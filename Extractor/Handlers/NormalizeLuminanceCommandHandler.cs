using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using ExifLibrary;
using Extractor.Algorithms;
using Extractor.Commands;
using Extractor.Extensions;
using Extractor.Structs;
using Spectre.Console;
using Spectre.Console.Advanced;
using TreeBasedCli;
using ColorSpace = Extractor.enums.ColorSpace;


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
            AnsiConsole.Console.MarkupLine($"[red]Error: Input directory {inputDirectory} does not exist.[/]");
            return;
        }

        double gamma = arguments.Gamma;

        double clipLimit = arguments.ClipLimit;

        int kernel = arguments.KernelSize;

        double headroom = arguments.Headroom;

        ColorSpace colorSpace = arguments.colorSpace;

        bool useCie = colorSpace == ColorSpace.LAB;

        bool optimalGamma = arguments.optimalGammaCorrection;


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
                    var task = ctx.AddTask(
                        $"[yellow]Extracting luminance values from images in {colorSpace.ToString()} color space...[/]",
                        maxValue: files.Length);

                    var luminanceTasks = files.Select(async file =>
                    {
                        await semaphore.WaitAsync();
                        try
                        {
                            var luminance = await Task.Run(() =>
                                new ImageLuminanceRecord(file, CalculateAverageImageLuminance(file, useCie)));
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


        GC.Collect();


        await AnsiConsole.Progress().StartAsync(async ctx =>
        {
            var task = ctx.AddTask("[yellow]Normalizing luminance and saving to disk...[/]", maxValue: files.Length);

            const int batchSize = 50;
            // Adjust the degree of parallelism as needed

            var tasks = new List<Task>();

            // Create a SemaphoreSlim for managing concurrency
            using var semaphore = new SemaphoreSlim(maxConcurrentTasks);

            for (var i = 0; i < luminanceValues.Length; i += batchSize)
            {
                var batch = luminanceValues.Skip(i).Take(batchSize).ToList();

                // Add the batch processing task
                tasks.Add(Task.Run(async () =>
                {
                    foreach (var record in batch)
                    {
                        await semaphore.WaitAsync();
                        try
                        {
                            var outputFilePath = Path.Combine(outputDirectory, Path.GetFileName(record.FilePath));

                            await NormalizeImageExposure(record.FilePath, outputFilePath,
                                globalAverage ? averageLuminance : record.Luminance, useCie, optimalGamma, gamma,
                                clipLimit,
                                kernel, headroom);
                            
                            
                            await TransferExifTagsAsync(record.FilePath, outputFilePath);
                            task.Increment(1);
                        }
                        finally
                        {
                            semaphore.Release();
                        }
                    }
                }));
            }

            // Wait for all tasks to complete
            await Task.WhenAll(tasks);

            return Task.CompletedTask;
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

            var luminance = LuminanceValues.GetCurrentLuminance(luminanceChannel);

            image.Dispose();
            luminanceChannel.Dispose();

            return luminance;
        }
        catch (Exception ex)
        {
            AnsiConsole.Console.WriteAnsi(
                $"[red]Warning: Error processing image at {imagePath}: {ex.Message}. Skipping this image.[/]");
            return new LuminanceValues();
        }
    }


    private static async Task<bool> NormalizeImageExposure(string imagePath, string outputFilePath,
        LuminanceValues targetLuminance,
        bool useLab, bool useOptimalGamma, double gamma = 0.5, double clipLimit = 10, int kernel = 8,
        double headroom = 0.2)
    {
        Mat image = CvInvoke.Imread(imagePath);


        if (useOptimalGamma)
        {
            image = OGGCPEAlgorithm.Apply(image);
        }

        Mat luminanceChannel;
        Mat colorSpaceImage = new Mat();

        if (useLab)
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

        if (!useOptimalGamma)
        {
            luminanceChannel = ApplyGammaCorrection(luminanceChannel, gamma, DepthType.Cv16U);
        }


        var tileGridSize = new System.Drawing.Size(kernel, kernel);
        CvInvoke.CLAHE(luminanceChannel, clipLimit, tileGridSize, luminanceChannel);

        var currentLuminance = LuminanceValues.GetCurrentLuminance(luminanceChannel);
        luminanceChannel.ConvertTo(luminanceChannel, DepthType.Cv32F);
        luminanceChannel -= currentLuminance.Mean;
        const double epsilon = 1e-6;

        var scale = targetLuminance.StdDev / (currentLuminance.StdDev + epsilon);
        luminanceChannel *= scale;
        luminanceChannel += targetLuminance.Max;


        // Merge back the channels and convert back to BGR
        CvInvoke.Normalize(luminanceChannel, luminanceChannel, 0, 255, NormType.MinMax);
        luminanceChannel.ConvertTo(luminanceChannel, DepthType.Cv8U);

        if (useLab)
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


        var parameters = Array.Empty<KeyValuePair<ImwriteFlags, int>>();

        await using var buf = new VectorOfByte();
        CvInvoke.Imencode(new FileInfo(outputFilePath).Extension, image, buf, parameters);
        var array = buf.ToArray();
        await File.WriteAllBytesAsync(outputFilePath, array);

        return true;
    }


    private static Mat ApplyGammaCorrection(Mat srcFloat, double gamma, DepthType depthType)
    {
        // convert the luminance channel to 32bit float 
        srcFloat = srcFloat.ConvertScaleTo(DepthType.Cv32F);

        // Apply gamma correction (element-wise power transformation)
        CvInvoke.Pow(srcFloat, gamma, srcFloat); // Applies the power transformation to each pixel

        // Normalize the luminance channel back to to DepthType

        return srcFloat.ConvertScaleTo(depthType);
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
    
    
    private async Task TransferExifTagsAsync(string sourceFilePath, string destFilePath)
    {
        var sourceFile = await ImageFile.FromFileAsync(sourceFilePath);

        var destinationFile = await ImageFile.FromFileAsync(destFilePath);


        foreach (ExifProperty property in sourceFile.Properties)
        {
            destinationFile.Properties.Set(property);
        }

        await destinationFile.SaveAsync(destFilePath);
    }
}

public struct ImageLuminanceRecord(string filePath, LuminanceValues luminance)
{
    public string FilePath { get; set; } = filePath;
    public LuminanceValues Luminance { get; set; } = luminance;
}


