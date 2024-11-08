using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Extractor.Commands;
using Spectre.Console;
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

        bool globalAverage = false;


        Directory.CreateDirectory(outputDirectory);


        // read all files in the input directory
        // for each file, read the image, convert to HSV, and calculate the average luminance

        var files = Constants.SupportedExtensions
            .SelectMany(ext => Directory.GetFiles(inputDirectory, ext, SearchOption.AllDirectories)).ToArray();

        var luminanceValues =
            await AnsiConsole.Progress()
                .StartAsync(async ctx =>
                {
                    int maxConcurrentTasks = Environment.ProcessorCount; // Set to number of logical cores

                    using var semaphore = new SemaphoreSlim(maxConcurrentTasks);
                    var task = ctx.AddTask("[yellow]Extracting luminance values from images...[/]",
                        maxValue: files.Length);

                    var luminanceTasks = files.Select(async file =>
                    {
                        await semaphore.WaitAsync();
                        try
                        {
                            var luminance = await Task.Run(() =>
                                new ImageLuminanceRecord(file, CalculateAverageImageLuminance(file)));
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
            var rollingAverages = CalculateRollingAverage(luminanceValuesList, 10);

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
                        globalAverage ? averageLuminance : record.Luminance);

                    return true;
                });
                task.Increment(1);
            }).ToArray();

            await Task.WhenAll(normalizationTasks);
        });
    }


    private static LuminanceValues CalculateAverageImageLuminance(string imagePath)
    {
        try
        {
            Mat image = CvInvoke.Imread(imagePath);

            Mat hsvImage = new Mat();
            CvInvoke.CvtColor(image, hsvImage, ColorConversion.Bgr2Hsv);

            Mat[] hsvChannels = hsvImage.Split();
            Mat valueChannel = hsvChannels[2];

            // CvInvoke.Normalize(valueChannel, valueChannel, 0, 255, NormType.MinMax);


            var value = GetCurrentLuminance(valueChannel);


            image.Dispose();
            hsvImage.Dispose();
            valueChannel.Dispose();

            return value;
        }
        catch (Exception ex)
        {
            AnsiConsole.Console.WriteLine(
                $"[red]Warning: Error processing image at {imagePath}: {ex.Message}. Skipping this image.[/]");
            return new LuminanceValues();
        }
    }


    static bool NormalizeImageExposure(string imagePath, string outputPath, LuminanceValues targetLuminance)
    {
        try
        {
            Mat image = CvInvoke.Imread(imagePath);


            // Convert the image to HSV color space
            var hsvImage = new Mat();
            CvInvoke.CvtColor(image, hsvImage, ColorConversion.Bgr2Hsv);

            // Split the HSV image into separate channels (Hue, Saturation, and Value)
            var hsvChannels = new VectorOfMat();
            CvInvoke.Split(hsvImage, hsvChannels);
            var valueChannel = hsvChannels[2]; // Value channel represents brightness


            // Apply Gamma Correction to boost dark areas
            double gamma = 0.5; // Lower values (<1) lighten dark areas, while values >1 darken


            var workingImage = new Mat();

            double
                alpha = 256 -
                        56; // normally alpha would be 256, but we're using 200 to avoid clipping in further processing

            valueChannel.ConvertTo(workingImage, DepthType.Cv16U, alpha);

            valueChannel = workingImage;


            valueChannel = ApplyGammaCorrection(valueChannel, gamma, DepthType.Cv16U);


            // Apply CLAHE to the Value channel
            double clipLimit = 10;
            System.Drawing.Size tileGridSize = new System.Drawing.Size(8, 8);
            CvInvoke.CLAHE(valueChannel, clipLimit, tileGridSize, valueChannel);


            // Calculate current luminance statistics (min, max, mean) of the value channel
            var currentLuminance = GetCurrentLuminance(valueChannel);

            //Console.WriteLine($"Current Luminance: Min={currentLuminance.Min}, Max={currentLuminance.Max}, Mean={currentLuminance.Mean}");


            // Calculate the scaling factor for the value channel


            valueChannel.ConvertTo(valueChannel, DepthType.Cv32F);


            valueChannel -= currentLuminance.Mean;
            

            double epsilon = 1e-6; // Small value to prevent division by zero
            double scale = targetLuminance.StdDev / (currentLuminance.StdDev + epsilon);
            
            
            valueChannel *= scale;

            valueChannel += targetLuminance.Max;


            // normalize the value channel to 0-255
            CvInvoke.Normalize(valueChannel, valueChannel, 0, 255, NormType.MinMax);


            valueChannel.ConvertTo(valueChannel, DepthType.Cv8U);


            // Merge the modified value channel back with the other HSV channels
            hsvChannels = new VectorOfMat(hsvChannels[0], hsvChannels[1], valueChannel);
            CvInvoke.Merge(hsvChannels, hsvImage);

            // Convert back to BGR color space
            var normalizedImage = new Mat();
            CvInvoke.CvtColor(hsvImage, normalizedImage, ColorConversion.Hsv2Bgr);


            // Save the normalized image to the output path with the same file name
            var outputFilePath = Path.Combine(outputPath, Path.GetFileName(imagePath));
            CvInvoke.Imwrite(outputFilePath, normalizedImage);


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


    static Mat ApplyGammaCorrection(Mat src, double gamma, DepthType depthType = DepthType.Cv8U)
    {
        Mat srcFloat = new Mat();
        Mat dst = new Mat();

        // Step 1: Convert source image to a 32-bit float and normalize based on depth type
        if (depthType == DepthType.Cv8U)
        {
            src.ConvertTo(srcFloat, DepthType.Cv32F, 1.0 / 255.0); // Normalize 8-bit to [0, 1]
        }
        else if (depthType == DepthType.Cv16U)
        {
            src.ConvertTo(srcFloat, DepthType.Cv32F, 1.0 / 65535.0); // Normalize 16-bit to [0, 1]
        }
        else if (depthType == DepthType.Cv32F)
        {
            // If the image is already 32F, no scaling needed, just copy to srcFloat
            srcFloat = src.Clone();
        }
        else
        {
            throw new ArgumentException("Unsupported depth type for gamma correction.");
        }

        // Step 2: Apply gamma correction (element-wise power transformation)
        CvInvoke.Pow(srcFloat, gamma, srcFloat); // Applies the power transformation to each pixel

        // Step 3: Convert back to the original depth type and scale accordingly
        if (depthType == DepthType.Cv8U)
        {
            srcFloat.ConvertTo(dst, DepthType.Cv8U, 255.0); // Scale back to [0, 255] for 8-bit
        }
        else if (depthType == DepthType.Cv16U)
        {
            srcFloat.ConvertTo(dst, DepthType.Cv16U, 65535.0); // Scale back to [0, 65535] for 16-bit
        }
        else if (depthType == DepthType.Cv32F)
        {
            dst = srcFloat.Clone(); // No scaling needed for 32F; just copy the result
        }

        return dst;
    }


    private LuminanceValues[] CalculateRollingAverage(IReadOnlyList<ImageLuminanceRecord> luminanceValues,
        int windowSize)
    {
        int n = luminanceValues.Count;
        var rollingAverages = new LuminanceValues[n];

        // Precompute cumulative sums and cumulative sum of squares
        var cumSumMin = new double[n + 1];
        var cumSumMax = new double[n + 1];
        var cumSumMean = new double[n + 1];
        var cumSumMeanSquares = new double[n + 1];

        for (int i = 0; i < n; i++)
        {
            var currentLuminance = luminanceValues[i].Luminance;
            cumSumMin[i + 1] = cumSumMin[i] + currentLuminance.Min;
            cumSumMax[i + 1] = cumSumMax[i] + currentLuminance.Max;
            cumSumMean[i + 1] = cumSumMean[i] + currentLuminance.Mean;
            cumSumMeanSquares[i + 1] = cumSumMeanSquares[i] + currentLuminance.Mean * currentLuminance.Mean;
        }

        // Determine the start index for the fixed window at the end
        int fixedWindowStartIndex = Math.Max(0, n - windowSize);
        int fixedWindowSize = n - fixedWindowStartIndex;

        for (int i = 0; i < n; i++)
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

            double sumMin = cumSumMin[windowEndIndex] - cumSumMin[windowStartIndex];
            double sumMax = cumSumMax[windowEndIndex] - cumSumMax[windowStartIndex];
            double sumMean = cumSumMean[windowEndIndex] - cumSumMean[windowStartIndex];
            double sumMeanSquares = cumSumMeanSquares[windowEndIndex] - cumSumMeanSquares[windowStartIndex];

            double avgMin = sumMin / effectiveWindowSize;
            double avgMax = sumMax / effectiveWindowSize;
            double avgMean = sumMean / effectiveWindowSize;

            double variance = (sumMeanSquares / effectiveWindowSize) - (avgMean * avgMean);
            variance = Math.Max(variance, 0);
            double stdDev = Math.Sqrt(variance);

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