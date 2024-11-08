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


        var averageLuminance = new MinMaxMean();

        if (globalAverage)
        {
            averageLuminance = new MinMaxMean(
                luminanceValues.Average(record => record.Luminance.Min),
                luminanceValues.Average(record => record.Luminance.Max),
                luminanceValues.Average(record => record.Luminance.Mean)
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


    private static MinMaxMean CalculateAverageImageLuminance(string imagePath)
    {
        try
        {
            Mat image = CvInvoke.Imread(imagePath);

            Mat hsvImage = new Mat();
            CvInvoke.CvtColor(image, hsvImage, ColorConversion.Bgr2Hsv);

            Mat[] hsvChannels = hsvImage.Split();
            Mat valueChannel = hsvChannels[2];

            // CvInvoke.Normalize(valueChannel, valueChannel, 0, 255, NormType.MinMax);


            var meanValue = CvInvoke.Mean(valueChannel);


            double minValue = 0, maxValue = 0;
            System.Drawing.Point minLoc = new(), maxLoc = new();
            CvInvoke.MinMaxLoc(valueChannel, ref minValue, ref maxValue, ref minLoc, ref maxLoc);


            image.Dispose();
            hsvImage.Dispose();

            valueChannel.Dispose();

            return new MinMaxMean
            {
                Min = minValue,
                Max = maxValue,
                Mean = meanValue.V0
            };
        }
        catch (Exception ex)
        {
            AnsiConsole.Console.WriteLine(
                $"[red]Warning: Error processing image at {imagePath}: {ex.Message}. Skipping this image.[/]");
            return new MinMaxMean();
        }
    }


    static bool NormalizeImageExposure(string imagePath, string outputPath, MinMaxMean targetLuminance)
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

            double alpha = 256 - 56; // normally alpha would be 256, but we're using 200 to avoid clipping in further processing
            
            valueChannel.ConvertTo(workingImage, DepthType.Cv16U, alpha); 
            
            valueChannel = workingImage;

            
            valueChannel = ApplyGammaCorrection(valueChannel, gamma, DepthType.Cv16U);

        



            // Apply CLAHE to the Value channel
            double clipLimit = 10; 
            System.Drawing.Size tileGridSize = new System.Drawing.Size(8, 8);
            CvInvoke.CLAHE(valueChannel, clipLimit, tileGridSize, valueChannel);


            // Calculate current luminance statistics (min, max, mean) of the value channel
            var currentLuminance = getCurrentLuminance(valueChannel);
            
            //Console.WriteLine($"Current Luminance: Min={currentLuminance.Min}, Max={currentLuminance.Max}, Mean={currentLuminance.Mean}");

            
            // Calculate the scaling factor for the value channel
            double scale = (targetLuminance.Mean * alpha) / currentLuminance.Mean;
            valueChannel *= scale;


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

    private static MinMaxMean getCurrentLuminance(Mat valueChannel)
    {
        double currentMin = 0, currentMax = 0;
        System.Drawing.Point minLoc = new(), maxLoc = new();
        CvInvoke.MinMaxLoc(valueChannel, ref currentMin, ref currentMax, ref minLoc, ref maxLoc);
        
        var currentMean = CvInvoke.Mean(valueChannel).V0;

        
        return new MinMaxMean(currentMin, currentMax, currentMean);
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
        CvInvoke.Pow(srcFloat, gamma, srcFloat);  // Applies the power transformation to each pixel

        // Step 3: Convert back to the original depth type and scale accordingly
        if (depthType == DepthType.Cv8U)
        {
            srcFloat.ConvertTo(dst, DepthType.Cv8U, 255.0);  // Scale back to [0, 255] for 8-bit
        }
        else if (depthType == DepthType.Cv16U)
        {
            srcFloat.ConvertTo(dst, DepthType.Cv16U, 65535.0);  // Scale back to [0, 65535] for 16-bit
        }
        else if (depthType == DepthType.Cv32F)
        {
            dst = srcFloat.Clone();  // No scaling needed for 32F; just copy the result
        }

        return dst;
    }





    private MinMaxMean[] CalculateRollingAverage(IReadOnlyList<ImageLuminanceRecord> luminanceValues, int windowSize)
    {
        var rollingAverages = new MinMaxMean[luminanceValues.Count];
        var rollingSums = new double[3]; // [0] = Min, [1] = Max, [2] = Mean

        for (var i = 0; i < luminanceValues.Count; i++)
        {
            // Add current luminance values to the rolling sums
            for (var j = 0; j < 3; j++)
            {
                rollingSums[j] += luminanceValues[i].Luminance[j];
                if (i >= windowSize)
                {
                    rollingSums[j] -= luminanceValues[i - windowSize].Luminance[j];
                }
            }

            // Calculate the rolling averages and assign to rollingAverages[i]
            var effectiveWindowSize = Math.Min(i + 1, windowSize);
            rollingAverages[i] = new MinMaxMean
            {
                Min = rollingSums[0] / effectiveWindowSize,
                Max = rollingSums[1] / effectiveWindowSize,
                Mean = rollingSums[2] / effectiveWindowSize
            };
        }

        return rollingAverages;
    }
}

public struct ImageLuminanceRecord(string filePath, MinMaxMean luminance)
{
    public string FilePath { get; set; } = filePath;
    public MinMaxMean Luminance { get; set; } = luminance;
}

public struct MinMaxMean(double min, double max, double mean)
{
    public double Min { get; set; } = min;
    public double Max { get; set; } = max;
    public double Mean { get; set; } = mean;

    public double this[int i]
    {
        get
        {
            return i switch
            {
                0 => Min,
                1 => Max,
                2 => Mean,
                _ => 0
            };
        }

        set
        {
            switch (i)
            {
                case 0:
                    Min = value;
                    break;
                case 1:
                    Max = value;
                    break;
                case 2:
                    Mean = value;
                    break;
            }
        }
    }
}