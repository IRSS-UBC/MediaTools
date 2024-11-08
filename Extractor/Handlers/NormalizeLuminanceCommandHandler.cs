using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Extractor.Commands;
using Spectre.Console;
using TreeBasedCli;
using TreeBasedCli.Exceptions;

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
                    var task = ctx.AddTask("[yellow]Scanning luminance for images...[/]", maxValue: files.Length);
                    var luminanceTasks = files.Select(async file =>
                    {
                        var luminance = await Task.Run(() =>
                            new ImageLuminanceRecord(file, CalculateAverageImageLuminance(file)));
                        task.Increment(1);
                        return luminance;
                    }).ToArray();

                    return await Task.WhenAll(luminanceTasks);
                });


        var luminanceValuesList = luminanceValues.Where(record => record.Luminance > 0).ToList();


        luminanceValuesList.Sort((a, b) => new NaturalSortComparer().Compare(a.FilePath, b.FilePath));
        
        var rollingAverages = CalculateRollingAverage(luminanceValuesList, 10);
            
            
        // update the luminance values with the rolling averages
        for (var i = 0; i < luminanceValuesList.Count; i++)
        {
            luminanceValuesList[i] = new ImageLuminanceRecord(luminanceValuesList[i].FilePath, rollingAverages[i]);
        }


        var averageLuminance = luminanceValues.Average(record => record.Luminance);

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


    private static double CalculateAverageImageLuminance(string imagePath)
    {
        try
        {
            Mat image = CvInvoke.Imread(imagePath);

            Mat hsvImage = new Mat();
            CvInvoke.CvtColor(image, hsvImage, ColorConversion.Bgr2Hsv);

            Mat[] hsvChannels = hsvImage.Split();
            Mat valueChannel = hsvChannels[2];

           // CvInvoke.Normalize(valueChannel, valueChannel, 0, 255, NormType.MinMax);


            MCvScalar meanValue = CvInvoke.Mean(valueChannel);
            return meanValue.V0;
        }
        catch (Exception ex)
        {
            AnsiConsole.Console.WriteLine(
                $"[red]Warning: Error processing image at {imagePath}: {ex.Message}. Skipping this image.[/]");
            return 0;
        }
    }


    static bool NormalizeImageExposure(string imagePath, string outputPath, double globalMeanLuminance)
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


            //CvInvoke.Normalize(valueChannel, valueChannel, 0, 255, NormType.MinMax);

            // Apply Gamma Correction to boost dark areas
            double gamma = 0.5; // Lower values (<1) lighten dark areas, while values >1 darken
            valueChannel = ApplyGammaCorrection(valueChannel, gamma);

            // Apply CLAHE to the Value channel
            double clipLimit = 2.0; // Adjust as needed
            System.Drawing.Size tileGridSize = new System.Drawing.Size(8, 8);
            CvInvoke.CLAHE(valueChannel, clipLimit, tileGridSize, valueChannel);


            // Calculate the current mean luminance of the image
            var currentMeanValue = CvInvoke.Mean(valueChannel);
            var currentLuminance = currentMeanValue.V0;

            // Calculate scaling factor based on global mean luminance
            var scaleFactor = globalMeanLuminance / currentLuminance;
            valueChannel *= scaleFactor;

            // Clip pixel values to the range [0, 255] to avoid overflows
            //CvInvoke.Min(valueChannel, new ScalarArray(255), valueChannel);
            //CvInvoke.Max(valueChannel, new ScalarArray(0), valueChannel);


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


    private static Mat ApplyGammaCorrection(Mat src, double gamma)
    {
        Mat dst = new Mat();
        src.ConvertTo(dst, DepthType.Cv8U, 1.0 / gamma, 0);
        return dst;
    }


    private double[] CalculateRollingAverage(IReadOnlyList<ImageLuminanceRecord> luminanceValues, int windowSize)
    {
        var rollingAverages = new double[luminanceValues.Count];
        double rollingSum = 0;

        for (var i = 0; i < luminanceValues.Count; i++)
        {
            // Add the current luminance value to the rolling sum
            rollingSum += luminanceValues[i].Luminance;

            // If we've exceeded the window size, subtract the oldest value
            if (i >= windowSize)
            {
                rollingSum -= luminanceValues[i - windowSize].Luminance;
            }

            // Calculate the rolling average
            var effectiveWindowSize = Math.Min(i + 1, windowSize);
            rollingAverages[i] = rollingSum / effectiveWindowSize;
        }

        return rollingAverages;
    }
}

public struct ImageLuminanceRecord(string filePath, double luminance)
{
    public string FilePath { get; set; } = filePath;
    public double Luminance { get; set; } = luminance;
}