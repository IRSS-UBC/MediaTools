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
                        var luminance = await Task.Run(() => CalculateAverageImageLuminance(file));
                        task.Increment(1);
                        return luminance;
                    }).ToArray();

                    return await Task.WhenAll(luminanceTasks);
                });


        luminanceValues = luminanceValues.Where(value => value > 0).ToArray();

        var averageLuminance = luminanceValues.Average();

        await AnsiConsole.Progress().StartAsync(async ctx =>
        {
            var task = ctx.AddTask("[yellow]Normalizing luminance and saving to disk...[/]", maxValue: files.Length);
            var normalizationTasks = files.Select(async file =>
            {
                await Task.Run(() =>
                {
                    NormalizeImageLuminance(file, outputDirectory, averageLuminance);
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

    static bool NormalizeImageLuminance(string imagePath, string outputPath, double averageLuminance)
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
            var valueChannel = hsvChannels[2]; // Value channel represents the brightness

            // Calculate the current mean luminance of the image
            var currentMeanValue = CvInvoke.Mean(valueChannel);
            var currentLuminance = currentMeanValue.V0;

            // Normalize the value channel based on the average luminance
            var scaleFactor = averageLuminance / currentLuminance;
            valueChannel *= scaleFactor;

            // Merge the modified value channel back with the other HSV channels
            hsvChannels = new VectorOfMat(hsvChannels[0], hsvChannels[1], valueChannel);
            CvInvoke.Merge(hsvChannels, hsvImage);


            // Convert back to BGR color space
            var normalizedImage = new Mat();
            CvInvoke.CvtColor(hsvImage, normalizedImage, ColorConversion.Hsv2Bgr);

            // Save the normalized image to the output path with the same file name
            var outputFilePath = Path.Combine(outputPath, Path.GetFileName(imagePath));
            
            AnsiConsole.Console.WriteLine($"[green]Normalized image saved to {outputFilePath}. Mean luminance adjusted from {currentLuminance} to {averageLuminance}[/] ");
            
            
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
}