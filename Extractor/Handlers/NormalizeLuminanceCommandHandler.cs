using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
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
        var inputDirectory = arguments.InputDirectory;

        // read all files in the input directory
        // for each file, read the image, convert to HSV, and calculate the average luminance

        var files = Constants.SupportedExtensions
            .SelectMany(ext => Directory.GetFiles(inputDirectory, ext, SearchOption.AllDirectories)).ToArray();


        var tasks = files.Select((file, index) => Task.Run(() => CalculateAverageImageLuminance(file))).ToArray();

        var luminanceValues = await Task.WhenAll(tasks);

        luminanceValues = luminanceValues.Where(value => value > 0).ToArray();

        if (luminanceValues.Length == 0)
        {
            throw new MessageOnlyException("[red]No valid images found to calculate average luminance.[/]");
        }

        AnsiConsole.Console.WriteLine(
            $"Calculating average luminance of all images... n={luminanceValues.Length} images");

        // calculate the average luminance of all images
        double averageLuminance = luminanceValues.Average();

        AnsiConsole.MarkupLineInterpolated(
            $"Average luminance of all images in {inputDirectory} is [yellow]{averageLuminance}[/].");
    }


    static double CalculateAverageImageLuminance(string imagePath)
    {
        try
        {
            Mat image = CvInvoke.Imread(imagePath, ImreadModes.Color);


            // Convert the image to HSV color space
            Mat hsvImage = new Mat();
            CvInvoke.CvtColor(image, hsvImage, ColorConversion.Bgr2Hsv);

            // Split the HSV image into separate channels (Hue, Saturation, and Value)
            Mat[] hsvChannels = hsvImage.Split();
            Mat valueChannel = hsvChannels[2]; // Value channel represents the brightness

            // Calculate the average luminance value
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
}