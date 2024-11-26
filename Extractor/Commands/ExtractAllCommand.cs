using Extractor.Handlers;
using TreeBasedCli;
namespace Extractor.Commands;

public class ExtractAllCommand : LeafCommand<ExtractAllCommand.ExtractAllArguments, ExtractAllCommand.Parser, ExtractAllCommandHandler>
{
    public ExtractAllCommand() : base(
        "extract",
        new[]
        {
            "extracts all frames from the input video file."
        },
        new[]
        {
            CommandOptions.InputFileOption,
            CommandOptions.OutputDirOption,
            CommandOptions.MaskOutputOption,
            CommandOptions.OutputFormatOption,
            new CommandOption(CommandOptions.OutputFrameDropRatioLabel, new[]
            {
                "The ratio of frames to drop from the output. (e.g., 0 = no drop, 0.5 = half, 1 = all). Default: 0"
            }),
            CommandOptions.OutputFrameRate,
            CommandOptions.Mask
        })
    {
    }

    public record ExtractAllArguments(string InputFile, string FramesOutputFolder, string MasksOutputFolder, string OutputFormat, float DropRatio, int? FrameRate, ImageMaskGeneration ImageMaskGeneration) : IParsedCommandArguments;

    public class Parser : ICommandArgumentParser<ExtractAllArguments>
    {
        public IParseResult<ExtractAllArguments> Parse(CommandArguments arguments)
        {
            var inputFile = arguments.GetArgument(CommandOptions.InputLabel).ExpectedAsSinglePathToExistingFile();
            var framesOutputFolder = arguments.GetArgumentOrNull(CommandOptions.OutputLabel)?.ExpectedAsSingleValue() ?? Path.GetFileNameWithoutExtension(inputFile);
            var masksOutputFolder = arguments.GetArgumentOrNull(CommandOptions.MaskOutputLabel)?.ExpectedAsSingleValue() ?? framesOutputFolder + "_mask";
            var outputFormat = (arguments.GetArgumentOrNull(CommandOptions.OutputFormatLabel)?.ExpectedAsSingleValue() ?? "jpg").ToLowerInvariant().Replace(".", "", StringComparison.InvariantCultureIgnoreCase);
            var dropRatioString = arguments.GetArgumentOrNull(CommandOptions.OutputFrameDropRatioLabel)?.ExpectedAsSingleValue();
            var fpsString = arguments.GetArgumentOrNull(CommandOptions.FrameRateLabel)?.ExpectedAsSingleValue();
            var mask = arguments.GetArgumentOrNull(CommandOptions.MaskLabel)?.ExpectedAsSingleValue();



            var imageMaskGeneration = ImageMaskGeneration.None;

            if (mask != null)
            {
                var maskParts = mask.Split(" ", StringSplitOptions.RemoveEmptyEntries);

                foreach (var maskPart in maskParts)
                {
                    if (Enum.TryParse(typeof(ImageMaskGeneration), maskPart, true, out var parsedValue))
                    {
                        imageMaskGeneration |= (ImageMaskGeneration)parsedValue;
                    }
                }
            }

            if (mask != null && imageMaskGeneration == ImageMaskGeneration.None)
            {
                var maskNames = Enum.GetNames(typeof(ImageMaskGeneration));

                var validMaskText = maskNames.Where(name => name != "None").Aggregate("", (current, name) => current + name + ", ");


                return new FailedParseResult<ExtractAllArguments>($"Invalid mask. Valid masks are: {validMaskText.TrimEnd(' ')} or a combination of them (e.g., \"{maskNames[1]} {maskNames.Last()}\").");
            }

            float dropRatio;

            if (float.TryParse(dropRatioString, out var parseResult))
                dropRatio = parseResult;
            else
                dropRatio = 0;


            int? fps;
            if (int.TryParse(fpsString, out var fpsResult))
            {
                fps = fpsResult;
            }
            else
            {
                fps = null;
            }
               
            

            var result = new ExtractAllArguments(
                inputFile,
                framesOutputFolder,
                masksOutputFolder,
                outputFormat,
                Math.Abs(dropRatio),
                fps,
                imageMaskGeneration
            );

            return new SuccessfulParseResult<ExtractAllArguments>(result);
        }
    }


    [Flags]
    public enum ImageMaskGeneration
    {
        None = 0,
        Sky = 1,
        Ground = 2,
        Vegetation = 4,
        Buildings = 8,
        Roads = 16
    }
}
