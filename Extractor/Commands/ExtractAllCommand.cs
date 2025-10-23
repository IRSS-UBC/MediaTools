using Extractor.Handlers;
using TreeBasedCli;
using static System.Boolean;

namespace Extractor.Commands;

public class ExtractAllCommand : LeafCommand<ExtractAllCommand.ExtractAllArguments, ExtractAllCommand.Parser,
    ExtractAllCommandHandler>
{
    public ExtractAllCommand() : base(
        "extract",
        [
            "extracts all frames from the input video file."
        ],
        [
            CommandOptions.InputFileOption,
            CommandOptions.InputDJISRTOption,
            CommandOptions.OutputDirOption,
            CommandOptions.OutputSpatialFileOption,
            CommandOptions.MaskOutputOption,
            CommandOptions.OutputFormatOption,
            new CommandOption(CommandOptions.OutputFrameDropRatioLabel, [
                "The ratio of frames to drop from the output. (e.g., 0 = no drop, 0.5 = half, 1 = all). Default: 0"
            ]),
            CommandOptions.OutputFrameRate,
            CommandOptions.Mask,
            CommandOptions.CPUCountOption
        ])
    {
    }

    public record ExtractAllArguments(
        string InputFile,
        string FramesOutputFolder,
        string MasksOutputFolder,
        string OutputFormat,
        float DropRatio,
        int? FrameRate,
        ImageMaskGeneration ImageMaskGeneration,
        bool ExtractGeoSpatial,
        int cpuCount) : IParsedCommandArguments;

    public class Parser : ICommandArgumentParser<ExtractAllArguments>
    {
        public IParseResult<ExtractAllArguments> Parse(CommandArguments arguments)
        {
            var inputFile = arguments.GetArgument(CommandOptions.InputLabel).ExpectedAsSinglePathToExistingFile();

            var InputDJISRTString =
                arguments.GetArgumentOrNull(CommandOptions.InputDjisrtLabel)?.ExpectedAsSingleValue() ?? "false";

            var outputSpatialFileOption =
                arguments.GetArgumentOrNull(CommandOptions.OutputSpatialFileLabel)?.ExpectedAsSingleValue() ??
                "ouput.json";

            var framesOutputFolder = arguments.GetArgumentOrNull(CommandOptions.OutputLabel)?.ExpectedAsSingleValue() ??
                                     Path.GetFileNameWithoutExtension(inputFile);
            var masksOutputFolder =
                arguments.GetArgumentOrNull(CommandOptions.MaskOutputLabel)?.ExpectedAsSingleValue() ??
                framesOutputFolder + "_mask";
            var outputFormat =
                (arguments.GetArgumentOrNull(CommandOptions.OutputFormatLabel)?.ExpectedAsSingleValue() ?? "jpg")
                .ToLowerInvariant().Replace(".", "", StringComparison.InvariantCultureIgnoreCase);
            var dropRatioString = arguments.GetArgumentOrNull(CommandOptions.OutputFrameDropRatioLabel)
                ?.ExpectedAsSingleValue();
            var fpsString = arguments.GetArgumentOrNull(CommandOptions.FrameRateLabel)?.ExpectedAsSingleValue();
            var mask = arguments.GetArgumentOrNull(CommandOptions.MaskLabel)?.ExpectedAsSingleValue();


            var cpuCountString = arguments.GetArgumentOrNull(CommandOptions.CoreCount)?.ExpectedAsSingleValue();


            int cpuCount;
            if (cpuCountString != null)
            {
                cpuCount = int.Parse(cpuCountString);
            }
            else
            {
                cpuCount = Environment.ProcessorCount * 2;
            }


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

                var validMaskText = maskNames.Where(name => name != "None")
                    .Aggregate("", (current, name) => current + name + ", ");


                return new FailedParseResult<ExtractAllArguments>(
                    $"Invalid mask. Valid masks are: {validMaskText.TrimEnd(' ')} or a combination of them (e.g., \"{maskNames[1]} {maskNames.Last()}\").");
            }


            var InputDJI = false;

            var parse = TryParse(InputDJISRTString, out InputDJI);


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
                imageMaskGeneration,
                InputDJI,
                cpuCount
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