using System.Globalization;
using Extractor.enums;
using Extractor.Handlers;
using TreeBasedCli;

namespace Extractor.Commands;

public class NormalizeLuminanceCommand : LeafCommand<NormalizeLuminanceCommand.NormalizeLuminanceArguments,
    NormalizeLuminanceCommand.Parser, NormalizeLuminanceCommandHandler>
{
    public NormalizeLuminanceCommand() : base(
        "normalize-luminance",
        new[]
        {
            "Normalizes the luminance values of all frames in a directory."
        },
        new[]
        {
            CommandOptions.InputDirOption,
            CommandOptions.OutputDirOption,
            CommandOptions.GlobalAverageOption,
            CommandOptions.WindowSizeOption,
            CommandOptions.MaxConcurrentTasksOption,
            CommandOptions.GammaOption,
            CommandOptions.ClipLimitOption,
            CommandOptions.KernelSizeOption,
            CommandOptions.HeadroomOption,
            CommandOptions.ColorSpaceOption
        })
    {
    }

    public record NormalizeLuminanceArguments(
        string InputDir,
        string OutputDir,
        bool GlobalAverage,
        int windowSize,
        int MaxConcurrentTasks,
        double Gamma,
        int ClipLimit,
        int KernelSize,
        double Headroom,
        ColorSpace colorSpace)
        : IParsedCommandArguments;

    public class Parser : ICommandArgumentParser<NormalizeLuminanceArguments>
    {
        public IParseResult<NormalizeLuminanceArguments> Parse(CommandArguments arguments)
        {
            var inputDir = arguments.GetArgument(CommandOptions.InputLabel).ExpectedAsSinglePathToExistingDirectory();
            var outputDir = arguments.GetArgumentOrNull(CommandOptions.OutputLabel)?.ExpectedAsSingleValue() ??
                            inputDir + "-normalized";

            var globalAverageVal =
                arguments.GetArgumentOrNull(CommandOptions.GlobalAverageLabel)?.ExpectedAsSingleValue() ??
                "false";

            var windowSize = arguments.GetArgumentOrNull(CommandOptions.WindowSizeLabel)?.ExpectedAsSingleInteger() ??
                             10;

            if (!bool.TryParse(globalAverageVal, out var globalAverage))
            {
                return new FailedParseResult<NormalizeLuminanceArguments>(
                    $"Invalid value for {CommandOptions.GlobalAverageLabel}. Must be a boolean."
                );
            }

            var maxConcurrentTasks =
                arguments.GetArgumentOrNull(CommandOptions.MaxConcurrentTasksLabel)?.ExpectedAsSingleInteger() ??
                Environment.ProcessorCount;

            var gammaVal = arguments.GetArgumentOrNull(CommandOptions.GammaLabel)?.ExpectedAsSingleValue() ?? "0.5";

            var gamma = double.Parse(gammaVal, CultureInfo.InvariantCulture);

            var clipLimit = arguments.GetArgumentOrNull(CommandOptions.ClipLimitLabel)?.ExpectedAsSingleInteger() ?? 10;

            var kernelSize = arguments.GetArgumentOrNull(CommandOptions.KernelSizeLabel)?.ExpectedAsSingleInteger() ??
                             8;

            var headroomVal = arguments.GetArgumentOrNull(CommandOptions.HeadroomLabel)?.ExpectedAsSingleValue() ??
                              "0.2";

            var headroom = double.Parse(headroomVal, CultureInfo.InvariantCulture);
            
            var colorSpace = arguments.GetArgumentOrNull(CommandOptions.ColorSpaceLabel)?.ExpectedAsSingleValue() ??
                            "lab";

            
            // convert color space to enum
            var colorSpaceEnum = colorSpace switch
            {
                "lab" => ColorSpace.LAB,
                "hsv" => ColorSpace.HsV,
                _ => throw new ArgumentOutOfRangeException(nameof(colorSpace), colorSpace, null)
            };

            var result = new NormalizeLuminanceArguments(
                inputDir,
                outputDir,
                globalAverage,
                windowSize,
                maxConcurrentTasks,
                gamma,
                clipLimit,
                kernelSize,
                headroom,
                colorSpaceEnum
            );

            return new SuccessfulParseResult<NormalizeLuminanceArguments>(result);
        }
    }
}