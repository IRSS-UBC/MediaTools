using Extractor.Handlers;
using TreeBasedCli;
namespace Extractor.Commands;

public class AnimateCommand : LeafCommand<AnimateCommand.AnimateArguments, AnimateCommand.Parser, AnimateCommandHandler>
{
    public AnimateCommand() : base(
        "animate",
        new[]
        {
            "creates an animated gif from the input image file."
        },
        new[]
        {
            CommandOptions.InputFileOption,
            CommandOptions.OutputDirOption,
            CommandOptions.OutputFrameRate,
            CommandOptions.OutputDuration,
            CommandOptions.OutputLength
        })
    {
    }

    public record AnimateArguments(string InputFile, string OutputFile, int length, int Fps, float Duration) : IParsedCommandArguments;

    public class Parser : ICommandArgumentParser<AnimateArguments>
    {
        public IParseResult<AnimateArguments> Parse(CommandArguments arguments)
        {
            var inputFile = arguments.GetArgument(CommandOptions.InputLabel).ExpectedAsSinglePathToExistingFile();
            var outputFile = arguments.GetArgumentOrNull(CommandOptions.OutputLabel)?.ExpectedAsSingleValue() ?? Path.GetFileNameWithoutExtension(inputFile) + ".gif";
            var fps = arguments.GetArgumentOrNull(CommandOptions.FrameRateLabel)?.ExpectedAsSingleValue() ?? "30";
            var dur = arguments.GetArgumentOrNull(CommandOptions.DurationLabel)?.ExpectedAsSingleValue() ?? "1";

            var inputLength = arguments.GetArgumentOrNull(CommandOptions.LengthLabel)?.ExpectedAsSingleValue() ?? "256";

            var framesPerSecond = 30;
            if (int.TryParse(fps, out var fpsInt))
                framesPerSecond = Math.Abs(fpsInt);

            float duration = 1;
            if (float.TryParse(dur, out var durationFloat))
                duration = MathF.Abs(durationFloat);

            var length = 256;
            if (int.TryParse(inputLength, out var lengthInt))
                length = Math.Abs(lengthInt);

            var result = new AnimateArguments(
                inputFile,
                outputFile,
                length,
                framesPerSecond,
                duration
            );


            if (framesPerSecond > 100)
                return new FailedParseResult<AnimateArguments>("Frame rate cannot be greater than 100.");

            return new SuccessfulParseResult<AnimateArguments>(result);
        }
    }
}
