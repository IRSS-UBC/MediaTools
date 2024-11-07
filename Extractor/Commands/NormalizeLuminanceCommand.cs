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
            CommandOptions.InputOption,
            CommandOptions.OutputOption
        })
    {
    }

    public record NormalizeLuminanceArguments(string InputDirectory, string OutputDir) : IParsedCommandArguments;

    public class Parser : ICommandArgumentParser<NormalizeLuminanceArguments>
    {
        public IParseResult<NormalizeLuminanceArguments> Parse(CommandArguments arguments)
        {
            var inputDir = arguments.GetArgument(CommandOptions.InputLabel).ExpectedAsSinglePathToExistingDirectory();
            var outputDir = arguments.GetArgumentOrNull(CommandOptions.OutputLabel)?.ExpectedAsSingleValue() ??
                            Path.GetFileNameWithoutExtension(inputDir);


            var result = new NormalizeLuminanceArguments(
                inputDir,
                outputDir
            );

            return new SuccessfulParseResult<NormalizeLuminanceArguments>(result);
        }
    }
}