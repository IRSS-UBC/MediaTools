using Extractor.Handlers;
using TreeBasedCli;
namespace Extractor.Commands;

public class ConvertCommand : LeafCommand<ConvertCommand.ConvertCommandArguments, ConvertCommand.Parser, ConvertCommandHandler>
{
    public ConvertCommand() : base("convert", new[]
    {
        "converts all images from the input folder to another format"
    }, new[]
    {
        CommandOptions.InputFileOption,
        CommandOptions.OutputDirOption,
        CommandOptions.OutputFormatOption,
        CommandOptions.InputFormatOption
    })
    {
    }


    public record ConvertCommandArguments(string InputFolder, string OutputFolder, string InputFormat, string OutputFormat) : IParsedCommandArguments;

    public class Parser : ICommandArgumentParser<ConvertCommandArguments>
    {
        public IParseResult<ConvertCommandArguments> Parse(CommandArguments arguments)
        {
            var inputFolder = arguments.GetArgument(CommandOptions.InputLabel).ExpectedAsSinglePathToExistingDirectory();
            var outputFolder = arguments.GetArgumentOrNull(CommandOptions.OutputLabel)?.ExpectedAsSingleValue() ?? Path.GetDirectoryName(inputFolder) + "_converted";

            var inputFormat = (arguments.GetArgumentOrNull(CommandOptions.InputFormatLabel)?.ExpectedAsSingleValue() ?? "jpg").ToLowerInvariant().Replace(".", "", StringComparison.InvariantCultureIgnoreCase);
            var outputFormat = (arguments.GetArgumentOrNull(CommandOptions.OutputFormatLabel)?.ExpectedAsSingleValue() ?? "jpg").ToLowerInvariant().Replace(".", "", StringComparison.InvariantCultureIgnoreCase);


            var results = new ConvertCommandArguments(inputFolder, outputFolder, inputFormat, outputFormat);

            return new SuccessfulParseResult<ConvertCommandArguments>(results);
        }
    }
}
