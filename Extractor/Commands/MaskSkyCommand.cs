using Extractor.Handlers;
using SkyRemoval;
using Spectre.Console;
using TreeBasedCli;
using TreeBasedCli.Exceptions;

namespace Extractor.Commands;

public class MaskSkyCommand : LeafCommand<MaskSkyCommand.MaskSkyArguments, MaskSkyCommand.Parser, MaskSkyCommandHandler>
{
    public MaskSkyCommand() : base(
        "mask_sky",
        new[]
        {
            "Generates sky masks for all images in a folder."
        },
        new[]
        {
            CommandOptions.InputFileOption,
            CommandOptions.OutputDirOption,
            CommandOptions.EngineOption,
            CommandOptions.ProcessorCountOption
        })
    {
    }

    public record MaskSkyArguments(
        string InputPath,
        string OutputPath,
        PathType InputType,
        ExecutionEngine Engine,
        int ProcessorCount) : IParsedCommandArguments;


    public class Parser : ICommandArgumentParser<MaskSkyArguments>
    {
        public IParseResult<MaskSkyArguments> Parse(CommandArguments arguments)
        {
            var inputPath = arguments.GetArgument(CommandOptions.InputLabel).ExpectedAsSingleValue();
            var outputFolder = arguments.GetArgumentOrNull(CommandOptions.OutputLabel)?.ExpectedAsSingleValue() ??
                               Path.GetFileNameWithoutExtension(inputPath) + "_mask";
            var engine = arguments.GetArgumentOrNull(CommandOptions.EngineLabel)
                ?.ExpectedAsEnumValue<ExecutionEngine>();
            var processorCount = arguments.GetArgumentOrNull(CommandOptions.GpuCount)?.ExpectedAsSingleValue() ?? "1";

            if (int.TryParse(processorCount, out var count))
            {
                if (count < 1)
                {
                    AnsiConsole.MarkupLine("[red]Processor count must be at least 1.[/]");
                    AnsiConsole.MarkupLine("[red]Setting count to 1.[/]");
                    count = 1;
                }
            }
            else
            {
                throw new MessageOnlyException("Processor count must be an integer.");
            }


            var inputDirectory = PathType.None;

            if (Directory.Exists(inputPath))
            {
                inputDirectory |= PathType.Directory;
            }

            if (File.Exists(inputPath))
            {
                inputDirectory |= PathType.File;
            }

            if (inputDirectory == PathType.None)
            {
                throw new MessageOnlyException("Input file or directory does not exist.");
            }


            var result = new MaskSkyArguments(
                inputPath,
                outputFolder,
                inputDirectory,
                engine ?? ExecutionEngine.Auto,
                count
            );


            return new SuccessfulParseResult<MaskSkyArguments>(result);
        }
    }
}