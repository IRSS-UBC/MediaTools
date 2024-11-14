using System.Diagnostics;
using System.Text;
using Extractor.Commands;
using Extractor.Extensions;
using Extractor.Patches;
using FFMediaToolkit;
using HarmonyLib;
using Spectre.Console;
using TreeBasedCli;
using TreeBasedCli.Exceptions;

namespace Extractor;

public static class Program
{
    private static FileStream? _fileStream;
    private static StreamWriter? _streamWriter;
    private static StreamReader? _streamReader;

    private static readonly Harmony Harmony = new("ca.ubc.forestry.irsslab.mediatools");

    private static async Task<int> Main(string[] args)
    {
        AppDomain.CurrentDomain.ProcessExit += CurrentDomain_ProcessExit;

        ValidateIo();

        try
        {
            SetFFmpegPath();
            
            var enableTiming = args.Contains("--time");
            var filteredArgs = args.Where(arg => arg != "--time").ToArray();
            
            await HandleArguments(filteredArgs, enableTiming);
        }
        catch (MessageOnlyException ex)
        {
            DisplayErrorMessage(ex.Message);
            return ex.ExitCode();
        }
        catch (Exception ex)
        {
            DisplayException(ex);
            return ex.ExitCode();
        }

        return 0;
    }

    private static void SetFFmpegPath()
    {
        FFmpegLoader.FFmpegPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "FFmpeg");
    }

    private static async Task HandleArguments(IReadOnlyCollection<string> args, bool enableTiming)
    {
        var settings = CreateArgumentHandlerSettings();
        var argumentHandler = new ArgumentHandler(settings);
        
        Stopwatch? stopwatch = null;
        if (enableTiming)
        {
            stopwatch = Stopwatch.StartNew();
        }
        
        await argumentHandler.HandleAsync(args);
        
        if (enableTiming && stopwatch != null)
        {
            stopwatch.Stop();
            AnsiConsole.MarkupLine($"[green]Execution Time: {stopwatch.Elapsed}[/]");
        }
    }

    private static ArgumentHandlerSettings CreateArgumentHandlerSettings()
    {
        return new ArgumentHandlerSettings(
            "IRSS Media Tools",
            $"{typeof(Program).Assembly.GetName().Version?.ToString()}",
            new CommandTree(
                CreateCommandTreeRoot(),
                DependencyInjectionService.Instance));
    }

    private static void ValidateIo()
    {
        CleanupOldLogFiles();

        if (!AreStandardStreamsAvailable())
            RedirectConsoleToLogFile();

        if (!IsConsoleHandleAvailable())
            Harmony.PatchCategory(PatchCategories.ConsoleHandlePatches);
    }

    private static bool AreStandardStreamsAvailable()
    {
        return Console.OpenStandardInput(1) != Stream.Null &&
               Console.OpenStandardOutput(1) != Stream.Null &&
               Console.OpenStandardError(1) != Stream.Null;
    }

    private static bool IsConsoleHandleAvailable()
    {
        try
        {
            _ = Console.WindowWidth;
        }
        catch (IOException)
        {
            return false;
        }

        return true;
    }

    private static void RedirectConsoleToLogFile()
    {
        var tempLogFile = CreateTempLogFile();

        _fileStream = new FileStream(tempLogFile, FileMode.OpenOrCreate, FileAccess.Write, FileShare.Read);
        _streamWriter = new StreamWriter(_fileStream, Encoding.ASCII);

        _streamWriter.AutoFlush = true;

        _streamReader = StreamReader.Null;

        Console.SetIn(_streamReader);
        Console.SetOut(_streamWriter);
        Console.SetError(_streamWriter);

        AnsiConsole.Create(new AnsiConsoleSettings
        {
            Ansi = AnsiSupport.No,
            ColorSystem = ColorSystemSupport.Legacy,
            Out = new AnsiConsoleOutput(_streamWriter),
            Interactive = InteractionSupport.No
        });
    }

    private static string CreateTempLogFile()
    {
        var logDirectoryPath = GetLogDirectoryPath();

        // Create the directory if it doesn't already exist
        Directory.CreateDirectory(logDirectoryPath);

        var fileName = $"{DateTime.Now:yyyyMMdd_HHmmssfff}_{Guid.NewGuid()}.log";
        return Path.Combine(logDirectoryPath, fileName);
    }


    private static void CleanupOldLogFiles()
    {
        var logDirectoryPath = GetLogDirectoryPath();

        if (!Directory.Exists(logDirectoryPath))
            return;
        var oneWeekAgo = DateTime.Now.AddDays(-7);
        DeleteFilesOlderThan(logDirectoryPath, oneWeekAgo);
    }

    private static string GetLogDirectoryPath()
    {
        var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
        return Path.Combine(appDataPath, "IRSSMediaTools");
    }

    private static void DeleteFilesOlderThan(string directoryPath, DateTime date)
    {
        var files = Directory.GetFiles(directoryPath);

        foreach (var file in files)
        {
            var fileInfo = new FileInfo(file);
            if (fileInfo.CreationTime < date)
                fileInfo.Delete();
        }
    }


    private static void DisplayErrorMessage(string message)
    {
        AnsiConsole.Markup($"[red]{message}[/]");
    }

    private static void DisplayException(Exception ex)
    {
        AnsiConsole.WriteException(ex,
            ExceptionFormats.ShortenPaths | ExceptionFormats.ShortenTypes |
            ExceptionFormats.ShortenMethods | ExceptionFormats.ShowLinks);
    }

    private static Command CreateCommandTreeRoot()
    {
        return new BranchCommandBuilder("")
            .WithDesription(new[]
            {
                "IRSS Media Tools is a command-line tool for extracting, converting, and working with media files.",
                "Unless otherwise stated, all commands will use the current working directory as the input/output folder."
            })
            .WithChildCommand(new ExtractAllCommand())
            .WithChildCommand(new ListInformationCommand())
            .WithChildCommand(new ConvertCommand())
            .WithChildCommand(new AnimateCommand())
            .WithChildCommand(new MaskSkyCommand())
            .WithChildCommand(new NormalizeLuminanceCommand())
            .Build();
    }

    private static void CurrentDomain_ProcessExit(object? sender, EventArgs e)
    {
        _streamWriter?.Close();
        _fileStream?.Close();
        _streamReader?.Close();
    }
}