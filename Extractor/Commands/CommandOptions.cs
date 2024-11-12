using TreeBasedCli;
namespace Extractor.Commands;

public static class CommandOptions
{
    public const string InputLabel = "--input";
    public const string OutputLabel = "--output";
    
    public const string MaskOutputLabel = "--mask-output";
    public const string InputFormatLabel = "--in-format";
    public const string OutputFormatLabel = "--format";
    public const string OutputFrameDropRatioLabel = "--drop";
    public const string FrameRateLabel = "--fps";
    public const string DurationLabel = "--duration";

    public const string LengthLabel = "--length";
    
    public const string MaskLabel = "--mask";
    
    public const string EngineLabel = "--engine";
    
    public const string GpuCount = "--gpu-count";
    
    public const string GlobalAverageLabel = "--global-average";
    
    public const string WindowSizeLabel = "--window-size";
    
    public const string MaxConcurrentTasksLabel = "--max-concurrent-tasks";
    
    public const string GammaLabel = "--gamma";
    
    public const string ClipLimitLabel = "--clip-limit";
    
    public const string KernelSizeLabel = "--kernel-size";
    
    public const string HeadroomLabel = "--headroom";
    
    


    public static readonly CommandOption InputFileOption = new(InputLabel, new[]
    {
        "Required. The input video file."
    });
    
    public static readonly CommandOption InputDirOption = new(InputLabel, new[]
    {
        "Required. The input directory."
    });

    public static readonly CommandOption OutputDirOption = new(OutputLabel, new[]
    {
        "The output path. Default: Name of the input file or dir with an appended suffix."
    });
    
    public static readonly CommandOption MaskOutputOption = new (MaskOutputLabel, new[]
    {
        "The output path for the mask. Default: Name of the input file with the relevant extension."
    });

    public static readonly CommandOption InputFormatOption = new(InputFormatLabel, new[]
    {
        "The input image file format (e.g., .png or .jpg). Default: jpg"
    });

    public static readonly CommandOption OutputFormatOption = new(OutputFormatLabel, new[]
    {
        "The output image file format (e.g., .png or .jpg). Default: jpg"
    });

    public static readonly CommandOption OutputFrameRate = new(FrameRateLabel, new[]
    {
        "The output frame rate. Defaults to 30"
    });

    public static readonly CommandOption OutputDuration = new(DurationLabel, new[]
    {
        "The output duration. Default: 1"
    });

    public static readonly CommandOption OutputLength = new(LengthLabel, new[]
    {
        "The output length for square resolutions. Defaults to 128."
    });
    
    public static readonly CommandOption Mask = new(MaskLabel, new[]
    {
        "The mask to apply to the output. Default: none. Example: --mask sky"
    });
    
    public static readonly CommandOption EngineOption = new(EngineLabel, new[]
    {
        "The engine to use for processing. Default: cpu. Options: cpu, cuda, tensor-rt, directml, auto"
    });
    
    public static readonly CommandOption ProcessorCountOption = new(GpuCount, new[]
    {
        "The number of GPUs to use. Default: 1"
    });

    public static CommandOption GlobalAverageOption = new(GlobalAverageLabel, new[]
    {
        "Whether or not to use the global average for normalization. Default: false"
    });
    
    public static CommandOption WindowSizeOption = new(WindowSizeLabel, new[]
    {
        "The window size for the global average. Default: 10"
    });
    
    public static CommandOption MaxConcurrentTasksOption = new(MaxConcurrentTasksLabel, new[]
    {
        "The maximum number of concurrent tasks. Default: Number of logical processors."
    });
    
    public static CommandOption GammaOption = new(GammaLabel, new[]
    {
        "The gamma value for the normalization. Default: 0.5"
    });
    
    public static CommandOption ClipLimitOption = new(ClipLimitLabel, new[]
    {
        "The clip limit for the normalization. Default: 10"
    });
    
    public static CommandOption KernelSizeOption = new(KernelSizeLabel, new[]
    {
        "The kernel size for the normalization. Default: 8"
    });
    
    public static CommandOption HeadroomOption = new(HeadroomLabel, new[]
    {
        "The percentage of headroom for the normalization. Default: 0.2. Range: 0-1, keep below 0.5 for most uses."
    });
}
