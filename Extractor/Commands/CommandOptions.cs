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


    public static readonly CommandOption InputOption = new(InputLabel, new[]
    {
        "Required. The input video file."
    });

    public static readonly CommandOption OutputOption = new(OutputLabel, new[]
    {
        "The output path. Default: Name of the input file with the relevant extension."
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
}
