using Emgu.CV;
using Emgu.CV.Structure;

namespace Extractor.Structs;

public struct LuminanceValues(double min, double max, double mean, double stdDev)
{
    public double Min { get; init; } = min;
    public double Max { get; init; } = max;
    public double Mean { get; init; } = mean;

    public double StdDev { get; init; } = stdDev;

    public double this[int i]
    {
        get
        {
            return i switch
            {
                0 => Min,
                1 => Max,
                2 => Mean,
                3 => StdDev,
                _ => 0
            };
        }
    }
    
    public static LuminanceValues GetCurrentLuminance(Mat valueChannel)
    {
        MCvScalar mean = new(0), stddev = new(0);
        CvInvoke.MeanStdDev(valueChannel, ref mean, ref stddev);


        double currentMin = 0, currentMax = 0;
        System.Drawing.Point minLoc = new(), maxLoc = new();
        CvInvoke.MinMaxLoc(valueChannel, ref currentMin, ref currentMax, ref minLoc, ref maxLoc);


        return new LuminanceValues(currentMin, currentMax, mean.V0, stddev.V0);
    }
}