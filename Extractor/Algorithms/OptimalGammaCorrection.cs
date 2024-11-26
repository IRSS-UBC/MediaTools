using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Extractor.Structs;

namespace Extractor.Algorithms;

class OGGCPEAlgorithm
{
    private const double Ta = 0.95;
    private const double SigmaW = 2.25;

    public static Mat DoGConvolution(Mat src)
    {
        Mat g1 = new Mat(), g2 = new Mat(), result = new Mat();
        CvInvoke.GaussianBlur(src, g1, new System.Drawing.Size(3, 3), 0.5);
        CvInvoke.GaussianBlur(src, g2, new System.Drawing.Size(3, 3), 1.5);
        CvInvoke.Subtract(g1, g2, result);
        CvInvoke.Add(src, result, result); // Add result back to src
        return result;
    }

    public static double CalculateMean(Mat input)
    {
        MCvScalar mean = CvInvoke.Mean(input);
        return mean.V0;
    }



    public static double AmbArgmin(Mat input, Mat recursive, double targetValue, double r)
    {
        const double tolerance = 1e-7;
        double optimalR = r;

        while (true)
        {
            Mat dst = new Mat();
            CvInvoke.Pow(input, optimalR, dst);
            double up = CalculateMean(dst) - targetValue;

            Mat logRecursive = new Mat();
            CvInvoke.Log(recursive, logRecursive);
            Mat weightedDst = new Mat();
            CvInvoke.Multiply(dst, logRecursive, weightedDst);

            double down = CalculateMean(weightedDst);
            double newR = optimalR - (up / down);

            if (Math.Abs(optimalR - newR) < tolerance)
                return newR;

            optimalR = newR;
        }
    }

    public static Mat Apply(Mat src)
    {
        Mat[] channels = src.Split();
        Mat lIn = new Mat();
        CvInvoke.AddWeighted(channels[2], 0.299, channels[1], 0.587, 0, lIn);
        CvInvoke.AddWeighted(lIn, 1, channels[0], 0.114, 0, lIn);

        double min = new double(), max = new double();

        System.Drawing.Point minLoc = new(), maxLoc = new();

        CvInvoke.MinMaxLoc(lIn, ref min, ref max, ref minLoc, ref maxLoc);

        // Convert lIn to floating-point type (CV_64F for higher precision)
        Mat lInFloat = new Mat();
        lIn.ConvertTo(lInFloat, DepthType.Cv64F);  // Convert lIn to CV_64F

// Create a matrix filled with 1.0 to add to lInFloat
        Mat ones = new Mat(lInFloat.Size, lInFloat.Depth, lInFloat.NumberOfChannels);
        ones.SetTo(new MCvScalar(1.0));

// Add the "ones" matrix to avoid log(0)
        CvInvoke.Add(lInFloat, ones, lInFloat);

// Apply the logarithm
        Mat lLog = new Mat();
        CvInvoke.Log(lInFloat, lLog);  // Apply log to lInFloat

        
        Mat maxLogMat = new Mat(lLog.Size, DepthType.Cv64F, 1);
        maxLogMat.SetTo(new MCvScalar(Math.Log(max + 1)));
        CvInvoke.Divide(lLog, maxLogMat, lLog);

        Mat brightSet = lLog.Clone();
        Mat darkSet = new Mat();
        CvInvoke.Threshold(lLog, brightSet, 0.5, 1.0, ThresholdType.ToZero);
        CvInvoke.Threshold(lLog, darkSet, 0.5, 1.0, ThresholdType.ToZeroInv);




        var darkset_lum = LuminanceValues.GetCurrentLuminance(darkSet);

        var brightset_lum = LuminanceValues.GetCurrentLuminance(brightSet);


// Extract standard deviation values for use in gamma calculations
        double gammaLow = AmbArgmin(darkSet, darkSet, darkset_lum.StdDev, 1.0);
        double gammaHigh = AmbArgmin(brightSet, brightSet, brightset_lum.StdDev, 1.0);


        Mat lDark = new Mat(), lBright = new Mat();
        CvInvoke.Pow(lLog, gammaLow, lDark);
        CvInvoke.Pow(lLog, gammaHigh, lBright);

        Mat commaDark = DoGConvolution(lDark);
        Mat commaBright = DoGConvolution(lBright);

        Mat w = new Mat();
        CvInvoke.Pow(lBright, 3.0, w);
        CvInvoke.Exp(w * SigmaW, w);

        Mat lOut = new Mat();
        CvInvoke.AddWeighted(commaDark, 1, commaBright, -1, 0, lOut);
        CvInvoke.Multiply(w, lOut, lOut);
        CvInvoke.Add(commaBright, lOut, lOut);

        Mat s = new Mat();
        s = TanhApprox(lBright * -1);
        Mat scalarMat = new Mat(s.Size, s.Depth, s.NumberOfChannels);
        scalarMat.SetTo(new MCvScalar(Ta)); // Fill scalarMat with the value of t_a
        CvInvoke.Add(s, scalarMat, s); // Add scalarMat to s

        Mat[] iC = [lIn.Clone(), lIn.Clone(), lIn.Clone()];
        using var vectorOfMat = new VectorOfMat(iC);
        var result = new Mat();
        CvInvoke.Merge(vectorOfMat, result); // Merge the channels into a single Mat

        // Multiply with 's'
        CvInvoke.Multiply(result, s, result);

        // Scale and convert the result
        result *= 255;
        result.ConvertTo(result, DepthType.Cv8U);

        return result;
    }


    public static Mat TanhApprox(Mat input)
    {
        Mat exp2x = new Mat();
        CvInvoke.Exp(input * 2, exp2x); // Calculate exp(2 * x) for each element in the matrix

        Mat numerator = exp2x - 1; // Calculate exp(2 * x) - 1
        Mat denominator = exp2x + 1; // Calculate exp(2 * x) + 1

        Mat result = new Mat();
        CvInvoke.Divide(numerator, denominator, result); // Compute (exp(2 * x) - 1) / (exp(2 * x) + 1)

        return result;
    }
}