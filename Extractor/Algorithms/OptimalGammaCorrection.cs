// Implementation of the paper [SCIE] Inho Jeong and Chul Lee, “An optimization-based approach to gamma correction parameter estimation for low-light image enhancement,” Multimedia Tools and Applications, vol. 80, no. 12, pp. 18027–18042, May 2021.
// Derived from: https://github.com/gitofinho/Optimal-Gamma-Correction-Parameter-Estimation/tree/master

using Emgu.CV;
using Emgu.CV.CvEnum;

namespace Extractor.Algorithms;

public class OptimalGammaCorrection
{
    public static Mat ApplyCorrection(Mat src, DepthType depthType = DepthType.Cv8U)
    {
        Mat srcFloat = new Mat();
        Mat dst = new Mat();
        switch (depthType)
        {
            // Convert source image to a 32-bit float and normalize based on depth type
            case DepthType.Cv8U:
                src.ConvertTo(srcFloat, DepthType.Cv32F, 1.0 / 255.0); // Normalize 8-bit to [0, 1]
                break;
            case DepthType.Cv16U:
                src.ConvertTo(srcFloat, DepthType.Cv32F, 1.0 / 65535.0); // Normalize 16-bit to [0, 1]
                break;
            case DepthType.Cv32F:
                // If the image is already 32F, no scaling needed, just copy to srcFloat
                srcFloat = src.Clone();
                break;
            default:
                throw new ArgumentException("Unsupported depth type for gamma correction.");
        }


        const double sigmaW = 2.25;


        var luminanceFloat = srcFloat;

        // Step 1: Calculate log-scaled luminance and separate into bright and dark sets
        Mat lLog = new Mat();
        CvInvoke.Log(luminanceFloat + 1, lLog); // Prevent log(0) issues
        CvInvoke.Normalize(lLog, lLog, 0, 1, NormType.MinMax);

        Mat brightSet = new(), darkSet = new();
        CvInvoke.Threshold(lLog, brightSet, 0.5, 1.0, ThresholdType.Binary);
        CvInvoke.Threshold(lLog, darkSet, 0.5, 1.0, ThresholdType.BinaryInv);

        // Step 2: Calculate gamma values for bright and dark sets
        var gammaLow = CalculateGamma(darkSet); // Adaptive gamma function
        var gammaHigh = CalculateGamma(brightSet);

        Mat luminanceDark = new(), luminanceBright = new();
        CvInvoke.Pow(lLog, gammaLow, luminanceDark);
        CvInvoke.Pow(lLog, gammaHigh, luminanceBright);

        // Step 3: DoG Convolution on dark and bright luminance maps
        var doGDark = DoGConvolution(luminanceDark);
        var doGBright = DoGConvolution(luminanceBright);

        // Step 4: Weighting function and output luminance
        var w = new Mat();
        CvInvoke.Pow(luminanceBright, 3.0, w);
        CvInvoke.Exp(w * sigmaW, w);

        var outputLuminance = new Mat();
        CvInvoke.AddWeighted(doGDark, 1.0, doGBright, 1.0, 0, outputLuminance);

        // Step 5: Reassemble image with adjusted luminance
        var result = new Mat();
        src.ConvertTo(result, DepthType.Cv32F, 1.0 / 255.0); // Ensure input is float type and normalized
        for (var i = 0; i < 3; i++)
        {
            var channel = new Mat();
            CvInvoke.ExtractChannel(result, channel, i);
            CvInvoke.Multiply(channel, outputLuminance, channel);
            CvInvoke.InsertChannel(channel, result, i);
        }

        result *= 255;
        result.ConvertTo(result, depthType); // Convert back to original depth type

        switch (depthType)
        {
            // Convert back to the original depth type and scale accordingly
            case DepthType.Cv8U:
                result.ConvertTo(dst, DepthType.Cv8U, 255.0); // Scale back to [0, 255] for 8-bit
                break;
            case DepthType.Cv16U:
                result.ConvertTo(dst, DepthType.Cv16U, 65535.0); // Scale back to [0, 65535] for 16-bit
                break;
            case DepthType.Cv32F:
                dst = result.Clone(); // No scaling needed for 32F; just copy the result
                break;
            default:
                throw new ArgumentException("Unsupported depth type for gamma correction.");
        }

        return dst;
    }


    // Perform Difference of Gaussians (DoG) Convolution
    private static Mat DoGConvolution(IInputArray src)
    {
        Mat g1 = new(), g2 = new(), result = new();
        CvInvoke.GaussianBlur(src, g1, new System.Drawing.Size(3, 3), 0.5);
        CvInvoke.GaussianBlur(src, g2, new System.Drawing.Size(3, 3), 1.5);
        CvInvoke.Subtract(g1, g2, result);
        CvInvoke.Add(src, result, result);
        return result;
    }


    // Calculate adaptive gamma based on the amb_argmin method
    private static double CalculateGamma(IInputArray input, IInputArray recursive, double targetValue, double r, double tolerance = 1e-7)
    {
        while (true)
        {
            // Step 1: Calculate the powered input (input ^ r) and mean of powered values
            var poweredInput = new Mat();
            CvInvoke.Pow(input, r, poweredInput);
            var up = CvInvoke.Mean(poweredInput).V0 - targetValue;

            // Step 2: Calculate logarithmic mean
            var logInput = new Mat();
            CvInvoke.Log(recursive, logInput);
            var adjustedInput = new Mat();
            CvInvoke.Multiply(poweredInput, logInput, adjustedInput);
            var down = CvInvoke.Mean(adjustedInput).V0;

            // Step 3: Compute the optimal new r value
            var optimalR = r - (up / (down + 1e-6)); // Small epsilon to avoid division by zero

            // Step 4: Recursive condition based on tolerance
            if (Math.Abs(r - optimalR) < tolerance)
                return optimalR;
            r = optimalR;
        }
    }

    // Calculate gamma with default recursive call parameters
    private static double CalculateGamma(Mat luminanceSet, double initialR = 1.0)
    {
        var recursiveLuminance = luminanceSet.Clone(); // Make a copy for recursive calculation
        var meanValue = CvInvoke.Mean(luminanceSet).V0; // Target mean value
        return CalculateGamma(luminanceSet, recursiveLuminance, meanValue, initialR);
    }
}