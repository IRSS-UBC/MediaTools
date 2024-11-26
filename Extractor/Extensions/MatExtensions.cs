using Emgu.CV;
using Emgu.CV.CvEnum;

namespace Extractor.Extensions;

public static class MatExtensions
{
    public static Mat ConvertScaleTo(this Mat src, DepthType depthType)
    {
        var dst = new Mat();
        var upperBound = depthType switch
        {
            DepthType.Cv8U => 255,
            DepthType.Cv16U => 65535,
            DepthType.Cv32F => 1,
            _ => throw new ArgumentOutOfRangeException(nameof(depthType), depthType,
                @"Only CV8u and CV16u are supported.")
        };


        // Make sure that the conversion is done in the correct order
        // If the source is smaller than the destination, convert first and then normalize
        // If the source is larger than the destination, normalize first and then convert
        // If the source is a float and the destination is an integer, normalize first and then convert
        // If the source is an integer and the destination is a float, convert first and then normalize

        bool convertFirst;

        if (src.Depth == DepthType.Cv32F && depthType != DepthType.Cv32F)
        {
            convertFirst = false;
        }
        else if (src.Depth != DepthType.Cv32F && depthType == DepthType.Cv32F)
        {
            convertFirst = true;
        }
        else switch (src.Depth)
        {
            case DepthType.Cv16U when depthType == DepthType.Cv8U:
                convertFirst = true;
                break;
            case DepthType.Cv8U when depthType == DepthType.Cv16U:
                convertFirst = false;
                break;
            case DepthType.Default:
            case DepthType.Cv8S:
            case DepthType.Cv16S:
            case DepthType.Cv32S:
            case DepthType.Cv32F:
            case DepthType.Cv64F:
            default:
                // throw an exception if the conversion is not possible
                throw new ArgumentException("Conversion order not defined for the given types.");
        }

        if (convertFirst)
        {
            src.ConvertTo(dst, depthType);
            CvInvoke.Normalize(src, src, 0, upperBound, NormType.MinMax);
        }
        else
        {
            CvInvoke.Normalize(src, src, 0, upperBound, NormType.MinMax);
            src.ConvertTo(dst, depthType);
        }


        return dst;
    }
}