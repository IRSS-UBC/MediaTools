using System.Text.RegularExpressions;

namespace Extractor;

public partial class NaturalSortComparer : IComparer<string>
{
    public int Compare(string? x, string? y)
    {
        if (ReferenceEquals(x, y)) return 0;
        if (x == null) return -1;
        if (y == null) return 1;

        int ix = 0, iy = 0;

        while (ix < x.Length && iy < y.Length)
        {
            if (char.IsDigit(x[ix]) && char.IsDigit(y[iy]))
            {
                int result = CompareNumerically(x, ref ix, y, ref iy);
                if (result != 0) return result;
            }
            else
            {
                int result = x[ix].CompareTo(y[iy]);
                if (result != 0) return result;
                ix++;
                iy++;
            }
        }

        return x.Length.CompareTo(y.Length);
    }

    private static int CompareNumerically(string x, ref int ix, string y, ref int iy)
    {
        int xStart = ix;
        while (ix < x.Length && x[ix] == '0') ix++;
        
        int yStart = iy;
        while (iy < y.Length && y[iy] == '0') iy++;

        int xDigitsStart = ix;
        while (ix < x.Length && char.IsDigit(x[ix])) ix++;
        int xLen = ix - xDigitsStart;

        int yDigitsStart = iy;
        while (iy < y.Length && char.IsDigit(y[iy])) iy++;
        int yLen = iy - yDigitsStart;

      
        if (xLen != yLen) return xLen.CompareTo(yLen);

    
        for (int i = 0; i < xLen; i++)
        {
            if (x[xDigitsStart + i] != y[yDigitsStart + i])
                return x[xDigitsStart + i].CompareTo(y[yDigitsStart + i]);
        }
        
        
        int xZeros = xDigitsStart - xStart;
        int yZeros = yDigitsStart - yStart;
        return yZeros.CompareTo(xZeros); 
    }
}