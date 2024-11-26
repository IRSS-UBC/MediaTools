using System.Text.RegularExpressions;

namespace Extractor;

public partial class NaturalSortComparer : IComparer<string>
{
    private static readonly Regex NumberRegex = NumberExtractionRegex();

    public int Compare(string? x, string? y)
    {
        if (x == null || y == null) return 0;

        // Extract numbers from the strings
        var xNumbers = ExtractNumbers(x);
        var yNumbers = ExtractNumbers(y);

        // Compare each number in sequence
        for (var i = 0; i < Math.Min(xNumbers.Count, yNumbers.Count); i++)
        {
            var result = xNumbers[i].CompareTo(yNumbers[i]);
            if (result != 0)
                return result;
        }

        // If all numbers are identical, fall back on full string comparison
        return string.Compare(x, y, StringComparison.Ordinal);
    }

    private static List<int> ExtractNumbers(string input)
    {
        var numbers = new List<int>();
        foreach (Match match in NumberRegex.Matches(input))
        {
            numbers.Add(int.Parse(match.Value));
        }
        return numbers;
    }

    [GeneratedRegex("\\d+", RegexOptions.Compiled)]
    private static partial Regex NumberExtractionRegex();
}