namespace Extractor;

public static class CoordinateConverter
{
    public static (float degree, float minute, float second, string direction) DecimalToDms(double coordinate, bool isLongitude)
    {
        switch (isLongitude)
        {
            // Ensure the coordinate is within a valid range
            case true when coordinate is < -180.0 or > 180.0:
                throw new ArgumentOutOfRangeException(nameof(coordinate), "Longitude must be between -180 and 180.");
            case false when coordinate is < -90.0 or > 90.0:
                throw new ArgumentOutOfRangeException(nameof(coordinate), "Latitude must be between -90 and 90.");
        }

        // Determine the direction (N, S, E, W)
        string direction;
        if (isLongitude)
        {
            direction = coordinate >= 0 ? "E" : "W";
        }
        else
        {
            direction = coordinate >= 0 ? "N" : "S";
        }

        // Work with the absolute value of the coordinate
        double absoluteCoordinate = Math.Abs(coordinate);

        // Get the whole number part for degrees
        int degrees = (int)absoluteCoordinate;

        // Get the remaining fractional part and multiply by 60 for minutes
        double remainingMinutes = (absoluteCoordinate - degrees) * 60;
        int minutes = (int)remainingMinutes;

        // Get the remaining fractional part and multiply by 60 for seconds
        float seconds = (float) (remainingMinutes - minutes) * 60;

        // Format the final string with 2 decimal places for seconds
        return (degrees, minutes, seconds, direction);
    }
}