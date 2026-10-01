using Microsoft.ML.Data;

namespace App02PredictBPM;

public class BpmData
{
    [LoadColumn(0)]
    public string WorkoutId { get; set; } = null!;
    [LoadColumn(1)]
    public float DurationMinutes { get; set; }
    [LoadColumn(2)]
    public float DistanceKm { get; set; }
    [LoadColumn(3)]
    public float AverageHeartRateBpm { get; set; }
}
