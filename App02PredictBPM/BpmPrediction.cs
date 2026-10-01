using Microsoft.ML.Data;

namespace App02PredictBPM;

public class BpmPrediction
{
    [ColumnName("Score")]
    public float AverageHeartRateBpm { get; set; }
}
