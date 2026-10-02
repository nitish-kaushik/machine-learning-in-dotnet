using Microsoft.ML.Data;

namespace App03EmailSpamPrediction;

public class EmailPrediction
{
    [ColumnName("PredictedLabel")]
    public bool IsSpam { get; set; }
}
