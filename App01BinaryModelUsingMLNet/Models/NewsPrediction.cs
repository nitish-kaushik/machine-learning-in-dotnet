using Microsoft.ML.Data;

namespace App01BinaryModelUsingMLNet.Models;

public class NewsPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Answer { get; set; }

    public float Probability { get; set; }

    public float Score { get; set; }
}
