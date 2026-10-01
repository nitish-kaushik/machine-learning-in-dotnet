using Microsoft.ML.Data;

namespace App01BinaryModelUsingMLNet.Models;

public class NewsPrediction
{
    public string Title { get; set; } = null!;
    [ColumnName("PredictedLabel")]
    public bool Answer { get; set; }

    public float Probability { get; set; }

    public float Score { get; set; }
}
