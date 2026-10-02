namespace App04NewsPredict;

public class NewsPrediction
{
    public string Content { get; set; } = null!;
    public string PredictedLabel { get; set; } = null!;
    public float[] Score { get; set; } = null!;
}
