using Microsoft.ML.Data;

namespace App04NewsPredict;

public class NewsData
{
    [LoadColumn(1)]
    public string Content { get; set; } = null!;
    [LoadColumn(2)]
    public string Category { get; set; } = null!;
}
