using Microsoft.ML.Data;

namespace App01BinaryModelUsingMLNet.Models;

public class News
{
    // [NoColumn]
    // [LoadColumn(0)]
    // public int Id { get; set; }
    [LoadColumn(1)]
    public string Title { get; set; } = null!;
    [LoadColumn(2)]
    public bool Result { get; set; }
}
