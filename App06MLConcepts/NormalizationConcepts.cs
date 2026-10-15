using Microsoft.ML;

namespace App06MLConcepts;

public static class NormalizationConcepts
{
    public static void Run(MLContext mlContext)
    {
        var trainingRows = new[]
        {
            new AmountRow { Amount = 10, Band = "Low" },
            new AmountRow { Amount = 20, Band = "Low" },
            new AmountRow { Amount = 30, Band = "Low" },
            new AmountRow { Amount = 40, Band = "Low" },
            new AmountRow { Amount = 80, Band = "High" },
            new AmountRow { Amount = 90, Band = "High" },
            new AmountRow { Amount = 100, Band = "High" },
            new AmountRow { Amount = 1000, Band = "High" }
        };
        var train = mlContext.Data.LoadFromEnumerable(trainingRows);
        var testRows = mlContext.Data.LoadFromEnumerable([
            new AmountInput { Amount = 15 },
            new AmountInput { Amount = 90 },
            new AmountInput { Amount = 500 }
        ]);

        var pipeline = mlContext.Transforms.NormalizeMinMax(nameof(AmountInput.Amount), nameof(AmountInput.Amount), fixZero: false);

        // formula - (x - min) / (max - min)
        // min = 10, max = 1000
        // 10-10 / 1000-10
        // 20-10 / 1000-10

        // 15-10/ 1000-10 = 5/990 = 0.00505

        var model = pipeline.Fit(train);
        var transformedData = model.Transform(testRows);
        var previewData = transformedData.Preview();
        foreach (var row in previewData.RowView)
        {
            foreach (var col in row.Values)
            {
                Console.WriteLine($"{col.Key}: {col.Value}");
            }

            Console.WriteLine();
        }
    }

    public static void RunNormalizeMeanVariance(MLContext mlContext)
    {
        var trainingRows = new[]
        {
            new AmountRow { Amount = 10, Band = "Low" },
            new AmountRow { Amount = 20, Band = "Low" },
            new AmountRow { Amount = 30, Band = "Low" }
        };
        var train = mlContext.Data.LoadFromEnumerable(trainingRows);

        var pipeline = mlContext.Transforms.NormalizeMeanVariance(nameof(AmountInput.Amount),
            nameof(AmountInput.Amount), fixZero: false);

        // formula - (x - mean) / standard deviation
        // mean = 20
        // (10-20) sqr = 100, (20-20) sqr = 0, (30-20) sqr = 100
        // variance = 100+0+100 / 3 = 66.6667
        // standard deviation = sqrt(66.6667) = 8.16497

        // 10-20 / 8.16497 = -1.22474
        // 20-20 / 8.16497 = 0
        // 30-20 / 8.16497 = 1.22474

        var model = pipeline.Fit(train);
        var transformedData = model.Transform(train);
        var previewData = transformedData.Preview();
        foreach (var row in previewData.RowView)
        {
            foreach (var col in row.Values)
            {
                Console.WriteLine($"{col.Key}: {col.Value}");
            }

            Console.WriteLine();
        }
    }
}

public class AmountInput
{
    public float Amount { get; set; }
}
public sealed class AmountRow : AmountInput
{
    public string Band { get; set; } = "";
}
