using Microsoft.ML;
using Microsoft.ML.Transforms;

namespace App06MLConcepts;

public static class ReplaceAndIndicateMissingValue
{
    public static void Run(MLContext mlContext)
    {

        var reference = mlContext.Data.LoadFromEnumerable([
            new ReadingRow { Reading = 100 },
            new ReadingRow { Reading = 120 },
            new ReadingRow { Reading = float.NaN },
            new ReadingRow { Reading = 140 }
        ]);
        var probes = mlContext.Data.LoadFromEnumerable([
            new ReadingRow { Reading = float.NaN },
            new ReadingRow { Reading = 0 },
            new ReadingRow { Reading = 120 }
        ]);

        var pipeline = mlContext.Transforms.ReplaceMissingValues(
            columns:
            [
                new InputOutputColumnPair(outputColumnName: nameof(ReadingRow.Reading),
                    inputColumnName: nameof(ReadingRow.Reading))
            ],
            replacementMode: MissingValueReplacingEstimator.ReplacementMode.DefaultValue);

        var indicateMissingPipeline = mlContext.Transforms.IndicateMissingValues(
            columns:
            [
                new InputOutputColumnPair(outputColumnName: "IsReadingMissing", inputColumnName: nameof(ReadingRow.Reading))
            ]);

        var estimator = indicateMissingPipeline.Append(pipeline);

        var model = estimator.Fit(reference);
        var transformedData = model.Transform(probes);
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

    private sealed class ReadingRow
    {
        public float Reading { get; set; }
    }
}
