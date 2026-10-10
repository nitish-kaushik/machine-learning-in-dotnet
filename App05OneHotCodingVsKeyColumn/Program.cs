using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

var mlContext = new MLContext();

var trainingData = mlContext.Data.LoadFromEnumerable(
[
    new ColorInput { Color = "Red" },
        new ColorInput { Color = "Blue" },
        new ColorInput { Color = "Green" },
        new ColorInput { Color = "Red" }
]);

var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(
    outputColumnName: "ColorEncoded",
    inputColumnName: nameof(ColorInput.Color),
    outputKind: OneHotEncodingEstimator.OutputKind.Indicator);


// Learn the category dictionary from training data.
var encoder = pipeline.Fit(trainingData);

// Apply the learned mapping.
var encodedData = encoder.Transform(trainingData);

var rows = mlContext.Data.CreateEnumerable<ColorOutput>(
    encodedData,
    reuseRowObject: false);

foreach (var row in rows)
{
    Console.WriteLine(
        $"{row.Color} -> [{string.Join(", ", row.ColorEncoded)}]");
}

public sealed class ColorInput
{
    public string Color { get; set; } = string.Empty;
}

public sealed class ColorOutput
{
    public string Color { get; set; } = string.Empty;

    public float[] ColorEncoded { get; set; } = [];
}
