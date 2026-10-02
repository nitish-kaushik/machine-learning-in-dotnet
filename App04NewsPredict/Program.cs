using App04NewsPredict;
using Microsoft.ML;

var mlContext = new MLContext();

var filePath = Path.Combine(AppContext.BaseDirectory, "data.csv");

var data = mlContext.Data.LoadFromTextFile<NewsData>(filePath, hasHeader: true, separatorChar: ',');

var trainTestSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(NewsData.Content))
    .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(NewsData.Category)))
    .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

var model = pipeline.Fit(trainTestSplit.TrainSet);

var transformedTestData = model.Transform(trainTestSplit.TestSet);

var result = mlContext.Data.CreateEnumerable<NewsPrediction>(transformedTestData, reuseRowObject: false);

foreach (var item in result)
{
    Console.WriteLine($" Title: {item.Content}  -  {item.PredictedLabel}");
}
