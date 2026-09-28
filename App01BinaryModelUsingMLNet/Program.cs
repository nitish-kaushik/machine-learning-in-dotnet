using App01BinaryModelUsingMLNet.Models;
using Microsoft.ML;
using Microsoft.ML.Data;

// var dummyInMemoryData = new List<News>
// {
//     new News { Title = "I am satisfied with ML.Net", Result = true },
//     new News { Title = "I am dissatisfied with ML.Net", Result = false },
//     new News { Title = "ML.Net is great for machine learning", Result = true },
//     new News { Title = "ML.Net is not good for machine learning", Result = false }
// };

var mlContext = new MLContext();

var textLoader = mlContext.Data.CreateTextLoader(new TextLoader.Options
{
    Separators = [','],
    HasHeader = true,
    Columns =
    [
        new TextLoader.Column(nameof(News.Title), DataKind.String, 1),
        new TextLoader.Column(nameof(News.Result), DataKind.Boolean, 2)
    ]
});

// Load the data
var dataPath1 = Path.Combine(Environment.CurrentDirectory, "Data", "info.csv");
var dataPath2 = Path.Combine(Environment.CurrentDirectory, "Data", "info2.csv");

var dataView = textLoader.Load(new MultiFileSource(dataPath1, dataPath2));
var trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.1);
var trainSet = trainTestSplit.TrainSet;
var testSet = trainTestSplit.TestSet;

// var testData = mlContext.Data.SkipRows(dataView, 5);
// var rows = mlContext.Data.TakeRows(testData, 10);
// var shuffleRows = mlContext.Data.ShuffleRows(rows);
// var previewData = mlContext.Data.CreateEnumerable<News>(shuffleRows, reuseRowObject: false).ToList();
//
// foreach (var sample in previewData)
// {
//     Console.WriteLine($"Title: {sample.Title}, Result: {sample.Result}");
// }

//var preview = dataView.Preview(maxRows:10);

//
// var dataView = mlContext.Data.LoadFromTextFile<News>(dataPath1, hasHeader: true, separatorChar: ',');
// var dataView2 = mlContext.Data.LoadFromTextFile<News>(dataPath2, hasHeader: true, separatorChar: ',');

//var dataView = mlContext.Data.LoadFromEnumerable(dummyInMemoryData);

// pipeline
var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(News.Title))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: nameof(News.Result), featureColumnName: "Features"));

// Train the model
var model = pipeline.Fit(trainSet);

var evaluation = mlContext.BinaryClassification.Evaluate(model.Transform(testSet), labelColumnName: nameof(News.Result));

Console.WriteLine($"Accuracy: {evaluation.Accuracy}, AUC: {evaluation.AreaUnderRocCurve}, F1 Score: {evaluation.F1Score}");

// PredictionEngine
var predictionEngine = mlContext.Model.CreatePredictionEngine<News, NewsPrediction>(model);

// Test the model with a sample news title
var sampleNews = new News { Title = "I am dissatisfied with ML.Net" };

var prediction = predictionEngine.Predict(sampleNews);

Console.WriteLine($"Prediction: {prediction.Answer}, Probability: {prediction.Probability}, Score: {prediction.Score}");
