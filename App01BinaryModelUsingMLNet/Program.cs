using App01BinaryModelUsingMLNet.Models;
using Microsoft.ML;

var mlContext = new MLContext();

// Load the data
var dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "info.csv");
var dataView = mlContext.Data.LoadFromTextFile<News>(dataPath, hasHeader: true, separatorChar: ',');

// pipeline
var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(News.Title))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: nameof(News.Result), featureColumnName: "Features"));

// Train the model
var model = pipeline.Fit(dataView);

// PredictionEngine
var predictionEngine = mlContext.Model.CreatePredictionEngine<News, NewsPrediction>(model);

// Test the model with a sample news title
var sampleNews = new News { Title = "I am dissatisfied with ML.Net" };

var prediction = predictionEngine.Predict(sampleNews);

Console.WriteLine($"Prediction: {prediction.PredictedLabel}, Probability: {prediction.Probability}, Score: {prediction.Score}");
