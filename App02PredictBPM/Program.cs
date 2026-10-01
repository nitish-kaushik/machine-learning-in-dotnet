using App02PredictBPM;
using Microsoft.ML;

var mlContext = new MLContext();

var dataPath = Path.Combine(AppContext.BaseDirectory, "data.csv");

var data = mlContext.Data.LoadFromTextFile<BpmData>(dataPath, hasHeader: true, separatorChar: ',');

var trainTestSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

var estimator =
    mlContext.Transforms.Concatenate("Features", nameof(BpmData.DurationMinutes), nameof(BpmData.DistanceKm));

var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: nameof(BpmData.AverageHeartRateBpm), featureColumnName: "Features");

var pipeline = estimator.Append(trainer);

var transformer = pipeline.Fit(trainTestSplit.TrainSet);

var transformedTrainData = transformer.Transform(trainTestSplit.TrainSet);

var predictionEngine = mlContext.Model.CreatePredictionEngine<BpmData, BpmPrediction>(transformer);

var dummyInput = new BpmData
{
    WorkoutId = "dummy",
    DurationMinutes = 30,
    DistanceKm = 5.0f
};

var prediction = predictionEngine.Predict(dummyInput);

Console.WriteLine($"Predicted Average Heart Rate (BPM): {prediction.AverageHeartRateBpm}");
