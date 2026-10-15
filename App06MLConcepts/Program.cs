using App06MLConcepts;
using Microsoft.ML;
using Microsoft.ML.Transforms;

var mlContext = new MLContext();

//ReplaceAndIndicateMissingValue.Run(mlContext);

//NormalizationConcepts.Run(mlContext);
NormalizationConcepts.RunNormalizeMeanVariance(mlContext);
