using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

namespace App06MLConcepts;

public static class TextCatalog
{
    private static readonly Review[] TrainingRows =
    [
        new() { ReviewText = "please This is a great product!" },
        new() { ReviewText = "I love an item, it works perfectly." },
        new() { ReviewText = "Not worth the money, very 123 disappointed." },
        new() { ReviewText = "Excellent quality and fast  567shipping." },
        new() { ReviewText = "Terrible experience, will not buy again." }
    ];

    public static void RunNormalizeText(MLContext mlContext)
    {
        var train = mlContext.Data.LoadFromEnumerable(TrainingRows);

        var estimator = mlContext.Transforms.Text
            .NormalizeText("NormalizedReviewText"
                , "ReviewText"
                , TextNormalizingEstimator.CaseMode.Upper,
                keepDiacritics: false,
                keepPunctuations: false,
                keepNumbers: false);

        var model = estimator.Fit(train);

        var transformedData = model.Transform(train);

        var preview = transformedData.Preview();

        foreach (var row in preview.RowView)
        {
            foreach (var col in row.Values)
            {
                Console.WriteLine($"{col.Key}: {col.Value}");
            }

            Console.WriteLine();
        }
    }

    public static void RunTokenizeIntoWords(MLContext mlContext)
    {
        var train = mlContext.Data.LoadFromEnumerable(TrainingRows);

        var estimator = mlContext.Transforms.Text
            .TokenizeIntoCharactersAsKeys("TokenizeIntoCharactersText"
                , "ReviewText");

        var model = estimator.Fit(train);

        var transformedData = model.Transform(train);

        var preview = transformedData.Preview();

        foreach (var row in preview.RowView)
        {
            foreach (var col in row.Values)
            {
                Console.WriteLine($"{col.Key}: {col.Value}");
            }

            Console.WriteLine();
        }
    }

    public static void RunStoppingWordsRemoving(MLContext mlContext)
    {
        var train = mlContext.Data.LoadFromEnumerable(TrainingRows);

        // var estimator = mlContext.Transforms.Text.TokenizeIntoWords("TokenizeIntoWordsText", "ReviewText")
        //     .Append(mlContext.Transforms.Text.RemoveDefaultStopWords("RemoveDefaultStopWordsText", "TokenizeIntoWordsText"));
        //

        string[] stopWords = ["please", "thankyou", "the", "and", "will", "not"];

        var estimator = mlContext.Transforms.Text.TokenizeIntoWords("TokenizeIntoWordsText", "ReviewText")
            .Append(mlContext.Transforms.Text.RemoveStopWords("RemoveDefaultStopWordsText", "TokenizeIntoWordsText", stopWords));

        var model = estimator.Fit(train);
        var transformedData = model.Transform(train);
        var preview = transformedData.Preview();

        foreach (var row in preview.RowView)
        {
            foreach (var col in row.Values)
            {
                Console.WriteLine($"{col.Key}: {col.Value}");
            }

            Console.WriteLine();
        }
    }
}

public class Review
{
    public string ReviewText { get; set; } = null!;
}
