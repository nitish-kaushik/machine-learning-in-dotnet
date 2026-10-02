using App03EmailSpamPrediction;
using Microsoft.ML;

var ml = new MLContext();

var filePath = Path.Combine(Environment.CurrentDirectory, "data.csv");

var data = ml.Data.LoadFromTextFile<EmailData>(
    path: filePath,
    hasHeader: true,
    separatorChar: ',');

var trainTestData = ml.Data.TrainTestSplit(data, testFraction: 0.2);

var estimator = ml.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(EmailData.IsSpam))
    .Append(ml.Transforms.Concatenate(
        "Features",
        nameof(EmailData.EmailBody),
        nameof(EmailData.SenderDomainReputation),
        nameof(EmailData.SenderFirstSeenDaysAgo),
        nameof(EmailData.HasUnsubscribeLink),
        nameof(EmailData.AttachmentCount),
        nameof(EmailData.IsExternalSender),
        nameof(EmailData.SentHourUtc),
        nameof(EmailData.RecipientCount)))
    // .Append(ml.Transforms.DropColumns(
    //     nameof(EmailData.EmailId),
    //     nameof(EmailData.SenderEmail),
    //     nameof(EmailData.SenderName),
    //     nameof(EmailData.RecipientEmail)))
    .Append(ml.Transforms.SelectColumns(
        nameof(EmailData.IsSpam),
        "Features"));

var pipeline = estimator.Append(ml.BinaryClassification.Trainers.SdcaLogisticRegression());

var transformer = pipeline.Fit(trainTestData.TrainSet);

var transformedData = transformer.Transform(trainTestData.TestSet);
