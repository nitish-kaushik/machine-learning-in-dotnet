using Microsoft.ML.Data;

namespace App03EmailSpamPrediction;

public class EmailData
{
    [LoadColumn(0)]  public string EmailId { get; set; } = "";
    [LoadColumn(1)]  public string SenderEmail { get; set; } = "";
    [LoadColumn(2)]  public string SenderName { get; set; } = "";
    [LoadColumn(3)]  public string RecipientEmail { get; set; } = "";
    [LoadColumn(4)]  public string EmailBody { get; set; } = "";
    [LoadColumn(5)]  public float SenderDomainReputation { get; set; }
    [LoadColumn(6)]  public float SenderFirstSeenDaysAgo { get; set; }
    [LoadColumn(7)]  public float HasUnsubscribeLink { get; set; }
    [LoadColumn(8)]  public float AttachmentCount { get; set; }
    [LoadColumn(9)]  public float IsExternalSender { get; set; }
    [LoadColumn(10)] public float SentHourUtc { get; set; }
    [LoadColumn(11)] public float RecipientCount { get; set; }
    [LoadColumn(12)] public bool IsSpam { get; set; }
}
