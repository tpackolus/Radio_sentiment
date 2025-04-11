import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['WANDB_DISABLED'] = 'true'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

data_df = pd.read_csv('machova.csv')

data_df = data_df.dropna(subset=['airline_sentiment'])
data_df = data_df[data_df['airline_sentiment'].isin(['positive', 'negative'])]
data_df['label'] = data_df['airline_sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

data_df['text'] = data_df['text'].astype(str)

train_df, temp_df = train_test_split(data_df, test_size=0.3, random_state=42, stratify=data_df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=2)

def tokenize_function(text_list):
    return tokenizer(text_list, padding="max_length", truncation=True, max_length=512)

train_texts = train_df['text'].tolist()
val_texts = val_df['text'].tolist()
test_texts = test_df['text'].tolist()

print("First few training texts:", train_texts[:5])
print("First few validation texts:", val_texts[:5])
print("First few test texts:", test_texts[:5])

train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)
test_encodings = tokenize_function(test_texts)

class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels) if self.labels is not None else len(self.encodings['input_ids'])

train_dataset = ToxicDataset(train_encodings, train_df['label'].values)
val_dataset = ToxicDataset(val_encodings, val_df['label'].values)
test_dataset = ToxicDataset(test_encodings)

training_args = TrainingArguments(
    output_dir='./resultsDistil',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logsDistil',
    logging_steps=10,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

eval_predictions = trainer.predict(val_dataset)
val_preds = torch.argmax(torch.tensor(eval_predictions.predictions), axis=1).numpy()
val_labels = val_df['label'].values

accuracy = accuracy_score(val_labels, val_preds)
f1 = f1_score(val_labels, val_preds, average='weighted')
precision = precision_score(val_labels, val_preds, average='weighted')
recall = recall_score(val_labels, val_preds, average='weighted')
cm = confusion_matrix(val_labels, val_preds, labels=[0, 1])

print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Overall Precision: {precision:.4f}")
print(f"Overall Recall: {recall:.4f}")
print(f"Overall F1 Score: {f1:.4f}")

report = classification_report(val_labels, val_preds, target_names=['negative', 'positive'], output_dict=True)

print("\nDetailed Metrics:")
for label, metrics in report.items():
    if label in ['negative', 'positive']:
        print(f"\nLabel: {label.capitalize()}")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

print("\nConfusion Matrix Details:")
labels_list = ['negative', 'positive']
for i, label in enumerate(labels_list):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - (tp + fn + fp)
    print(f"{label.capitalize()} - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

test_predictions = trainer.predict(test_dataset).predictions
predicted_labels = np.argmax(test_predictions, axis=1)

test_df_no_label = test_df.drop(columns=['label'])
test_df_no_label['predicted_label'] = predicted_labels
test_df_no_label.to_csv('predictions.csv', index=False)

pred_df = pd.read_csv('predictions.csv')
test_labels_file = 'test_with_labels.csv'
if os.path.exists(test_labels_file):
    label_df = pd.read_csv(test_labels_file)

    correct_predictions = (pred_df['predicted_label'] == label_df['label']).sum()
    total_predictions = len(pred_df)

    print(f"Model trafil správne {correct_predictions} z {total_predictions} predikcií.")
    print(f"Presnost modelu na testovacom sete: {correct_predictions / total_predictions:.2f}")
else:
    print(f"File {test_labels_file} does not exist. Skipping comparison with test labels.")

detailed_metrics = {
    'Class': labels_list,
    'Accuracy': [accuracy] * 2,
    'Precision': [report[label]['precision'] for label in labels_list],
    'Recall': [report[label]['recall'] for label in labels_list],
    'F1-score': [report[label]['f1-score'] for label in labels_list],
    'True Positives': [cm[i, i] for i in range(2)],
    'True Negatives': [cm.sum() - (cm[i, i] + cm[i, :].sum() - cm[i, i] + cm[:, i].sum() - cm[i, i]) for i in range(2)],
    'False Positives': [cm[:, i].sum() - cm[i, i] for i in range(2)],
    'False Negatives': [cm[i, :].sum() - cm[i, i] for i in range(2)],
}

detailed_metrics_df = pd.DataFrame(detailed_metrics)
print(detailed_metrics_df)

new_data_df = pd.read_csv('machova.csv')

new_data_df = new_data_df.dropna(subset=['airline_sentiment'])
new_data_df = new_data_df[new_data_df['airline_sentiment'].isin(['positive', 'negative'])]
new_data_df['label'] = new_data_df['airline_sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
new_data_df['text'] = new_data_df['text'].astype(str)

new_texts = new_data_df['text'].tolist()
new_encodings = tokenize_function(new_texts)
new_dataset = ToxicDataset(new_encodings)

new_predictions = trainer.predict(new_dataset).predictions
new_predicted_labels = np.argmax(new_predictions, axis=1)

new_data_df['predicted_label'] = new_predicted_labels
new_data_df.to_csv('new_predictions.csv', index=False)

print("Predictions for the new dataset have been saved to 'new_predictions.csv'.")
