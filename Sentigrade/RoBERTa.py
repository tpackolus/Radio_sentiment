import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['WANDB_DISABLED'] = 'true'
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from imblearn.over_sampling import RandomOverSampler

new_data_df = pd.read_csv('transdata.csv')

sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
new_data_df['label'] = new_data_df['scale'].map(sentiment_mapping)

new_data_df['comment'] = new_data_df['comment'].astype(str)

print(new_data_df['label'].value_counts())

oversample = RandomOverSampler(sampling_strategy='auto')
X_resampled, y_resampled = oversample.fit_resample(new_data_df['comment'].values.reshape(-1, 1), new_data_df['label'])

resampled_df = pd.DataFrame({'comment': X_resampled.flatten(), 'label': y_resampled})

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)

def tokenize_function(text_list):
    return tokenizer(text_list, padding="max_length", truncation=True, max_length=512)

resampled_encodings = tokenize_function(resampled_df['comment'].tolist())

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

resampled_dataset = ToxicDataset(resampled_encodings, resampled_df['label'].values)

training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    logging_dir='./logs',
    logging_steps=10,
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=resampled_dataset,
    eval_dataset=resampled_dataset
)

trainer.train()

new_predictions = trainer.predict(resampled_dataset)
new_predicted_labels = np.argmax(new_predictions.predictions, axis=1)

resampled_df['predicted_label'] = new_predicted_labels
resampled_df.to_csv('new_predictions.csv', index=False)

new_labels = resampled_df['label'].values

accuracy = accuracy_score(new_labels, new_predicted_labels)
f1 = f1_score(new_labels, new_predicted_labels, average='weighted', zero_division=0)
precision = precision_score(new_labels, new_predicted_labels, average='weighted', zero_division=0)
recall = recall_score(new_labels, new_predicted_labels, average='weighted', zero_division=0)
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Overall F1 Score: {f1:.4f}")
print(f"Overall Precision: {precision:.4f}")
print(f"Overall Recall: {recall:.4f}")

report = classification_report(new_labels, new_predicted_labels, target_names=['negative', 'neutral', 'positive'], output_dict=True, zero_division=0)
print("\nDetailed Metrics:")
for label, metrics in report.items():
    if label in ['negative', 'neutral', 'positive']:
        print(f"\nLabel: {label.capitalize()}")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

cm = confusion_matrix(new_labels, new_predicted_labels, labels=[0, 1, 2])
print("\nConfusion Matrix Details:")
labels_list = ['negative', 'neutral', 'positive']
for i, label in enumerate(labels_list):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - (tp + fn + fp)
    print(f"{label.capitalize()} - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

print("Predictions on the new dataset have been saved to 'new_predictions.csv'.")
