import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['WANDB_DISABLED'] = 'true'
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss

df = pd.read_csv('transdata.csv')

sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
df['label'] = df['scale'].map(sentiment_mapping)

df['comment'] = df['comment'].astype(str)

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)

def tokenize_function(text_list):
    return tokenizer(text_list, padding="max_length", truncation=True, max_length=512)

data_encodings = tokenize_function(df['comment'].tolist())

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

data_dataset = ToxicDataset(data_encodings, df['label'].values)

class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1, 2], y=df['label'].values)
class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()

class WeightedLoss(CrossEntropyLoss):
    def __init__(self, weight):
        super().__init__(weight=weight)

loss_fn = WeightedLoss(weight=class_weights)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(model.device)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=data_dataset,
    eval_dataset=data_dataset
)

trainer.train()

predictions = trainer.predict(data_dataset).predictions
predicted_labels = np.argmax(predictions, axis=1)

df['predicted_label'] = predicted_labels
df.to_csv('predictions.csv', index=False)

labels = df['label'].values

accuracy = accuracy_score(labels, predicted_labels)
f1 = f1_score(labels, predicted_labels, average='weighted', zero_division=0)
precision = precision_score(labels, predicted_labels, average='weighted', zero_division=0)
recall = recall_score(labels, predicted_labels, average='weighted', zero_division=0)
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Overall F1 Score: {f1:.4f}")
print(f"Overall Precision: {precision:.4f}")
print(f"Overall Recall: {recall:.4f}")

report = classification_report(labels, predicted_labels, target_names=['negative', 'neutral', 'positive'], output_dict=True, zero_division=0)
print("\nDetailed Metrics:")
for label, metrics in report.items():
    if label in ['negative', 'neutral', 'positive']:
        print(f"\nLabel: {label.capitalize()}")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

cm = confusion_matrix(labels, predicted_labels, labels=[0, 1, 2])
print("\nConfusion Matrix Details:")
labels_list = ['negative', 'neutral', 'positive']
for i, label in enumerate(labels_list):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - (tp + fn + fp)
    print(f"{label.capitalize()} - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

detailed_metrics = {
    'Class': labels_list,
    'Accuracy': [accuracy] * 3,
    'Precision': precision_score(labels, predicted_labels, average=None, zero_division=0),
    'Recall': recall_score(labels, predicted_labels, average=None, zero_division=0),
    'F1-score': f1_score(labels, predicted_labels, average=None, zero_division=0),
    'True Positives': [cm[i, i] for i in range(3)],
    'True Negatives': [cm.sum() - (cm[i, i] + cm[i, :].sum() - cm[i, i] + cm[:, i].sum() - cm[i, i]) for i in range(3)],
    'False Positives': [cm[:, i].sum() - cm[i, i] for i in range(3)],
    'False Negatives': [cm[i, :].sum() - cm[i, i] for i in range(3)],
}

detailed_metrics_df = pd.DataFrame(detailed_metrics)
print(detailed_metrics_df)

print("Predictions on the dataset have been saved to 'predictions.csv'.")
