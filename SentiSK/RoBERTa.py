import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['WANDB_DISABLED'] = 'true'
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

data_df = pd.read_csv('Nove_data.csv')

print("Columns in the DataFrame:", data_df.columns)

if 'comment' in data_df.columns:
    data_df.rename(columns={'comment': 'text'}, inplace=True)

print("First few rows of the DataFrame:", data_df.head())

data_df = data_df.dropna(subset=['airline_sentiment', 'text'])

data_df['label'] = data_df['airline_sentiment'].apply(lambda x: 1 if x == 'negative' else 0)

data_df['text'] = data_df['text'].astype(str)

train_df, temp_df = train_test_split(data_df, test_size=0.3, random_state=42, stratify=data_df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print("First few rows of the training set:", train_df.head())
print("First few rows of the validation set:", val_df.head())
print("First few rows of the test set:", test_df.head())

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)

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

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_df['label']), y=train_df['label'])
class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()

class WeightedLoss(torch.nn.CrossEntropyLoss):
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
    output_dir='./resultsroberta',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir='./logsroberta',
    logging_steps=10,
    evaluation_strategy="epoch"
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

eval_predictions = trainer.predict(val_dataset)
val_preds = np.argmax(eval_predictions.predictions, axis=1)
val_labels = val_df['label'].values

accuracy = accuracy_score(val_labels, val_preds)
f1 = f1_score(val_labels, val_preds, average='weighted')
precision = precision_score(val_labels, val_preds, average='weighted')
recall = recall_score(val_labels, val_preds, average='weighted')
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

report = classification_report(val_labels, val_preds, target_names=['positive', 'negative'], output_dict=True)
print("\nDetailed Metrics:")
for label, metrics in report.items():
    if label in ['positive', 'negative']:
        print(f"\nLabel: {label.capitalize()}")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

cm = confusion_matrix(val_labels, val_preds, labels=[0, 1])
labels_list = ['positive', 'negative']
print("\nConfusion Matrix Details:")
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

test_labels_file = 'test_with_labels.csv'
if os.path.exists(test_labels_file):
    label_df = pd.read_csv(test_labels_file)

    correct_predictions = (pred_df['predicted_label'] == label_df['label']).sum()
    total_predictions = len(pred_df)

    print(f"Model correctly predicted {correct_predictions} out of {total_predictions} predictions.")
    print(f"Model accuracy on the test set: {correct_predictions / total_predictions:.2f}")
else:
    print(f"File {test_labels_file} does not exist. Skipping comparison with test labels.")
