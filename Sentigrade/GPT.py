import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['WANDB_DISABLED'] = 'true'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
import torch.nn as nn

data_df = pd.read_csv('transdata.csv')

data_df['label'] = data_df['scale'].apply(lambda x: 2 if x == 'positive' else (1 if x == 'neutral' else 0))

data_df['comment'] = data_df['comment'].astype(str)

train_df, temp_df = train_test_split(data_df, test_size=0.3, random_state=42, stratify=data_df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

tokenizer = AutoTokenizer.from_pretrained("Milos/slovak-gpt-j-405M")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained("Milos/slovak-gpt-j-405M")
model.resize_token_embeddings(len(tokenizer))

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)
    
    def forward(self, features):
        x = features[:, -1, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CustomModel(nn.Module):
    def __init__(self, model, hidden_size, num_labels):
        super(CustomModel, self).__init__()
        self.model = model
        self.classifier = ClassificationHead(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        logits = self.classifier(hidden_states)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_proj.out_features), labels.view(-1))
        return (loss, logits) if loss is not None else logits

hidden_size = model.config.hidden_size
num_labels = 3
custom_model = CustomModel(model, hidden_size, num_labels)

def tokenize_function(text_list):
    return tokenizer(text_list, padding="max_length", truncation=True, max_length=512)

train_encodings = tokenize_function(train_df['comment'].tolist())
val_encodings = tokenize_function(val_df['comment'].tolist())
test_encodings = tokenize_function(test_df['comment'].tolist())

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
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=custom_model,
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

print(f"Overall Accuracy: {accuracy}")
print(f"Overall F1 Score: {f1}")
print(f"Overall Precision: {precision}")
print(f"Overall Recall: {recall}")

report = classification_report(val_labels, val_preds, target_names=['negative', 'neutral', 'positive'], output_dict=True)
print("\nDetailed Metrics:")
for label, metrics in report.items():
    if label in ['negative', 'neutral', 'positive']:
        print(f"\nLabel: {label}")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

cm = confusion_matrix(val_labels, val_preds, labels=[0, 1, 2])
print("\nConfusion Matrix Details:")
labels_list = ['negative', 'neutral', 'positive']
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
label_df = pd.read_csv('test_with_labels.csv')

correct_predictions = (pred_df['predicted_label'] == label_df['label']).sum()
total_predictions = len(pred_df)

print(f"Model correctly predicted {correct_predictions} out of {total_predictions} predictions.")
print(f"Model accuracy on the test set: {correct_predictions / total_predictions:.2f}")

detailed_metrics = {
    'Class': labels_list,
    'Accuracy': [accuracy] * 3,
    'Precision': precision_score(val_labels, val_preds, average=None, zero_division=0),
    'Recall': recall_score(val_labels, val_preds, average=None, zero_division=0),
    'F1-score': f1_score(val_labels, val_preds, average=None, zero_division=0),
    'True Positives': [cm[i, i] for i in range(3)],
    'True Negatives': [cm.sum() - (cm[i, i] + cm[i, :].sum() - cm[i, i] + cm[:, i].sum() - cm[i, i]) for i in range(3)],
    'False Positives': [cm[:, i].sum() - cm[i, i] for i in range(3)],
    'False Negatives': [cm[i, :].sum() - cm[i, i] for i in range(3)],
}

detailed_metrics_df = pd.DataFrame(detailed_metrics)
print(detailed_metrics_df)
