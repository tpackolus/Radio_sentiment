import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['WANDB_DISABLED'] = 'true'
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

df = pd.read_csv('machova.csv')

df = df[df['airline_sentiment'].isin(['positive', 'negative'])]
df['label'] = df['airline_sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
df['text'] = df['text'].astype(str)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    inputs = ["classify: " + str(example) for example in examples['text']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
    labels = [str(label) for label in examples['label']]
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(labels, max_length=2, padding='max_length', truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

columns_to_remove = ['airline_sentiment', 'tweet_id\t', 'tweet_created', 'name', 'text', 'label']
train_dataset = train_dataset.remove_columns(columns_to_remove)
val_dataset = val_dataset.remove_columns(columns_to_remove)

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()

predictions = trainer.predict(val_dataset)
logits = predictions.predictions

if isinstance(logits, tuple):
    logits = logits[0]

preds = torch.argmax(torch.tensor(logits), dim=-1)

preds = [pred_seq[0].item() for pred_seq in preds]
labels = predictions.label_ids

labels = [label_seq[0] for label_seq in labels]

preds = [tokenizer.decode([pred], skip_special_tokens=True).strip() for pred in preds]
labels = [tokenizer.decode([label], skip_special_tokens=True).strip() for label in labels]

preds = [int(pred) if pred.isdigit() else 0 for pred in preds]
labels = [int(label) if label.isdigit() else 0 for label in labels]

if set(preds) <= {0, 1} and set(labels) <= {0, 1}:
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=None)
    precision = precision_score(labels, preds, average=None)
    recall = recall_score(labels, preds, average=None)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall Precision: {precision.mean():.4f}")
    print(f"Overall Recall: {recall.mean():.4f}")
    print(f"Overall F1 Score: {f1.mean():.4f}")
    
    print("\nDetailed Metrics:")
    labels_list = ['negative', 'positive']
    for i, label in enumerate(labels_list):
        print(f"\nLabel: {label}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision[i]:.4f}")
        print(f"Recall: {recall[i]:.4f}")
        print(f"F1 Score: {f1[i]:.4f}")
    
    print("\nConfusion Matrix Details:")
    for i, label in enumerate(labels_list):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        print(f"{label.capitalize()} - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    print("\nOverall Metrics in Tabular Format:")
    metrics = {
        'Accuracy': [accuracy],
        'Precision': [precision.mean()],
        'Recall': [recall.mean()],
        'F1-score': [f1.mean()],
        'True Positives': [sum(tp for i, tp in enumerate(cm.diagonal()))],
        'True Negatives': [sum(tn for i, tn in enumerate(cm.diagonal()))],
        'False Positives': [sum(fp for i, fp in enumerate(cm.sum(axis=0) - cm.diagonal()))],
        'False Negatives': [sum(fn for i, fn in enumerate(cm.sum(axis=1) - cm.diagonal()))],
    }

    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)

    detailed_metrics = {
        'Class': labels_list,
        'Accuracy': [accuracy] * 2,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'True Positives': [cm[i, i] for i in range(2)],
        'True Negatives': [sum(cm[i, j] for j in range(2) if j != i) for i in range(2)],
        'False Positives': [sum(cm[j, i] for j in range(2) if j != i) for i in range(2)],
        'False Negatives': [sum(cm[i, j] for j in range(2) if j != i) for i in range(2)],
    }

    detailed_metrics_df = pd.DataFrame(detailed_metrics)
    print(detailed_metrics_df)
else:
    print("Predictions and/or labels are not within the expected set of {0, 1}, skipping detailed metric calculation.")

new_data_df = pd.read_csv('machova.csv')

new_data_df = new_data_df[new_data_df['airline_sentiment'].isin(['positive', 'negative'])]
new_data_df['label'] = new_data_df['airline_sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
new_data_df['text'] = new_data_df['text'].astype(str)

new_texts = new_data_df['text'].tolist()
new_encodings = preprocess_function({'text': new_texts})
new_dataset = Dataset.from_dict(new_encodings)

new_predictions = trainer.predict(new_dataset).predictions
new_logits = new_predictions

if isinstance(new_logits, tuple):
    new_logits = new_logits[0]

new_preds = torch.argmax(torch.tensor(new_logits), dim=-1)

new_preds = [pred_seq[0].item() for pred_seq in new_preds]

new_preds = [tokenizer.decode([pred], skip_special_tokens=True).strip() for pred in new_preds]

new_preds = [int(pred) if pred.isdigit() else 0 for pred in new_preds]

new_data_df['predicted_label'] = new_preds
new_data_df.to_csv('new_predictions.csv', index=False)

print("Predictions for the new dataset have been saved to 'new_predictions.csv'.")
