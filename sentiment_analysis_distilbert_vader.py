import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import nltk

from transformers import DistilBERTTokenizer, DistilBertForSequenceClassification, AdamW
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Download stopwords if needed
nltk.download('stopwords')

# Load MELD dataset
df = pd.read_csv("MELD_dataset.csv")  # Change to actual path

# Encode emotion labels
label_encoder = LabelEncoder()
df['emotion'] = label_encoder.fit_transform(df['Emotion'])

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(df['Utterance'], df['emotion'], test_size=0.2, random_state=42)

# Load DistilBERT tokenizer
tokenizer = DistilBERTTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenization function
def tokenize_function(texts):
    return tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=64, return_tensors='pt')

train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

# Convert to torch tensors
train_labels = torch.tensor(train_labels.values)
val_labels = torch.tensor(val_labels.values)

# Apply VADER Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
def vader_score(text):
    sentiment = analyzer.polarity_scores(text)
    return [sentiment['pos'], sentiment['neu'], sentiment['neg']]

train_vader = np.array([vader_score(text) for text in train_texts])
val_vader = np.array([vader_score(text) for text in val_texts])

# Convert to tensors
train_vader = torch.tensor(train_vader, dtype=torch.float32)
val_vader = torch.tensor(val_vader, dtype=torch.float32)

# Hybrid Model (DistilBERT + VADER)
class HybridModel(nn.Module):
    def __init__(self, num_labels):
        super(HybridModel, self).__init__()
        self.distilbert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
        self.fc_vader = nn.Linear(3, 32)
        self.fc_final = nn.Linear(num_labels + 32, num_labels)

    def forward(self, input_ids, attention_mask, vader_features):
        bert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask).logits
        vader_output = F.relu(self.fc_vader(vader_features))
        combined = torch.cat((bert_output, vader_output), dim=1)
        return self.fc_final(combined)

num_classes = len(label_encoder.classes_)
model = HybridModel(num_classes)

# Training Parameters
batch_size = 16
epochs = 5
learning_rate = 2e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Custom Dataset Class
class EmotionDataset(Dataset):
    def __init__(self, encodings, vader, labels):
        self.encodings = encodings
        self.vader = vader
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'vader_features': self.vader[idx],
            'labels': self.labels[idx]
        }

# Create Dataloaders
train_dataset = EmotionDataset(train_encodings, train_vader, train_labels)
val_dataset = EmotionDataset(val_encodings, val_vader, val_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train(model, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            vader_features = batch['vader_features'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, vader_features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={train_acc:.4f}")

train(model, train_loader, val_loader, epochs)

# Evaluation
def evaluate(model, val_loader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            vader_features = batch['vader_features'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, vader_features)
            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print("Classification Report:\n", classification_report(true_labels, predictions))
    print("Accuracy:", accuracy_score(true_labels, predictions))

evaluate(model, val_loader)

# Confusion Matrix & Graphs
cm = confusion_matrix(true_labels, predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Accuracy & F1-Score Graph
metrics = {"Accuracy": accuracy_score(true_labels, predictions),
           "F1-Score": classification_report(true_labels, predictions, output_dict=True)['weighted avg']['f1-score']}
plt.bar(metrics.keys(), metrics.values(), color=['blue', 'red'])
plt.ylim(0, 1)
plt.title("Evaluation Metrics")
plt.show()
