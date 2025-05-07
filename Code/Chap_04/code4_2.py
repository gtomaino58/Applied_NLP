#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data, datasets

def simple_tokenizer(text):
    return text.split()


TEXT = data.Field(tokenize=simple_tokenizer)
LABEL = data.LabelField(dtype=torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)


TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)


BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True
)


class SentimentAnalysisNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(SentimentAnalysisNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        avg_pool = torch.mean(output, dim=0)
        predictions = self.fc(avg_pool)
        return predictions

input_dim = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 1

model = SentimentAnalysisNN(input_dim, embedding_dim, hidden_dim, output_dim)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())


epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch in train_iterator:
        text, labels = batch.text, batch.label
        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_iterator)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}')


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_iterator:
        text, labels = batch.text, batch.label
        predictions = model(text).squeeze(1)
        predicted_labels = torch.round(torch.sigmoid(predictions))
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

accuracy = correct / totalprint(f'Test Accuracy: {accuracy * 100:.2f}%')
print(accuracy)

torch.save(model.state_dict(), 'sentiment_analysis_model.pth')


loaded_model = SentimentAnalysisNN(input_dim, embedding_dim, hidden_dim, output_dim)
loaded_model.load_state_dict(torch.load('sentiment_analysis_model.pth'))
print(loaded_model.eval())


# In[ ]:





# In[ ]:





# In[ ]:




