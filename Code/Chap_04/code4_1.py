#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data, datasets


# In[18]:


def simple_tokenizer(text):
    return text.split()


# In[ ]:


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
class SimpleNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        hidden = self.relu(self.fc(pooled))
        output = self.out(hidden)
        return output


input_dim = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 1

model = SimpleNN(input_dim, embedding_dim, hidden_dim, output_dim)

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


torch.save(model.state_dict(), 'text_classification_model.pth')


loaded_model = SimpleNN(input_dim, embedding_dim, hidden_dim, output_dim)
loaded_model.load_state_dict(torch.load('text_classification_model.pth'))
loaded_model.eval()


# In[ ]:




