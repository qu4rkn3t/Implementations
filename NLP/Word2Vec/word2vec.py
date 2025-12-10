import re

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

CONFIG = {
    "embedding_dim": 10,
    "window_size": 2,
    "epochs": 100,
    "learning_rate": 0.001,
}


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return self.idx


class DataProcessor:
    def __init__(self, text):
        self.text = text

    def clean(self):
        self.text = self.text.lower()
        self.text = re.sub(r"[^a-z0-9\s]", "", self.text)

    def tokenize(self):
        self.text = self.text.split()


class Word2Vec(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.embeddings = nn.Embedding(size, dim)
        self.linear = nn.Linear(dim, size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)

        return out

    def get_embedding(self, word_idx):
        return self.embeddings(torch.tensor([word_idx]))


def create_training_data(tokens, window_size):
    data = []

    for i, center_word in enumerate(tokens):
        start = max(0, i - window_size)
        end = min(len(tokens) - 1, i + window_size)

        for j in range(start, end + 1):
            if i != j:
                context_word = tokens[j]
                data.append((center_word, context_word))

    return data


def train(vocab, training_data, embedding_dim, epochs, learning_rate):
    model = Word2Vec(len(vocab), embedding_dim)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for center, context in training_data:
            center_idx = torch.tensor([vocab.word2idx[center]], dtype=torch.long)
            context_idx = torch.tensor([vocab.word2idx[context]], dtype=torch.long)

            model.zero_grad()

            log_probs = model(center_idx)

            loss = loss_function(log_probs, context_idx)
            loss.backward()

            optimizer.step()

    return model


if __name__ == "__main__":
    corpus = """
    Machine learning is a field of inquiry devoted to understanding and building methods that 'learn', 
    that is, methods that leverage data to improve performance on some set of tasks. 
    It is seen as a part of artificial intelligence.
    """

    processor = DataProcessor(corpus)
    processor.clean()
    processor.tokenize()

    vocab = Vocabulary()
    for word in processor.text:
        vocab.add_word(word)

    training_data = create_training_data(processor.text, CONFIG["window_size"])

    model = train(
        vocab,
        training_data,
        CONFIG["embedding_dim"],
        CONFIG["epochs"],
        CONFIG["learning_rate"],
    )
