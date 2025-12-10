import random
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

CONFIG = {
    "embedding_dim": 10,
    "window_size": 2,
    "epochs": 100,
    "learning_rate": 0.01,
    "neg_samples": 5,
    "subsample_t": 1e-5,
}


class Vocabulary:
    def __init__(self):
        self.idx = 0
        self.word2idx = {}
        self.idx2word = {}
        self.counts = Counter()

    def add_word(self, word):
        self.counts[word] += 1
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
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, dim)
        self.out_embed = nn.Embedding(vocab_size, dim)

        self.in_embed.weight.data.uniform_(-0.5 / dim, 0.5 / dim)
        self.out_embed.weight.data.uniform_(-0.5 / dim, 0.5 / dim)

    def forward(self, input_words, target_words):
        in_vectors = self.in_embed(input_words)
        out_vectors = self.out_embed(target_words)
        scores = torch.sum(in_vectors * out_vectors, dim=1)

        return scores

    def get_embedding(self, word_idx):
        return self.in_embed(torch.tensor([word_idx]))


def subsample_tokens(tokens, vocab, threshold=1e-5):
    kept_tokens = []
    total_count = sum(vocab.counts.values())

    for word in tokens:
        if word not in vocab.word2idx:
            continue

        freq = vocab.counts[word] / total_count
        p_keep = (np.sqrt(freq / threshold) + 1) * (threshold / freq)

        if random.random() < p_keep:
            kept_tokens.append(word)

    return kept_tokens


def get_noise_distribution(vocab):
    probs = []
    for idx in range(len(vocab)):
        word = vocab.idx2word[idx]
        count = vocab.counts[word]
        probs.append(count**0.75)

    probs = np.array(probs)
    probs = probs / np.sum(probs)
    return torch.tensor(probs, dtype=torch.float)


def create_training_data(tokens, window_size):
    data = []
    for i, center_word in enumerate(tokens):
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)

        for j in range(start, end):
            if i != j:
                context_word = tokens[j]
                data.append((center_word, context_word))
    return data


def train(vocab, training_data, embedding_dim, epochs, learning_rate, k_neg_samples):
    model = Word2Vec(len(vocab), embedding_dim)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    noise_dist = get_noise_distribution(vocab)

    for epoch in range(epochs):
        total_loss = 0

        for center, context in training_data:
            center_idx = torch.tensor([vocab.word2idx[center]], dtype=torch.long)
            pos_context_idx = torch.tensor([vocab.word2idx[context]], dtype=torch.long)

            neg_context_idxs = torch.multinomial(
                noise_dist, k_neg_samples, replacement=True
            )

            optimizer.zero_grad()

            pos_score = model(center_idx, pos_context_idx)
            pos_loss = loss_function(pos_score, torch.tensor([1.0]))

            neg_scores = model(center_idx.repeat(k_neg_samples), neg_context_idxs)
            neg_loss = loss_function(neg_scores, torch.zeros(k_neg_samples))

            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

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

    subsampled_tokens = subsample_tokens(processor.text, vocab, CONFIG["subsample_t"])
    training_data = create_training_data(subsampled_tokens, CONFIG["window_size"])

    model = train(
        vocab,
        training_data,
        CONFIG["embedding_dim"],
        CONFIG["epochs"],
        CONFIG["learning_rate"],
        CONFIG["neg_samples"],
    )
