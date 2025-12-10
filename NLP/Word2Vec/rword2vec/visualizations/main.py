import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

RUST_OUTPUT_FILE = "rust_embeddings.json"
TOP_N_FOR_VIZ = 50


def visualize_embeddings(embeddings_dict):
    words = list(embeddings_dict.keys())
    vectors = np.array(list(embeddings_dict.values()))

    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        vectors_2d[:, 0], vectors_2d[:, 1], c="blue", edgecolors="k", s=100, alpha=0.7
    )

    for i, word in enumerate(words):
        plt.annotate(
            word,
            xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
        )

    plt.title("Word2Vec Embeddings Visualization (PCA projection)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def load_and_filter_rust_json(filepath, top_n):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        print(e)
        return None

    word_vectors = data["word_vectors"]
    word_counts = data["word_counts"]

    word_counts_counter = Counter(word_counts)
    most_common_words = [word for word, count in word_counts_counter.most_common(top_n)]

    filtered_embeddings_dict = {}
    for word in most_common_words:
        filtered_embeddings_dict[word] = np.array(word_vectors[word], dtype=np.float32)

    return filtered_embeddings_dict


embeddings_to_plot = load_and_filter_rust_json(RUST_OUTPUT_FILE, TOP_N_FOR_VIZ)
if embeddings_to_plot:
    visualize_embeddings(embeddings_to_plot)
