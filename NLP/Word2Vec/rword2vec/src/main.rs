use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::{collections::HashMap, fs};

use ndarray::{Array1, Array2};
use rand::Rng;
use regex::Regex;
use serde::Serialize;

const EMBEDDING_DIM: usize = 20;
const WINDOW_SIZE: usize = 5;
const EPOCHS: usize = 20;
const LEARNING_RATE: f32 = 0.025;
const NEG_SAMPLES: usize = 5;
const SUBSAMPLE_T: f32 = 1e-4;
const CORPUS_FILE: &str = "corpus.txt";
const OUTPUT_FILE: &str = "rust_embeddings.json";

#[derive(Serialize)]
struct ModelOutput {
    word_vectors: HashMap<String, Vec<f32>>,
    word_counts: HashMap<String, usize>,
}

struct Vocabulary {
    word2idx: HashMap<String, usize>,
    idx2word: Vec<String>,
    counts: HashMap<String, usize>,
}

impl Vocabulary {
    fn new() -> Self {
        Self {
            word2idx: HashMap::new(),
            idx2word: Vec::new(),
            counts: HashMap::new(),
        }
    }

    fn add_token(&mut self, token: String) {
        *self.counts.entry(token.clone()).or_insert(0) += 1;
        if !self.word2idx.contains_key(&token) {
            self.word2idx.insert(token.clone(), self.idx2word.len());
            self.idx2word.push(token);
        }
    }
}

fn process(text: &str) -> Vec<String> {
    let re = Regex::new(r"[^a-zA-Z0-9\s]").unwrap();
    let lower = text.to_ascii_lowercase();
    let cleaned = re.replace_all(&lower, "");

    cleaned.split_whitespace().map(|s| s.to_string()).collect()
}

fn subsample(tokens: Vec<String>, vocab: &Vocabulary) -> Vec<String> {
    let mut kept_tokens = Vec::new();
    let total_words = vocab.counts.values().sum::<usize>() as f32;
    let mut rng = rand::rng();

    for token in tokens {
        let count = *vocab.counts.get(&token).unwrap();
        let freq = count as f32 / total_words;
        let keep_prob = ((freq / SUBSAMPLE_T).sqrt() + 1.0) * (SUBSAMPLE_T / freq);

        if keep_prob >= rng.random() {
            kept_tokens.push(token);
        }
    }

    kept_tokens
}

fn get_noise_distribution(vocab: &Vocabulary) -> Vec<f32> {
    let mut probs = Vec::with_capacity(vocab.idx2word.len());
    let mut sum = 0.0;

    for word in &vocab.idx2word {
        let count = *vocab.counts.get(word).unwrap() as f32;
        let p = count.powf(0.75);
        probs.push(p);
        sum += p;
    }

    probs.iter().map(|&x| x / sum).collect()
}

fn sample_negative(cumulative_dist: &[f32], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.random();

    for (i, &prob) in cumulative_dist.iter().enumerate() {
        if r < prob {
            return i;
        }
    }

    cumulative_dist.len() - 1
}

fn train(vocab: &Vocabulary, tokens: &[String]) -> Array2<f32> {
    let vocab_size = vocab.idx2word.len();
    let mut rng = rand::rng();

    let mut w_in = Array2::from_shape_fn((vocab_size, EMBEDDING_DIM), |_| {
        (rng.random::<f32>() - 0.5) / EMBEDDING_DIM as f32
    });
    let mut w_out = Array2::zeros((vocab_size, EMBEDDING_DIM));

    let noise_probs = get_noise_distribution(vocab);
    let mut cumulative_noise = Vec::with_capacity(vocab_size);
    let mut acc = 0.0;
    for p in noise_probs {
        acc += p;
        cumulative_noise.push(acc);
    }

    for _ in 0..EPOCHS {
        for (i, token) in tokens.iter().enumerate() {
            let center_idx = *vocab.word2idx.get(token).unwrap();

            let start = i.saturating_sub(WINDOW_SIZE);
            let end = (i + WINDOW_SIZE + 1).min(tokens.len());

            for j in start..end {
                if i == j {
                    continue;
                }

                let context_token = &tokens[j];
                let context_idx = *vocab.word2idx.get(context_token).unwrap();

                let mut target_indices = Vec::with_capacity(NEG_SAMPLES + 1);
                let mut labels = Vec::with_capacity(NEG_SAMPLES + 1);

                target_indices.push(context_idx);
                labels.push(1.0);

                for _ in 0..NEG_SAMPLES {
                    let noise_idx = sample_negative(&cumulative_noise, &mut rng);
                    if noise_idx != context_idx {
                        target_indices.push(noise_idx);
                        labels.push(0.0);
                    }
                }

                let v_center = w_in.row(center_idx).to_owned();
                let mut grad_center = Array1::<f32>::zeros(EMBEDDING_DIM);

                for (k, &target_idx) in target_indices.iter().enumerate() {
                    let label = labels[k];
                    let v_target = w_out.row(target_idx);

                    let score = v_center.dot(&v_target);
                    let sigma = 1.0 / (1.0 + (-score).exp());
                    let g = (sigma - label) * LEARNING_RATE;

                    let grad_context = &v_center * g;
                    grad_center = grad_center + (&v_target * g);

                    let mut target_row = w_out.row_mut(target_idx);
                    target_row -= &grad_context;
                }

                let mut center_row = w_in.row_mut(center_idx);
                center_row -= &grad_center;
            }
        }
    }

    w_in
}

fn main() {
    let raw_text = if Path::new(CORPUS_FILE).exists() {
        fs::read_to_string(CORPUS_FILE).expect("Failed to read file.")
    } else {
        "king queen prince princess man woman boy girl
        paris london rome berlin capital city
        apple orange banana grape fruit food
        computer code data ai tech keyboard"
            .repeat(50)
    };

    let tokens = process(&raw_text);

    let mut vocab = Vocabulary::new();
    for t in &tokens {
        vocab.add_token(t.clone());
    }

    let training_tokens = subsample(tokens, &vocab);
    let embeddings_matrix = train(&vocab, &training_tokens);

    let mut output_map = HashMap::new();
    for (i, word) in vocab.idx2word.iter().enumerate() {
        let vec = embeddings_matrix.row(i).to_vec();
        output_map.insert(word.clone(), vec);
    }

    let output_data = ModelOutput {
        word_vectors: output_map,
        word_counts: vocab.counts,
    };

    let json = serde_json::to_string(&output_data).unwrap();
    let mut file = File::create(OUTPUT_FILE).unwrap();
    file.write_all(json.as_bytes()).unwrap();
}
