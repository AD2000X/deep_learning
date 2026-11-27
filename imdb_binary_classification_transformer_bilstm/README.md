# IMDB Sentiment Classification: From Bigrams to Transformers

How much word order does a model need to classify a movie review as positive or negative? This project compares three ways to read the same IMDB reviews: looking for words and short phrases, reading words in sequence with a BiLSTM, and learning relationships between words with a Transformer encoder.

The strongest recorded test result came from a simple model: **a dense classifier using word and word-pair features reached 89.64% accuracy**. The Transformer with positional embeddings reached **88.39%**.

[Open the notebook to see the experiments, learning curves, and saved outputs.](./imdb_binary_classification_transformer_bilstm.ipynb)

## Data and approach

The project uses the [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). The models were developed on 20,000 training reviews and 5,000 validation reviews. The official test set contains 25,000 reviews. Unlabelled reviews were excluded.

The experiments were implemented in TensorFlow/Keras:

| Approach | What information does it use? |
|---|---|
| Word and phrase features | The unigram model records which words appear. The bigram model also records adjacent word pairs, so `not good` can be a feature in its own right. A third version weights these features using TF-IDF. |
| Bidirectional LSTM | Reads the review in both directions to use the words before and after each position. A separate experiment uses frozen, pre-trained GloVe word embeddings. |
| Transformer encoder | Uses self-attention to connect words across the review. The notebook compares encoders with and without positional embeddings and examines how model size and training settings affect overfitting. |

## Results

| Model | Test accuracy |
|---|---:|
| Transformer without positional embeddings | 87.09% |
| Transformer with positional embeddings | 88.39% |
| Binary unigram | 88.66% |
| **Binary unigram + bigram** | **89.64%** |
| TF-IDF unigram + bigram | 89.26% |

The table reports saved **test** results only. The BiLSTM experiments are discussed in the notebook using their training and validation results; those scores are not mixed into the test comparison.

## What I learned

Adding adjacent word pairs improved accuracy by **0.98 percentage points** over single words alone. Giving those features TF-IDF weights did not improve on simply recording whether they appeared. For these reviews, short phrases provided useful sentiment information without requiring a sequence model.

Position information also mattered in the Transformer experiments: the encoder with positional embeddings scored **1.30 percentage points higher** than the version without them. However, it still did not outperform the simpler bigram classifier in this run.

## Limitations

Several models were evaluated on the same test set during the project. These results describe the recorded experiments, not a fresh evaluation of a model selected without seeing test scores. They do not establish that bigram models are generally better than Transformers.
