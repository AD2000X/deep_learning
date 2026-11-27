# IMDB Movie Review Sentiment Classification

This repository contains three studies on the same task: predicting whether an IMDB movie review is positive or negative. The first builds a simple baseline, the second explores what changes when word order is available, and the third asks how much careful tuning can improve the baseline.

## The three studies

| Study | Main question | What to look for |
|---|---|---|
| [1. A small dense network in Keras](./imdb_binary_classification_sequential/) | How well can a model classify reviews using only the words that appear? | Training and validation curves show how network size, learning rate, and regularisation affect overfitting. |
| [2. Word pairs, BiLSTM, and Transformer](./imdb_binary_classification_transformer_bilstm/) | Does the model need the whole sequence, or are short phrases enough? | The binary unigram + bigram classifier reached **89.64% test accuracy**, compared with **88.39%** for the Transformer with positional embeddings in this study. |
| [3. Validation-based tuning in PyTorch](./imdb_movie_review_sentiment_classification_pytorch/) | How much does tuning improve a small network, and how stable is the choice across seeds? | The two finalists were checked across three seeds. On the test set, the selected model reached **89.08% accuracy**, compared with **88.55%** for its baseline. |

## What the results show

In the second study, adding adjacent word pairs improved accuracy from **88.66% to 89.64%**. A phrase such as `not good` can be a feature in its own right, even when the model does not read the whole sentence in order.

In the third study, tuning changed the kinds of mistakes the model made. On 25,000 test reviews, the selected model missed **223 fewer positive reviews** but made **89 more false positive predictions**. That is **134 more reviews classified correctly** overall.

The studies use different training and validation splits and different model-selection procedures. Their scores are useful for comparisons **within each study**, not as a ranking of TensorFlow versus PyTorch.
