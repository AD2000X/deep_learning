# IMDB Movie Review Sentiment Classification

How much does tuning improve a small neural network for sentiment classification? On 25,000 IMDB test reviews, the selected model reached **89.08% accuracy**, compared with **88.55%** for a four-unit baseline. It classified 134 more reviews correctly, but its precision for positive reviews was slightly lower.

[Open the notebook to see the experiments, learning curves, and confusion matrices.](./imdb_movie_review_sentiment_classification_pytorch.ipynb) The saved outputs are included, so you can read the results without running the notebook.

## Data and method

The study uses the [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) in its pre-tokenised IMDB format. Of the 25,000 official training reviews, 15,000 were used for training and 10,000 for validation during model development. Each review was represented by a multi-hot vector with a vocabulary limit of 10,000: the model knows which words appear, but not their order or frequency.

Starting from one hidden layer with four units, the experiments compared network size, learning rate, regularisation, Dropout, early stopping, loss function, and optimiser. A local grid search then explored combinations around the settings selected in the earlier stages.

Each experiment kept the epoch with the highest validation accuracy. When two configurations had the same accuracy, the comparison used binary cross-entropy (BCE) from their selected checkpoints, even if one model had trained with mean squared error. Screening used one random seed; the two finalists were compared across three seeds. The final settings were fixed before retraining on all 25,000 training reviews and evaluating on the test set.

## Results

| Test metric | Baseline | Selected model |
|---|---:|---:|
| Accuracy | 88.55% | 89.08% |
| Precision (positive class) | 89.96% | 89.50% |
| Recall (positive class) | 86.78% | 88.56% |
| F1 (positive class) | 0.8834 | 0.8903 |
| ROC AUC | 0.9518 | 0.9554 |

The selected model used one hidden layer with **16 ReLU units**, Dropout of **0.5**, an L2 penalty of **0.0001**, and RMSprop with a learning rate of **0.0002**. It trained for **15 epochs** on the full training set.

Two findings were more informative than the final accuracy alone:

- **Learning rate changed when to stop.** With 16 hidden units, learning rates of 0.001 and 0.0002 reached similar peak validation accuracy: 89.23% and 89.27%. The higher rate peaked at epoch 3, after which validation BCE rose sharply. The lower rate peaked at epoch 12 and was less sensitive to the exact stopping point.
- **The gain came mainly from finding more positive reviews.** Compared with the baseline, the selected model missed 223 fewer positive reviews but incorrectly labelled 89 more negative reviews as positive. This explains why recall improved while precision fell.

## Limitations

The two finalists differed by only **0.03 percentage points** in mean validation accuracy across three seeds. That is not enough evidence to say that one configuration is consistently better. The search also reused one validation split, and the test set had been used in an earlier version of the project. The test scores describe this comparison, not a fresh independent evaluation.

## Run in Google Colab

1. Open the notebook in Colab and select a GPU runtime if available.
2. For a quick check, set `RUN_MODE = "smoke"` in section 4.1 and choose **Run all**. This mode uses smaller subsets and skips the final test.
3. For the full experiment, use `RUN_MODE = "full"`. The default `RUN_FINAL_TEST = True` evaluates the selected model and baseline after their settings are fixed. Set it to `False` for a validation-only run.
4. Allow Google Drive access when prompted. Results and checkpoints are stored under `MyDrive/IMDB_Classification_Coursework/`. After a disconnection, use the same run tag and settings to reuse verified results.

**Tools:** PyTorch for modelling, Lightning for training and checkpoints, TorchMetrics for accuracy, and scikit-learn for data splitting and final evaluation.
