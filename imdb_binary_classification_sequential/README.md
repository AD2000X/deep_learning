# IMDB Sentiment Classification with a Small Dense Network

How well can a model classify a movie review if it knows which words appear, but not their order? This notebook starts with a four-unit neural network and explores how model size, learning rate, and regularisation affect training and overfitting.

[Open the notebook to see the experiments, learning curves, and saved results.](./imdb_binary_classification_sequential_model.ipynb)

## Data and model

The study uses 15,000 IMDB reviews for training, 10,000 for validation, and 25,000 for testing. Each review is converted into a 10,000-dimensional multi-hot vector: a feature is 1 if its word appears in the review and 0 otherwise. Repeating a word or moving it to a different position does not change the vector.

The starting model has one hidden layer with four ReLU units and a sigmoid output for positive or negative sentiment. It is built with the Keras `Sequential` API. Here, *Sequential* describes how the network layers are stacked; it does not mean the model reads the review word by word.

## Experiments and results

The notebook compares wider and deeper networks, different learning rates, L1 and L2 penalties, Dropout, and early stopping. It also tests RMSprop against Adam and binary cross-entropy against mean squared error.

One useful example is the network with two four-unit hidden layers, using ReLU followed by tanh. Its **peak validation accuracy was 89.28%**, while the model evaluated at the end of that run reached **86.32% test accuracy**. These numbers come from different datasets and refer to different points in training. The learning curve helps explain what a single peak score leaves out.

The final search fixed the batch size at 256 and the learning rate at 0.0005. Of the two L2 penalties tested, 0.0001 gave the higher test accuracy (**88.31%**). Because the test set was used to choose between the settings, this score is exploratory rather than an independent estimate of final performance.

## Where this fits in the project

- [Text representations, Transformer, and BiLSTM](../imdb_binary_classification_transformer_bilstm/) asks what changes when the model can use word pairs or word order.
- [Validation-based tuning with PyTorch](../imdb_movie_review_sentiment_classification_pytorch/) returns to the small dense network, selects settings using validation data, and checks the finalists across three random seeds before final testing.
