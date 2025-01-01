# Linear and Logistic Regression from Scratch

This repository contains the complete implementation of **Linear Regression** and **Logistic Regression** from scratch in Python, using **Batch Gradient Descent** as the optimization algorithm. The models are demonstrated with a sample dataset, and the performance of each model is evaluated through cost functions and predictions.

## Overview

The primary goal of this repository is to demonstrate the basic principles of machine learning algorithms (Linear and Logistic Regression) using raw Python code, without relying on libraries like `scikit-learn`.

### Key Features

- **Linear Regression**: A simple model to predict a continuous value based on a set of input features.
- **Logistic Regression**: A model used for binary classification problems, which predicts probabilities and classifies data points into two categories.

## Formulae & Algorithms

### 1. **Linear Regression**

Linear regression models the relationship between the input features and a continuous target variable as a linear equation:

$$
y = Xw + b
$$

Where:
- `X` is the feature matrix (size: `m x n`),
- `w` is the weight vector (size: `n`),
- `b` is the bias term,
- `y` is the predicted output (size: `m`).

The **cost function** for linear regression is given by the Mean Squared Error (MSE) with L2 regularization:

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f(x^{(i)}) - y^{(i)} \right)^2 + \frac{lambda}{2m} \sum_{j=1}^{n} w_j^2
$$

Where:
- `f(x^(i)) = Xw + b` is the hypothesis,
- `lambda` is the regularization parameter.

#### Gradient Descent Update for Linear Regression:

$$
w := w - eta \cdot \frac{1}{m} \sum_{i=1}^{m} \left( f(x^{(i)}) - y^{(i)} \right) \cdot x^{(i)} + \frac{lambda}{m} w
$$
$$
b := b - eta \cdot \frac{1}{m} \sum_{i=1}^{m} \left( f(x^{(i)}) - y^{(i)} \right)
$$

Where:
- `eta` is the learning rate,
- `m` is the number of training examples.

### 2. **Logistic Regression**

Logistic regression is used for binary classification problems. It models the probability that a given input belongs to class 1 using the sigmoid function:

$$
f(x) = \frac{1}{1 + e^{-(Xw + b)}}
$$

Where:
- `f(x)` is the predicted probability.

The **cost function** for logistic regression is the **Log-Loss** with L2 regularization:

$$
J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f(x^{(i)})) + (1 - y^{(i)}) \log(1 - f(x^{(i)})) \right] + \frac{lambda}{2m} \sum_{j=1}^{n} w_j^2
$$

#### Gradient Descent Update for Logistic Regression:

$$
w := w - eta \cdot \frac{1}{m} \sum_{i=1}^{m} \left( f(x^{(i)}) - y^{(i)} \right) \cdot x^{(i)} + \frac{lambda}{m} w
$$
$$
b := b - eta \cdot \frac{1}{m} \sum_{i=1}^{m} \left( f(x^{(i)}) - y^{(i)} \right)
$$

Where:
- `eta` is the learning rate,
- `m` is the number of training examples.

Results & Evaluation

    Linear Regression: The predictions and cost are evaluated based on the Mean Squared Error.
    Logistic Regression: The predictions are evaluated using accuracy or classification metrics (precision, recall, F1-score).

## Contributing

Feel free to fork the repository, submit issues, and create pull requests to improve the model!

## License

This project is licensed under the MIT License.
