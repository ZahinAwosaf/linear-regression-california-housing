# Linear Regression from Scratch using NumPy

## Problem Statement

The goal of this project was to implement linear regression from scratch using NumPy without relying on `sklearn`'s regression algorithm. This included implementing the hypothesis function, Mean Squared Error (MSE) cost function, gradient computation, and batch gradient descent. The manual implementation was then validated against scikit-learn's `LinearRegression` model.

## Dataset

The project uses the **California Housing** dataset, which contains information about California districts such as median income, house age, average number of rooms, population, and geographic location. The target variable is the median house value.

This dataset was chosen because it is a standard regression benchmark with multiple numerical features, making it well suited for evaluating a linear regression model.

## Approach

1. Performed exploratory data analysis (EDA).
2. Split the dataset into training and testing sets.
3. Standardized the features using StandardScaler.
4. Implemented linear regression from scratch using:
   * Vectorized hypothesis function
   * Mean Squared Error cost function
   * Analytical gradient computation
   * Batch Gradient Descent
5. Trained a scikit-learn LinearRegression model using the same training and testing data.
6. Compared coefficients, predictions, and evaluation metrics to validate the NumPy implementation.

## Results



## Validation Against scikit-learn

### Coefficient Comparison

The learned intercepts were almost identical:
* **NumPy**: 2.071857
* **Scikit-learn**: 2.071947

Most feature coefficients were also reasonably close. Small differences are expected because the NumPy implementation uses iterative gradient descent, whereas scikit-learn computes the exact least-squares solution. The largest differences occurred in the final two coefficients, suggesting that additional iterations or a different learning rate could move the gradient descent solution closer to the optimum.

Prediction performance was also similar:

* **NumPy MSE**: 0.5672
* **Scikit-learn MSE**: 0.5559
* **Difference**: 0.0113

### Gradient Descent Convergence

The cost decreases rapidly during the first iterations before gradually leveling off, indicating that gradient descent converged successfully.

!(/assets/gradient-descent-convergence.png)
