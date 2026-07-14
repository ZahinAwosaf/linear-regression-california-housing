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

| Model | R² | MAE | RMSE | MSE |
|---|---:|---:|---:|---:|
| NumPy Gradient Descent | 0.5672 | 0.5477 | 0.7531 | 0.5672 |
| Scikit-learn LinearRegression | 0.5758 | 0.5332 | 0.7456 | 0.5559 |

The NumPy implementation achieved results close to scikit-learn's LinearRegression model. The small difference is expected because gradient descent is an iterative optimization method, while scikit-learn computes the least-squares solution directly.

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

## Gradient Descent Convergence

The cost decreases rapidly during the first iterations before gradually leveling off, indicating that gradient descent converged successfully.

![Gradient Descent Convergence](/assets/gradient-descent-convergence.png)

## Learning Rate Comparison

A learning rate of 0.001 converged slowly, while 0.01 and 0.1 converged much faster. For this dataset, 0.1 reached the minimum cost in the fewest iterations without diverging.

![Learning Rate Comparison](/assets/learning-rate-comparison.png)

## Residual Analysis

The residuals are centered around zero, indicating little overall prediction bias. However, they are not completely random and show a fan-shaped pattern, suggesting some heteroscedasticity. A few outliers are also present, indicating larger prediction errors for some observations.

![Residual Analysis](/assets/residual-plot.png)

## Metric Interpretation

* R² measures how much of the variation in house values is explained by the model. Higher values indicate a better fit.
* MAE is the average absolute prediction error. Lower values indicate more accurate predictions.
* RMSE measures the average prediction error while giving greater weight to larger errors. Lower values indicate better performance.

## Future Improvements

If this project were extended, I would:

* Tune the learning rate and number of gradient descent iterations to further reduce the gap with scikit-learn.
* Implement mini-batch and stochastic gradient descent for faster optimization.
* Add regularization (Ridge and Lasso) to reduce overfitting.
* Explore polynomial features or more advanced regression models to better capture non-linear relationships in the California Housing dataset.

