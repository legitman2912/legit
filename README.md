# Machine Learning Viva – Answers

## 1. Define Machine Learning.
Machine learning (ML) is a part of artificial intelligence where a computer learns from data instead of being told every rule. The algorithm looks at past data, finds patterns, and uses them to make predictions or decisions. The more good data it sees, the better it gets.

## 2. What are the different types of Machine Learning?
- **Supervised learning:** The model learns from *labeled* data (inputs with the correct answers). Example: predicting house prices, spam detection.
- **Unsupervised learning:** The model gets *unlabeled* data and finds hidden patterns or groups by itself. Example: customer grouping (clustering).
- **Reinforcement learning:** An agent learns by trial and error. It gets rewards for good actions and penalties for bad ones. Example: game-playing AI, robots.

## 3. What is a loss or cost function?
A loss function measures how far the model's prediction is from the actual value. A big loss means bad predictions, a small loss means good predictions.
- **Mean Squared Error (MSE):** used for regression.
- **Cross-Entropy Loss:** used for classification.

Training tries to make the loss as small as possible, usually using gradient descent.

## 4. What is MLE?
**Maximum Likelihood Estimation** is a method to find the parameter values (θ) that make the observed data most probable.
In simple words: *"Which parameters would most likely produce the data I actually saw?"*

## 5. What are activation functions?
Activation functions are added to neurons in a neural network to bring in **non-linearity**. Without them, the network would just be a linear model and could not learn complex patterns.
Common ones:
- **ReLU:** outputs 0 for negative values and the value itself for positive values.
- **Sigmoid:** squashes values between 0 and 1.
- **Tanh:** squashes values between -1 and 1.

## 6. What is overfitting in machine learning and how can it be avoided?
Overfitting happens when a model learns the training data too well, including its noise. It gives great results on training data but poor results on new (test) data.

**How to avoid it:**
- Cross-validation
- Regularization (L1/L2)
- Pruning (for decision trees)
- Using more training data
- Using a simpler model

## 7. What is Regularization?
Regularization adds a **penalty term** to the loss function so the model doesn't become too complex. It keeps the model weights small, which reduces overfitting (lower variance) without increasing bias much.
- **L1 (Lasso):** can shrink some weights to exactly zero (feature selection).
- **L2 (Ridge):** shrinks weights close to zero but not exactly zero.

## 8. Explain Confusion Matrix.
A confusion matrix is a table that shows how well a classification model performed.

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

From it we calculate:
- **Accuracy** = (TP + TN) / Total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-score** = harmonic mean of precision and recall

## 9. What is the difference between classification and regression?
| Classification | Regression |
|---|---|
| Predicts a **category** (discrete) | Predicts a **number** (continuous) |
| Example: spam / not spam, yes / no | Example: stock price, rainfall |
| Algorithms: Logistic Regression, Decision Trees | Algorithms: Linear Regression |

## 10. What is feature scaling?
Feature scaling brings all features to a similar range so that no feature dominates just because it has bigger numbers (for example, salary vs. age).
- **Min-Max normalization:** scales values to a range like 0 to 1.
- **Z-score standardization:** makes the mean 0 and standard deviation 1.

## 11. What is Dimensionality Reduction in Machine Learning?
It means reducing the number of features (columns/variables) in a dataset while keeping most of the important information. It makes models faster, reduces overfitting, and helps with visualization.
Example techniques: **PCA** (Principal Component Analysis), LDA, t-SNE.

## 12. What is data preprocessing?
Data preprocessing means cleaning and preparing raw data so a model can learn from it properly. It includes:
- Handling missing values
- Removing outliers
- Normalizing / scaling data
- Encoding categorical variables (e.g., one-hot encoding)

Good preprocessing gives better quality data and a better model.

## 13. What is Bias-Variance tradeoff?
- **Bias:** error from a model that is too simple. High bias leads to **underfitting**.
- **Variance:** error from a model that is too sensitive to small changes in training data. High variance leads to **overfitting**.

Making a model more complex lowers bias but raises variance. The goal is to find a good balance between the two so the model works well on new data.

## 14. What is Gradient Descent?
Gradient descent is an optimization method that reduces the cost function step by step. It updates the model parameters in the direction of the steepest decrease of the loss, until it reaches a minimum.

Variants:
- **Batch:** uses the whole dataset for each step.
- **Stochastic (SGD):** uses one sample at a time.
- **Mini-batch:** uses small groups of samples (fast and stable).

## 15. How to choose an optimal number of clusters?
- **Elbow Method:** Plot the within-cluster sum of squares (WCSS) against the number of clusters (K). The point where the curve starts to flatten (the "elbow") is a good K.
- **Silhouette Score:** Measures how close each point is to its own cluster compared to other clusters. A higher average score means better clusters, so choose the K with the highest score.

## 16. Explain the working of a decision tree.
A decision tree splits the data again and again based on feature values, choosing the split that gives the highest **information gain**.
- **Internal nodes:** features being tested
- **Branches:** the conditions / outcomes of the test
- **Leaf nodes:** final prediction

It is easy to understand, but it can overfit, so pruning is used.

## 17. What are the types of Naive Bayes algorithm?
- **Gaussian Naive Bayes:** for continuous features that follow a normal distribution (height, weight, temperature).
- **Multinomial Naive Bayes:** for count data, like word counts in text classification (spam detection).
- **Bernoulli Naive Bayes:** for binary features (0/1), like whether a word is present or not.
- **Categorical Naive Bayes:** for categorical features (e.g., color: red, green, blue).

## 18. What is the decision boundary in SVM?
In SVM, the decision boundary is the line (in 2D) or hyperplane (in higher dimensions) that separates the classes. It is chosen so that the **margin** is as large as possible. The margin is the distance between the hyperplane and the nearest data points, which are called **support vectors**.

## 19. Explain Bagging and Boosting.
| Bagging | Boosting |
|---|---|
| Full name: Bootstrap Aggregating | Models are built one after another |
| Trains many models **in parallel** on random subsets of data | Each new model focuses on the **mistakes** of the previous one |
| Reduces **variance** | Reduces **bias** and improves accuracy |
| Example: Random Forest | Example: AdaBoost, XGBoost |

## 20. Explain K-Means Clustering.
K-Means groups data into K clusters.
1. Choose the number of clusters **K**.
2. Randomly pick **K centroids** (cluster centers).
3. Assign each data point to the **nearest centroid**.
4. Recalculate each centroid as the **mean** of the points in its cluster.
5. Repeat steps 3–4 until the centroids stop changing (convergence).

## 21. Important Python libraries
- **NumPy:** works with large arrays and matrices and does fast math operations.
- **Pandas:** gives DataFrame and Series for cleaning and handling structured data.
- **Matplotlib:** creates graphs and plots to understand data and model performance.
- **Scikit-learn:** ready-made tools for classification, regression, clustering, preprocessing and model evaluation.
- **TensorFlow:** open-source deep learning framework from Google for building and training neural networks.
- **Keras:** easy, high-level API for building neural networks quickly (runs on top of TensorFlow).
