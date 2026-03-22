1. Perform kNN
   a. Why does accuracy drop when k becomes too large?
   - When the K becomes too large, the model will underfit, meaning the model is too simple to learn any patterns in the data because the model's decision boundary is very large where it simply predicts the majority class of the dataset resulting to pattern loss.

   b. Why is scaling required for kNN?
   - Scaling is required for KNN to normalize and equalize the importance of each feature in a dataset, which prevents features with varying ranges from dominating model training.

   c. What is the trade-off between small and large k?
   - Having a small k parameter can result in overfitting the model, where the model does generalize well with unseen data. Having a large k parameter can result in underfitting, where the model is too simple to handle complex relationships between variables

2. Perform CART (Decision Tree Classification)
   a. Which tree is easier to interpret?
   b. Which model performs better on test data?
   c. Which tree is more likely to overfit?
