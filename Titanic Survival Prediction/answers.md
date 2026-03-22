1. Perform kNN

   a. Why does accuracy drop when k becomes too large?
   - When the K becomes too large, the model will underfit, meaning the model is too simple to capture local patterns in the data. When an unlabeled data is inserted, it will consider so many neighbors that the influence of nearby, relevant points is diluted by distant points, casuing the model to predict more general often incorrect, resulting in lower accuracy of the classification model.

   b. Why is scaling required for kNN?
   - Scaling is required for KNN to normalize and equalize the importance of each feature in a dataset, which prevents features with varying ranges from dominating model training.

   c. What is the trade-off between small and large k?
   - Having a small k parameter can result in overfitting the model, where the model does not generalize well with unseen data. Having a large k parameter can result in underfitting, where the model is too simple to handle complex relationships between variables, there will be high bias and will the presence of nearby neighbors will be diluted or taken over by the distant neighbors.

2. Perform CART (Decision Tree Classification)

   a. Which tree is easier to interpret?
   - Decision Tree

   b. Which model performs better on test data?
   - Knn

   c. Which tree is more likely to overfit?
   - idk
