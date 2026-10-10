# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on algorithms 
that can learn from and make predictions based on data.

## Supervised Learning

Supervised learning uses labeled data to train models. Common algorithms include:

- **Linear Regression**: Predicts continuous values
- **Decision Trees**: Makes decisions based on feature values
- **Neural Networks**: Inspired by biological neural networks

### Linear Regression Example

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data. Key techniques include:

1. **Clustering**: Groups similar data points together
2. **Dimensionality Reduction**: Reduces the number of features
3. **Anomaly Detection**: Identifies outliers in data

### K-Means Clustering

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
labels = kmeans.predict(data)
```

## Reinforcement Learning

Reinforcement learning involves agents learning through interaction with an environment.

> "The goal of reinforcement learning is to maximize cumulative reward." - Sutton & Barto

## Conclusion

Machine learning continues to evolve, enabling new applications in various domains.