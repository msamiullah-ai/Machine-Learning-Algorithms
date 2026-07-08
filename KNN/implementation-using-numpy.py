import numpy as np
from collections import Counter

def euc_distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))

def knn_predict(X_train, X_test, y_train, k = 3):
    predictions = []
    for test_point in X_test:

        distances = [euc_distance(test_point, train_point) for train_point in X_train]

        k_indices = np.argsort(distances)[:k]

        k_labels = [y_train[i] for i in k_indices]

        most_common = Counter(k_labels).most_common(1)

        predictions.append(most_common[0][0])

    return predictions

# Simulated training data (Features: [Weight in grams, Sweetness scale 1-10])
X_train = [
    [15, 8],   # Grape 1
    [12, 7],   # Grape 2
    [18, 9],   # Grape 3
    [800, 6],  # Melon 1
    [950, 5],  # Melon 2
    [850, 7]   # Melon 3
]
# Labels: 0 = Grape, 1 = Melon
y_train = [1, 0, 0, 0, 0, 1]

# Simulated test data (Two mystery fruits we want to classify)

X_test = [
    [14, 8],   
    [900, 6]   
]

predictions = knn_predict(X_test, X_train, y_train, k=3)

print("Predictions:", predictions)

# Predictions: [1, 1, 1, 0, 0, 0]
