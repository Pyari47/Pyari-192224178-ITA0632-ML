import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class NaiveBayes:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}
        
    def fit(self, X_train, y_train):
        classes, counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        
        for class_, count in zip(classes, counts):
            self.class_probabilities[class_] = count / total_samples
            
            class_indices = np.where(y_train == class_)[0]
            class_samples = X_train[class_indices]
            feature_probabilities = {}
            
            for feature in range(X_train.shape[1]):
                unique_values, value_counts = np.unique(class_samples[:, feature], return_counts=True)
                feature_probabilities[feature] = {
                    "unique_values": unique_values,
                    "value_counts": value_counts / count
                }
                
            self.feature_probabilities[class_] = feature_probabilities
            
    def predict(self, X_test):
        predictions = []
        
        for sample in X_test:
            max_prob = -1
            predicted_class = None
            
            for class_ in self.class_probabilities:
                class_probability = self.class_probabilities[class_]
                feature_probabilities = self.feature_probabilities[class_]
                sample_prob = class_probability
                
                for feature, value in enumerate(sample):
                    if value in feature_probabilities[feature]["unique_values"]:
                        prob_index = np.where(feature_probabilities[feature]["unique_values"] == value)[0][0]
                        sample_prob *= feature_probabilities[feature]["value_counts"][prob_index]
                    else:
                        sample_prob *= 0  # Laplace smoothing for unseen values
                
                if sample_prob > max_prob:
                    max_prob = sample_prob
                    predicted_class = class_
                    
            predictions.append(predicted_class)
            
        return predictions

# Sample data
X = np.array([[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'],
              [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'],
              [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']])
y = np.array(['N', 'N', 'Y', 'Y', 'N', 'N', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes model
nb = NaiveBayes()
nb.fit(X_train, y_train)

# Make predictions
predictions = nb.predict(X_test)

# Display confusion matrix and accuracy
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nAccuracy:", accuracy_score(y_test, predictions))
