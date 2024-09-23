import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step a: Load the dataset
iris_dataset_path = 'iris.csv'
iris_data = pd.read_csv(iris_dataset_path)

# Step b: Split the dataset into features (X) and target (y)
X = iris_data.drop(columns=['Species'])  # Features
y = iris_data['Species']  # Target

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step c: Train a Decision Tree model on the training set
clf = DecisionTreeClassifier(criterion='entropy')  # Using entropy for ID3 algorithm
clf.fit(X_train, y_train)

# Step d: Test the model on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print accuracy and confusion matrix
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Iris Classification')
plt.show()
