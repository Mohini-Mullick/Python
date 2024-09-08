from ucimlrepo import fetch_ucirepo

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

# metadata
print(iris.metadata)

# variable information
print(iris.variables)
X.shape
X.head()
X.describe()
y.value_counts()
import matplotlib.pyplot as plt

X.hist(figsize=(10, 10))
plt.show()
import seaborn as sns

sns.pairplot(X)
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.model_selection import KFold
# set up 10-fold cross validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}

# Train and evaluate models
for name, model in models.items():
  print(f"Training {name}...")
  model.fit(X_train, y_train.values.ravel())
  y_pred = model.predict(X_valid)
  accuracy = accuracy_score(y_valid, y_pred)
  print(f"Accuracy: {accuracy:.4f}\n")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Classification and Regression Trees": DecisionTreeClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC()
}

# Train and evaluate models
for name, model in models.items():
  print(f"Training {name}...")
  model.fit(X_train, y_train.values.ravel())
  y_pred = model.predict(X_valid)
  accuracy = accuracy_score(y_valid, y_pred)
  print(f"Accuracy: {accuracy:.4f}\n")
  import numpy as np
# Create new data for prediction
new_data = np.array([[5.1, 3.5, 1.4, 0.2],
                     [6.2, 2.9, 4.3, 1.3],
                     [7.3, 2.8, 6.3, 1.8]])

# Make predictions using the trained Random Forest model (you can choose any model)
model = RandomForestClassifier()
model.fit(X_train, y_train.values.ravel())
predictions = model.predict(new_data)

# Visualize predictions
plt.figure(figsize=(8, 6))
plt.scatter(X['sepal length'], X['sepal width'], cmap='viridis', label='Original Data')
plt.scatter(new_data[:, 0], new_data[:, 1], marker='x', s=100, cmap='viridis', label='Predictions')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Original Data vs. Predictions')
plt.legend()
plt.show()
