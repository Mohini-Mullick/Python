class LinearRegression:
  def __init__(self):
    self.coefficients = None

  def fit(self, X, y):
    X = np.c_[np.ones(X.shape[0]), X]  # Add a column of ones for the intercept
    self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

  def predict(self, X):
    X = np.c_[np.ones(X.shape[0]), X]
    return X @ self.coefficients

# Generate some sample data
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the results
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()