import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1) 
X_b = np.c_[np.ones((100, 1)), x]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
x_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), x_new]  
y_predict = X_new_b.dot(theta_best)
plt.plot(x_new, y_predict, "r-", label="Predictions")
plt.plot(x, y, "b.", label="Data points")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression")
plt.legend()
plt.show()
