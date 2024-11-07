import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(gradient, initial_point, learning_rate, n_iterations):
    # Initialize the starting point
    point = initial_point

    # Create a list to store the progress
    points = [point]

    for _ in range(n_iterations):
        # Calculate the gradient at the current point
        grad = gradient(point)

        # Update the point using the gradient and learning rate
        point = point - learning_rate * grad

        # Save the new point to the progress list
        points.append(point)

    return point, points

# Example: Minimizing the function f(x) = x^2
# The gradient of f(x) = x^2 is f'(x) = 2x
def gradient(x):
    return 2 * x

# Parameters
initial_point = 10.0  # Starting point
learning_rate = 0.1   # Step size
n_iterations = 100    # Number of iterations

# Run the gradient descent
final_point, points = gradient_descent(gradient, initial_point, learning_rate, n_iterations)

# Output the final point
print(f"Final point after {n_iterations} iterations: {final_point}")

# Plotting the progress of the gradient descent
plt.figure(figsize=(10, 6))

# Plotting f(x) = x^2
x = np.linspace(-initial_point, initial_point, 400)
y = x**2
plt.plot(x, y, label="f(x) = x^2", color="blue")

# Plotting the points visited by gradient descent
points = np.array(points)
plt.scatter(points, points**2, color="red", label="Gradient Descent Steps")
plt.plot(points, points**2, color="red", linestyle="dashed", alpha=0.6)

# Highlighting the final point
plt.scatter(final_point, final_point**2, color="green", s=100, label=f"Final Point ({final_point:.2f}, {final_point**2:.2f})")

# Adding labels and legend
plt.title("Gradient Descent on f(x) = x^2")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
