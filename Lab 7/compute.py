import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def gaussian(x, mu, sigma):
  return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )

x = np.linspace(-5, 5, 100)

# Varying mean
plt.figure(figsize=(8, 6))
for mu in [-2, 0, 2]:
  plt.plot(x, gaussian(x, mu, 1), label=f'Mean = {mu}')
plt.title('Gaussian Distribution with Varying Mean')
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# Varying variance
plt.figure(figsize=(8, 6))
for sigma in [0.5, 1, 2]:
  plt.plot(x, gaussian(x, 0, sigma), label=f'Standard Deviation = {sigma}')
plt.title('Gaussian Distribution with Varying Standard Deviation')
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.legend()
plt.show()