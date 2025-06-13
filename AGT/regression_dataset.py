import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate x values between -3 and 3
x = np.linspace(-3, 3, 1000000)

# Generate normal noise
noise = np.random.normal(loc=0, scale=5, size=x.shape)

# Compute y = x^3 + noise
y = x**3 + noise

# Create a DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# Save to CSV (optional)
df.to_csv('AGT/cubic_with_noise.csv', index=False)

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', alpha=0.6, label='Data with noise')
plt.plot(x, x**3, color='red', label='True function: $x^3$')
plt.title('Dataset for $y = x^3 + \\text{noise}$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig("AGT/reg.png")
