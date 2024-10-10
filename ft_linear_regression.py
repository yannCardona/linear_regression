import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LinearModel:
	def __init__(self, lr=0.01, learning_iterations=1000, slope=0, intercept=0):
		"""
		Initialize the LinearModel with learning rate, number of iterations, and initial slope/intercept.
		"""
		self.lr = lr
		self.learning_iterations = learning_iterations
		self.slope = slope
		self.intercept = intercept
		self.df = None
		self.df_normalized = None
		self.mean_mileage = 0
		self.std_mileage = 0
		self.mean_price = 0
		self.std_price = 0

	def estimate_price(self, mileage):
		"""
		Estimate the price based on mileage using the current model parameters.
		"""
		return self.slope * mileage + self.intercept

	def normalize_data(self):
		"""
		Normalize the data by standardizing the mileage and price columns.
		"""
		self.mean_mileage = self.df['km'].mean()
		self.std_mileage = self.df['km'].std()
		self.mean_price = self.df['price'].mean()
		self.std_price = self.df['price'].std()
		self.df_normalized = (self.df - self.df.mean()) / self.df.std()
	
	def denormalize_slope(self, slope):
		"""
		Denormalize the slope to the original scale.
		"""
		return slope * (self.std_price / self.std_mileage)

	def denormalize_intercept(self, intercept, slope):
		"""
		Denormalize the intercept to the original scale.
		"""
		return intercept * self.std_price + self.mean_price - slope * self.mean_mileage

	def calculate_gradients(self):
		"""
		Calculate the gradients for the intercept and slope based on normalized data.
		"""
		residuals = self.df_normalized['price'] - self.estimate_price(self.df_normalized['km'])
		grad_intercept = -2 * residuals.sum()
		grad_slope = -2 * (residuals * self.df_normalized['km']).sum()
		return grad_intercept, grad_slope
	
	def plot_data(self):

		# Scatter plot of original data
		plt.scatter(self.df['km'], self.df['price'], color='blue', label='Data')

		# Plotting the predicted prices (regression line)
		plt.plot(self.df['km'], self.slope * self.df['km'] + self.intercept, color='red', label='Regression Line')

		# Adding labels and title
		plt.xlabel('Mileage (km)')
		plt.ylabel('Price')
		plt.title('Linear Regression: Mileage vs Price')
		plt.legend()

		# Show plot
		plt.show()

	def ssr(self, slope, intercept):
		return np.sum((self.df['price'] - (intercept + slope * self.df['km'])) ** 2)
	
	def plot_ssr(self):

		# Define ranges for slopes and intercepts
		slopes = np.linspace(-2, 2, 100)
		intercepts = np.linspace(-100000, 100000, 100)

		# Generate meshgrid
		X, Y = np.meshgrid(intercepts, slopes)

		# Compute Z as the sum of squared residuals for each (slope, intercept) pair
		Z = np.zeros_like(X)
		for i in range(len(intercepts)):
			for j in range(len(slopes)):
				Z[j, i] = self.ssr(slopes[j], intercepts[i])

		# Create a 3D surface plot
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot_surface(X, Y, Z, cmap='viridis')

		ax.set_title("Surface Plot of Sum of Squared Residuals")
		ax.set_xlabel("Intercept")
		ax.set_ylabel("Slope")
		ax.set_zlabel("Sum of Squared Residuals")

		plt.show()
		

	def train(self, data):
		"""
		Train the linear model using gradient descent.
		"""
		self.df = pd.read_csv(data)

		if 'km' not in self.df.columns or 'price' not in self.df.columns:
			raise ValueError("DataFrame must contain 'km' and 'price' columns")

		if self.df.empty:
			raise ValueError("DataFrame is empty")

		self.normalize_data()

		for _ in range(self.learning_iterations):
			grad_intercept, grad_slope = self.calculate_gradients()

			if abs(grad_intercept) < 0.001 and abs(grad_slope) < 0.001:
				break
			
			self.intercept -= self.lr * grad_intercept
			self.slope -= self.lr * grad_slope
		
		# Final denormalization
		self.slope = self.denormalize_slope(self.slope)
		self.intercept = self.denormalize_intercept(self.intercept, self.slope)

# Example usage
model = LinearModel(lr=0.01, learning_iterations=1000)
model.train('data.csv')

print(model.estimate_price(60949))
model.plot_ssr()