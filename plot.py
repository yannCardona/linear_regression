import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_data(model):

	# Scatter plot of original data
	plt.scatter(model.df['km'], model.df['price'], color='blue', label='Data')

	# Plotting the predicted prices (regression line)
	plt.plot(model.df['km'], model.slope * model.df['km'] + model.intercept, color='red', label='Regression Line')

	# Adding labels and title
	plt.xlabel('Mileage (km)')
	plt.ylabel('Price')
	plt.title('Linear Regression: Mileage vs Price')
	plt.legend()

	# Show plot
	plt.show()

def ssr(self, slope, intercept):
	return np.sum((self.df['price'] - (intercept + slope * self.df['km'])) ** 2)
	
def plot_ssr(model):

	# Define ranges for slopes and intercepts
	slopes = np.linspace(-2, 2, 100)
	intercepts = np.linspace(0, 100000, 100)

	# Generate meshgrid
	X, Y = np.meshgrid(intercepts, slopes)

	# Compute Z as the sum of squared residuals for each (slope, intercept) pair
	Z = np.zeros_like(X)
	for i in range(len(intercepts)):
		for j in range(len(slopes)):
			Z[j, i] = model.ssr(slopes[j], intercepts[i])

	# Create a 3D surface plot
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X, Y, Z, cmap='viridis')

	ax.set_title("Surface Plot of Sum of Squared Residuals")
	ax.set_xlabel("Intercept")
	ax.set_ylabel("Slope")
	ax.set_zlabel("Sum of Squared Residuals")

	plt.show()