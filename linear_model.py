import pandas as pd
import numpy as np

class LinearModel:
	def __init__(self, lr=0.01, training_iterations=1000, slope=0, intercept=0):
		"""
		Initialize the LinearModel with learning rate, number of iterations, and initial slope/intercept.
		"""
		self.lr = lr
		self.training_iterations = training_iterations
		self.slope = slope
		self.slope_norm = slope
		self.intercept = intercept
		self.intercept_norm = intercept
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

	def calculate_gradients(self, predicted_prices):
		"""
		Calculate the gradients for the intercept and slope based on normalized data.
		"""
		residuals =  predicted_prices - self.df_normalized['price']
		m = len(self.df_normalized['km'])

		#more or less the derivative of MSE with respect to slope
		grad_intercept = 1 / m * residuals.sum()

		#more or less the derivative of MSE with respect to slope
		grad_slope = 1 / m * (residuals * self.df_normalized['km']).sum()
		return grad_intercept, grad_slope

	def calculate_precision(self):
		"""
		Calculate the precision of the model using R² and Mean Absolute Error (MAE).
		"""
		actual_prices = self.df['price']
		predicted_prices = self.df['km'].apply(self.estimate_price)

		# R² score
		ss_total = ((actual_prices - actual_prices.mean()) ** 2).sum()
		ss_residual = ((actual_prices - predicted_prices) ** 2).sum()
		r2_score = 1 - (ss_residual / ss_total)

		# Mean Absolute Error (MAE)
		mae = np.mean(np.abs(actual_prices - predicted_prices))

		return r2_score, mae
	
	def loadData(self, data):
		"""
		Load the data and normalize it.
		"""
		self.df = pd.read_csv(data)
		if 'km' not in self.df.columns or 'price' not in self.df.columns:
			raise ValueError("DataFrame must contain 'km' and 'price' columns")
		if self.df.empty:
			raise ValueError("DataFrame is empty")
		self.normalize_data()

		
	def train(self, data):
		"""
		Train the linear model using gradient descent.
		"""
		self.loadData(data)

		#start the training iterations
		for i in range(self.training_iterations):

			#calculate the predictions with the last computed values for slope and intercept
			predicted_prices = self.slope_norm * self.df_normalized['km'] + self.intercept_norm

			#calculate the gradients for the predicted prices
			grad_intercept, grad_slope = self.calculate_gradients(predicted_prices)

			#calc the steps
			step_intercept = self.lr * grad_intercept
			step_slope = self.lr * grad_slope

			#stop training for small steps
			if step_intercept < 0.001 and step_slope < 0.001:
				break
			
			#update intercept and slope
			self.intercept_norm -= step_intercept
			self.slope_norm -= step_slope
			
			#denormailze intercept and slope to calculate precision
			# self.slope = self.denormalize_slope(self.slope_norm)
			# self.intercept = self.denormalize_intercept(self.intercept_norm, self.slope)
			# r2_score, mae = self.calculate_precision()
			# print(f"Training iteration: {i} | r2_score: {r2_score} | mae: {mae}")
		self.slope = self.denormalize_slope(self.slope_norm)
		self.intercept = self.denormalize_intercept(self.intercept_norm, self.slope)
