from linear_model import LinearModel
from plot import plot_data, plot_mse
import argparse

def main():
	# Parse command-line arguments
	parser = argparse.ArgumentParser(description="Train a linear regression model on car price data.")
	parser.add_argument("data_file", type=str, help="Path to the CSV file containing the data (must include 'km' and 'price' columns).")
	parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for gradient descent.")
	parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations.")
	args = parser.parse_args()

	# Initialize the LinearModel
	model = LinearModel(lr=args.learning_rate, training_iterations=args.iterations)

	# Train the model
	try:
		model.train(args.data_file)
	except Exception as e:
		print(f"Error: {e}")
		return

	# Calculate model precision
	r2_score, mae = model.calculate_precision()
	print(f"Model Precision: r2_score: {r2_score} | mae: {mae}")
	
	# Estimate a price (example 1)
	mileage = 99980  # Replace with desired mileage for prediction
	predicted_price = model.estimate_price(mileage)
	print(f"Predicted price for mileage {mileage}: {predicted_price}")
	
	# Estimate a price (example 2)
	mileage = 300  # Replace with desired mileage for prediction
	predicted_price = model.estimate_price(mileage)
	print(f"Predicted price for mileage {mileage}: {predicted_price}")

	# Estimate a price (example 3)
	mileage = 200000  # Replace with desired mileage for prediction
	predicted_price = model.estimate_price(mileage)
	print(f"Predicted price for mileage {mileage}: {predicted_price}")
	
	# Plot the data and results
	plot_data(model)

if __name__ == "__main__":
	main()
