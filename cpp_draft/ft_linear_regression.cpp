//c++ -std=c++20 -I/usr/local/include -L/usr/local/lib -lmatplot ft_linear_regression.cpp -o ft_linear_regression

#ifndef LINEARMODEL_CPP
# define LINEARMODEL_CPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <matplot/matplot.h>
#include "normalize.cpp"

class LinearModel{
	private:
		double	_intercept;
		double	_slope;
		double	_lr; //learning rate
		std::vector<int>	_mileage;
		std::vector<double>	_n_mileage;
		std::vector<int>	_price;
		std::vector<double>	_n_price;
		std::vector<double> _n_slopes;
		std::vector<double> _n_intercepts;

		double	ssr(double slope, double intercept);

	public:
		LinearModel() : _intercept(0), _slope(0), _lr(0.01) {};
		~LinearModel() {};
		double	estimatePrice(double mileage);
		void	trainModel(const std::string &data_csv);
		void	readCsv(const std::string &data_csv);
		void	plot_data();
		void	plot_ssr();
} ;

void	LinearModel::plot_data() {
	using namespace matplot;
	
	//converting to double
    std::vector<double> mileage(_mileage.begin(), _mileage.end());
    std::vector<double> price(_price.begin(), _price.end());
    scatter(mileage, price);

	// Calculate the range for x values of the line
	auto min_max = std::minmax_element(_mileage.begin(), _mileage.end());
	double x_min = *min_max.first;
	double x_max = *min_max.second;

	// Calculate the y values for the regression line
	std::vector<double> line_x = {x_min, x_max};
	std::vector<double> line_y = {_slope * x_min + _intercept, _slope * x_max + _intercept};
    
	hold(on);

    // Plot the regression line
    plot(line_x, line_y, "b-"); // "b-" is the blue line style

	title("Training Data");
	xlabel("Mileage in km");
    ylabel("Price");
    show();
}

double	LinearModel::ssr(double slope, double intercept) {
	double	ssr = 0;
	for (int i = 0; i < _mileage.size(); i ++) {
		ssr += pow(_price[i] - (intercept + slope * _mileage[i]), 2);
	}
	return ssr;
}

void	LinearModel::plot_ssr() {
	using namespace matplot;

	std::vector<double>	slopes = linspace(-2, 2, 100);;
	std::vector<double>	intercepts = linspace(-100000, 100000, 100);;

	// Generate meshgrid
    auto [X, Y] = meshgrid(intercepts, slopes);

    // Compute Z as the sum of squared residuals for each (slope, intercept) pair
    std::vector<std::vector<double>> Z(intercepts.size(), std::vector<double>(slopes.size()));
    for (size_t i = 0; i < intercepts.size(); ++i) {
        for (size_t j = 0; j < slopes.size(); ++j) {
            Z[i][j] = ssr(slopes[j], intercepts[i]);
        }
    }

    // Create a 3D surface plot
    mesh(X, Y, Z);
    title("Surface Plot of Sum of Squared Residuals");
    ylabel("Slope");
    xlabel("Intercept");
    zlabel("Sum of Squared Residuals");

    show();
}

double	LinearModel::estimatePrice(double mileage) {
	return (this->_intercept + this->_slope * mileage);
}

void	LinearModel::readCsv(const std::string &data_csv){
	std::ifstream file(data_csv);
	std::string line;
	bool isFirstLine = true;

	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string value;
		
		if (isFirstLine) {
			isFirstLine = false;
			continue;
		}

		std::getline(ss, value, ',');
		_mileage.emplace_back(std::stoi(value));
		std::getline(ss, value, ',');
		_price.emplace_back(std::stoi(value));		
		std::cout << _mileage.back() << " | " << _price.back() << std::endl;
	}
}

void	LinearModel::trainModel(const std::string &data_csv) {
	std::size_t	data_size;

	int			learning_iterations = 1000;
	double		residual;
	double		grad_inter, grad_slo;

	readCsv(data_csv);
	data_size = _mileage.size();
	z_score_normalize(_mileage, _price, _n_mileage, _n_price);

	for (int i = 0; i < learning_iterations; i++) {
		grad_inter = 0.0;
		grad_slo = 0.0;
		//calculate the derivation of the sum of the squared residuals with respect to intercept and slope
		//2 or more derivatives of the same functions are called gradient
		for (int j = 0; j < data_size; j++) {
			residual = _n_price[j] - estimatePrice(_n_mileage[j]);
			grad_inter += residual;
			grad_slo += residual * _n_mileage[j]; 
		}
		//update intersect and slope
		grad_inter *= -2;
		grad_slo *= -2;

		if (fabs(grad_inter) < 0.001 && fabs(grad_slo) < 0.001) {
			this->_slope = denormalize_slope(this->_slope, _mileage, _price);
			this->_intercept = denormalize_intercept(this->_intercept, this->_slope, _mileage, _price);
			return;
		}
		this->_intercept -= this->_lr * grad_inter;
		this->_slope -= this->_lr * grad_slo;
		//for plotting ssr
		_n_slopes.emplace_back(_slope);
		_n_intercepts.emplace_back(_intercept);
	}
	this->_slope = denormalize_slope(this->_slope, _mileage, _price);
	this->_intercept = denormalize_intercept(this->_intercept, this->_slope, _mileage, _price);
}

int	main() {

	LinearModel lm;
	std::cout << "estimate price for 84000 km before training: " << std::endl;
	std::cout << lm.estimatePrice(84000) << std::endl;
	lm.trainModel("data.csv");
	std::cout << "estimate price for 60949 km after training: " << std::endl;
	std::cout << lm.estimatePrice(60949) << std::endl;
	std::cout << "estimate price for 185530 km after training: " << std::endl;
	std::cout << lm.estimatePrice(185530) << std::endl;
	std::cout << "estimate price for 0 km after training: " << std::endl;
	std::cout << lm.estimatePrice(0) << std::endl;
	//lm.plot_data();
	//lm.plot_ssr();
}

#endif