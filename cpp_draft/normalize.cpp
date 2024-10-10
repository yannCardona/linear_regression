#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>

// Function to calculate mean of a vector
double mean(const std::vector<int>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// Function to calculate standard deviation of a vector
double standard_deviation(const std::vector<int>& v, double mean) {
    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0,
        std::plus<double>(), [mean](int x, int y) {
            return (x - mean) * (x - mean);
        });
    return std::sqrt(sq_sum / v.size());
}

// Function to Z-Score normalize a vector of pairs
void    z_score_normalize(std::vector<int> mileage, std::vector<int> price, std::vector<double> &n_mileage, std::vector<double> &n_price) {

    // Calculate means and standard deviations
    double mean_mileage = mean(mileage);
    double std_mileage = standard_deviation(mileage, mean_mileage);
    double mean_price = mean(price);
    double std_price = standard_deviation(price, mean_price);

    // Normalize the data
    for (size_t i = 0; i < mileage.size(); ++i) {
        n_mileage.emplace_back((mileage[i] - mean_mileage) / std_mileage);
        n_price.emplace_back((price[i] - mean_price) / std_price);
    }
}

double denormalize_slope(double n_slope, std::vector<int> &mileage, std::vector<int> &price) {

    // Calculate means and standard deviations
    double mean_mileage = mean(mileage);
    double std_mileage = standard_deviation(mileage, mean_mileage);
    double mean_price = mean(price);
    double std_price = standard_deviation(price, mean_price);
    
    return n_slope * (std_price / std_mileage);
}

double denormalize_intercept(double n_inter, double slope, std::vector<int> &mileage, std::vector<int> &price) {

    // Calculate means and standard deviations
    double mean_mileage = mean(mileage);
    double std_mileage = standard_deviation(mileage, mean_mileage);
    double mean_price = mean(price);
    double std_price = standard_deviation(price, mean_price);
    
    return n_inter * std_price + mean_price - slope * mean_mileage;
}