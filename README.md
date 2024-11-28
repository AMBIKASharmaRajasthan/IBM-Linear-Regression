# IBM  Regression Project

This repository contains an implementation of **Multiple Linear Regression** using Python and Jupyter Notebook, as part of the IBM Developer Skills Network. The project demonstrates how to use regression models to predict a target variable based on multiple independent variables.

## Overview

The notebook explores the following concepts:

- **Linear Regression**: Predicting a continuous target variable using one or more predictors.
- **Multiple Linear Regression**: Extending linear regression to multiple input features.
- **Evaluation Metrics**: Assessing model performance using metrics like R-squared and Mean Squared Error.
- **Data Preprocessing**: Handling missing values, normalizing features, and preparing data for modeling.

## Key Features

- **Dataset**: A dataset that includes information on CO2 emissions and vehicle features.
- **Tools and Libraries**:
  - Python
  - NumPy, Pandas: For data manipulation.
  - Matplotlib, Seaborn: For data visualization.
  - Scikit-learn: For building and evaluating the regression model.
- **Visualization**: Scatter plots, residual plots, and heatmaps to analyze relationships between variables.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ibm-linear-regression.git
   cd ibm-linear-regression
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open the `ML0101EN-Reg-Mulitple-Linear-Regression-Co2.ipynb` file in Jupyter Notebook.

## Usage

1. **Load the Data**: The dataset is loaded into a Pandas DataFrame.
2. **Explore Relationships**: Visualize data using scatter plots and correlation heatmaps.
3. **Build the Model**: Use Scikit-learn’s `LinearRegression` to train a multiple linear regression model.
4. **Evaluate Performance**: Assess the model using metrics like R-squared and Mean Squared Error.


## Example Output

### Sample Code


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and train the model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Predict and evaluate
predictions = lm.predict(X_test)
print("R-squared:", r2_score(y_test, predictions))
print("Mean Squared Error:", mean_squared_error(y_test, predictions))


### Visualizations
- **Scatter Plot**: Relationship between features and target variable.
- **Residual Plot**: Checking assumptions of linear regression.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

This project was developed as part of the **IBM Developer Skills Network** course on Machine Learning. Special thanks to the IBM team for providing the dataset and resources.


