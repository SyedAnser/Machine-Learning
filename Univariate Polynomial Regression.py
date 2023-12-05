import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
import statsmodels.api as sm

df=pd.read_excel("C:/Users/syeda/OneDrive/Desktop/Datasets/concrete+compressive+strength/Concrete_Data.xls")

df.replace('?', np.nan, inplace=True)
new_df=df.dropna()
x=new_df.drop(columns='Concrete compressive strength(MPa, megapascals) ')
y=new_df['Concrete compressive strength(MPa, megapascals) ']

mse_values = []
p_values = []

for predictor_variable in x.columns:
    # Create a scatterplot
    mse_per_variable = []
    p_per_variable = []
    plt.figure(figsize=(8, 6))
    plt.scatter(x[predictor_variable], [y], label='Data points')
    plt.xlabel(predictor_variable)
    plt.ylabel("Strength")
    plt.title(f'Scatterplot of {"Strength"} vs. {predictor_variable}')

    for k in range(1, 5):
        


        poly = PolynomialFeatures(degree=k)
        X_poly = poly.fit_transform(x[[predictor_variable]])
        X_poly = sm.add_constant(X_poly)  # Add a constant term for the intercept
        model = sm.OLS(y, X_poly).fit()
        
        
        # Make predictions
        y_pred = model.predict(X_poly)
        
        # Calculate Mean Squared Error
        mse = mean_squared_error(y, y_pred)
        mse_per_variable.append(mse)
        
        # Perform a linear regression to calculate the p-value
        p_value = model.pvalues[1]  # Assuming the predictor variable is the second coefficient
        p_per_variable.append(p_value)
        
        print(f'Polynomial Regression (k={k}) for {"Strength"} vs. {predictor_variable}')
        print(f'Mean Squared Error: {mse}')
        print(f'p-value: {p_value}')
        print(f'---')
        
        # Plot the regression line
        x_range = np.linspace(min(x[predictor_variable]), max(x[predictor_variable]), 100)
        y_range = model.predict(poly.fit_transform(x_range.reshape(-1, 1)))
        plt.plot(x_range, y_range, label=f'Poly k={k}', linewidth=2)

    plt.legend()
    plt.show()

    mse_values.append(mse_per_variable)
    p_values.append(p_per_variable)

plt.figure(figsize=(12, 12))
for i, predictor_variable in enumerate(x.columns):
    plt.plot(range(1, 5), mse_values[i], label=f'{predictor_variable}', marker='o')
plt.xlabel('Polynomial Order (k)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE vs. Polynomial Order (k) for Each Predictor Variable')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()

# Create plots of p-value vs. k for each predictor variable
plt.figure(figsize=(12, 12))
for i, predictor_variable in enumerate(x.columns):
    plt.plot(range(1, 5), p_values[i], label=f'{predictor_variable}', marker='o')
plt.xlabel('Polynomial Order (k)')
plt.ylabel('p-value')
plt.title('p-value vs. Polynomial Order (k) for Each Predictor Variable')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()