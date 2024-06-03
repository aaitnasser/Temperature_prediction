import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


weather = pd.read_csv("london_weather.csv")


weather.info()


weather['date']=pd.to_datetime(weather['date'], format="%Y%m%d")
weather['year']=weather['date'].dt.year
weather['month']=weather['date'].dt.month



weather_metrics= ['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'mean_temp', 'min_temp', 'precipitation', 'pressure', 'snow_depth']
weather_per_month=weather.groupby(['year','month'], as_index=False)[weather_metrics].mean()


sns.lineplot(x="year", y="mean_temp", data=weather_per_month, ci=None)
plt.savefig('mean_temperature.png', format='png')
sns.heatmap(weather.corr(), annot=True)
plt.savefig('correlation heatmap.png', format='png')


feature_selection = ['month', 'cloud_cover', 'sunshine', 'precipitation', 'pressure', 'global_radiation']
target_var = 'mean_temp'
weather = weather.dropna(subset=['mean_temp'])


X = weather[feature_selection]
y = weather[target_var]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


y_train


X_train


# Impute missing values
imputer = SimpleImputer(strategy="mean")
# Fit on the training data
X_train = imputer.fit_transform(X_train)
# Transform on the test data
X_test  = imputer.transform(X_test)


# Scale the data
scaler = StandardScaler()

# Fit the scaler on the training data and transform the training data
# This computes the mean and standard deviation on the training data, then scales the data to have a mean of 0 and a standard deviation of 1
X_train = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
# This scales the test data using the mean and standard deviation computed from the training data
X_test = scaler.transform(X_test)


# Create models with default settings
lin_reg = LinearRegression().fit(X_train, y_train)
tree_reg = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
forest_reg = RandomForestRegressor(random_state=42).fit(X_train, y_train)

# Evaluate performance for Linear Regression
y_pred_lin_reg = lin_reg.predict(X_test)
lin_reg_rmse = mean_squared_error(y_test, y_pred_lin_reg, squared=False)
lin_reg_mae = mean_absolute_error(y_test, y_pred_lin_reg)
lin_reg_r2 = r2_score(y_test, y_pred_lin_reg)

# Evaluate performance for Decision Tree
y_pred_tree_reg = tree_reg.predict(X_test)
tree_reg_rmse = mean_squared_error(y_test, y_pred_tree_reg, squared=False)
tree_reg_mae = mean_absolute_error(y_test, y_pred_tree_reg)
tree_reg_r2 = r2_score(y_test, y_pred_tree_reg)

# Evaluate performance for Random Forest
y_pred_forest_reg = forest_reg.predict(X_test)
forest_reg_rmse = mean_squared_error(y_test, y_pred_forest_reg, squared=False)
forest_reg_mae = mean_absolute_error(y_test, y_pred_forest_reg)
forest_reg_r2 = r2_score(y_test, y_pred_forest_reg)
print(f"-----------Linear Regression-----------------")
# Print performance for Linear Regression
print(f"Linear Regression RMSE: {lin_reg_rmse:.4f}")
print(f"Linear Regression MAE: {lin_reg_mae:.4f}")
print(f"Linear Regression R²: {lin_reg_r2:.4f}")
print(f"-----------Decision Tree-----------------")
# Print performance for Decision Tree
print(f"Decision Tree RMSE: {tree_reg_rmse:.4f}")
print(f"Decision Tree MAE: {tree_reg_mae:.4f}")
print(f"Decision Tree R²: {tree_reg_r2:.4f}")
print(f"-----------Random Forest-----------------")
# Print performance for Random Forest
print(f"Random Forest RMSE: {forest_reg_rmse:.4f}")
print(f"Random Forest MAE: {forest_reg_mae:.4f}")
print(f"Random Forest R²: {forest_reg_r2:.4f}")

# Find and print the best model based on RMSE
best_model_name = 'Linear Regression'
best_rmse = lin_reg_rmse

if tree_reg_rmse < best_rmse:
    best_model_name = 'Decision Tree'
    best_rmse = tree_reg_rmse

if forest_reg_rmse < best_rmse:
    best_model_name = 'Random Forest'
    best_rmse = forest_reg_rmse

print(f"\nBest model: {best_model_name} with RMSE={best_rmse:.4f}")
