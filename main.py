import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline # prevents data leakages
from sklearn.neural_network import MLPRegressor #Multi-layer Perceptron regressor-feedforward artificial neural network-one or more hidden layers between the input and output layers.
import lightgbm as lgb
import xgboost as xgb    
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

data = pd.read_csv(r"D:\Naresh It Classes\August\29th\USA_Housing.csv")

x =data.drop(['Price','Address'],axis=1)
y = data['Price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

models = {
    'LinearRegression': LinearRegression(),
    'RobustRegression': HuberRegressor(),
    'RidgeRegression': Ridge(),
    'LassoRegression': Lasso(),
    'ElasticNet': ElasticNet(),
    'PolynomialRegression': Pipeline([
        ('poly', PolynomialFeatures(degree=4)),
        ('linear', LinearRegression())
    ]),
    'SGDRegressor': SGDRegressor(),
    'ANN': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
    'RandomForest': RandomForestRegressor(),
    'SVM': SVR(),
    'LGBM': lgb.LGBMRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'KNN': KNeighborsRegressor()
}

'''
This dictionary maps a name to an instance of a model. The code will iterate over this dictionary and train each model.

Special notes:
PolynomialRegression is a Pipeline that first expands features to degree 4 and then fits a linear model. Polynomial degree 4 can create many features and may overfit if you don’t have lots of data.
MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000) means one hidden layer with 100 neurons, and a maximum of 1000 iterations for training.
'''

results = []  #Empty list: each trained model’s evaluation metrics will be appended here.

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    })
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
        
'''
Loops over each model:
model.fit(...): trains the model on the training data.
model.predict(x_test): gets predicted prices for the test set.
Compute metrics:
    MAE (mean absolute error): average absolute difference between predicted and true values (same units as the target). Lower is better.
    MSE (mean squared error): average squared difference (penalizes large errors more). Lower is better.
    R2 (coefficient of determination): how much variance in y is explained by the model. 1.0 is perfect, 0 means the model does no better than predicting the mean; can be negative if model is worse than constant mean predictor.
    Append the results to the results list.
    Save the trained model to a file named like LinearRegression.pkl using pickle.

Important notes / gotchas:
If a model fails to train (e.g., because of NaN values, categorical data in x, or incompatible data types), model.fit will raise an error. Always check x.dtypes and x.isna().sum() first.
Some models need scaled input; you should use a Pipeline with StandardScaler or MinMaxScaler for those models to avoid unfairly bad performance.
Saving with pickle mixes code + data — when re-loading, you must use the same library versions (scikit-learn, xgboost, etc.) or you may get errors.
'''

results_df = pd.DataFrame(results)
########################

# add at top of file if not already present
import os
import matplotlib.pyplot as plt

# ensure plots folder exists
plot_dir = os.path.join(os.path.dirname(__file__), "static", "plots")
os.makedirs(plot_dir, exist_ok=True)

# --- After results_df = pd.DataFrame(results) ---
# Ensure column names are what we expect
# expected columns: ['Model','MAE','MSE','R2']
# (If your keys were different, adjust accordingly.)

# Clean model names (avoid space/case mismatch later)
results_df['Model'] = results_df['Model'].str.strip()

# Add Accuracy (%) column and round numbers:
results_df['Accuracy (%)'] = results_df['R2'] * 100
results_df = results_df.round({'MAE': 2, 'MSE': 2, 'R2': 4, 'Accuracy (%)': 2})

# Mark best model (highest R2)
best_idx = results_df['R2'].idxmax()
results_df['Best'] = False
results_df.loc[best_idx, 'Best'] = True

# Save CSV
results_df.to_csv('model_evaluation_results.csv', index=False)

# -----------------------
# Create simple matplotlib plots (one per file)
# 1) R2 bar chart
plt.figure()
plt.bar(results_df['Model'], results_df['R2'])
plt.xticks(rotation=45, ha='right')
plt.yscale('log')
plt.ylabel('R2')
plt.title('R2 by Model')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "r2.png"))
plt.close()

# 2) MAE bar chart
plt.figure()
plt.bar(results_df['Model'], results_df['MAE'])
plt.xticks(rotation=45, ha='right')
plt.yscale('log')
plt.ylabel('MAE')
plt.title('MAE by Model')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "mae.png"))
plt.close()

# 3) MSE bar chart (optional)
plt.figure()
plt.bar(results_df['Model'], results_df['MSE'])
plt.xticks(rotation=45, ha='right')

plt.ylabel('MSE')
plt.title('MSE by Model')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "mse.png"))
plt.close()


##########################

print("Models have been trained and saved as pickle files. Evaluation results have been saved to model_evaluation_results.csv.")



















