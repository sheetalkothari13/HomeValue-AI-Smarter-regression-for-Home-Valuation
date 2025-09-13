from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load models
model_names = [
    'LinearRegression', 'RobustRegression', 'RidgeRegression', 'LassoRegression', 'ElasticNet', 
    'PolynomialRegression', 'SGDRegressor', 'ANN', 'RandomForest', 'SVM', 'LGBM', 
    'XGBoost', 'KNN'
]
models = {name: pickle.load(open(f'{name}.pkl', 'rb')) for name in model_names}  #Loads all models from their .pkl files.Creates a dictionary mapping model names to loaded model objects.

# Load evaluation results
results_df = pd.read_csv('model_evaluation_results.csv')
results_df['Model'] = results_df['Model'].str.strip()
# model evaluation results stored in results_df so it can displayed on the /results page

##########
for col in ['MAE', 'MSE', 'R2']:
    if col in results_df.columns:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
results_df = results_df.dropna(subset=['R2'])

# Ensure Accuracy (%) exists (if not created in training script)
if 'Accuracy (%)' not in results_df.columns and 'R2' in results_df.columns:
    results_df['Accuracy (%)'] = results_df['R2'] * 100

# Round for display
results_df = results_df.round({'MAE':2, 'MSE':2, 'R2':4, 'Accuracy (%)':2})

# Best model by R2
if 'R2' in results_df.columns:
    best_model = results_df.loc[results_df['R2'].idxmax(), 'Model']
else:
    best_model = None

records = results_df.to_dict(orient='records')

#########


@app.route('/')  
def index():
    return render_template('index.html', model_names=model_names)
#Renders index.html.Passes the list of model names (model_names) so the frontend can show a dropdown to select a model.

@app.route('/predict', methods=['POST'])  
def predict():
    model_name = request.form['model'] #Defines the predict page (/predict). Only works for POST requests (form submission). Gets the chosen model name from the form (request.form).
    input_data = {
        'Avg. Area Income': float(request.form['Avg. Area Income']),
        'Avg. Area House Age': float(request.form['Avg. Area House Age']),
        'Avg. Area Number of Rooms': float(request.form['Avg. Area Number of Rooms']),
        'Avg. Area Number of Bedrooms': float(request.form['Avg. Area Number of Bedrooms']),
        'Area Population': float(request.form['Area Population'])
    }
    input_df = pd.DataFrame([input_data])
    
    if model_name in models:
        model = models[model_name]
        prediction = model.predict(input_df)[0]
        return render_template('results.html', prediction=round(prediction,2), model_name=model_name)
    else:
        return jsonify({'error': 'Model not found'}), 400

@app.route('/results')
def results():
    return render_template('model.html', records=records, best_model=best_model)
    # return render_template('model.html', tables=[results_df.to_html(classes='data')], titles=results_df.columns.values)
# Converts the DataFrame results_df into an HTML table using .to_html(). Passes this table into model.html, where it gets displayed nicely.

if __name__ == '__main__':  #Runs the Flask app when this file is executed.
    app.run(debug=True)
    
    
