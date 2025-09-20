import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, is_train=True, columns=None):
    categorical_columns = ['Crop', 'Season', 'State']
    if is_train:
        original_categories = {col: data[col].unique() for col in categorical_columns}
        data = pd.get_dummies(data, columns=categorical_columns)
        data = data.dropna()
        columns = data.columns
        return data, columns, original_categories
    else:
        data = pd.get_dummies(data, columns=categorical_columns)
        data = data.reindex(columns=columns, fill_value=0)
        return data, columns, categorical_columns

def train_model(data):
    X = data.drop(columns=['Yield'])
    y = data['Yield']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, X_train.columns, mae, r2

def save_model(model, columns, mae, r2, file_name):
    joblib.dump((model, columns, mae, r2), file_name)

def load_model(file_name):
    return joblib.load(file_name)

if __name__ == "__main__":
    data_path = 'data/dataset.csv'
    data = load_data(data_path)
    processed_data, columns, _ = preprocess_data(data)
    model, feature_columns, mae, r2 = train_model(processed_data)
    save_model(model, feature_columns, mae, r2, 'model/crop_yield_model.pkl')
