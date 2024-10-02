import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def load_and_split_data(file_path):
    """Load data and split it into features and target for trip demand prediction."""
    data = pd.read_csv(file_path)
    X = data[['active_vehicles', 'year', 'month', 'day', 'weekday', 'hour', 'lag_1_trips']]
    y = data['trips']
    
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_model(X_train, y_train):
    """Train a Random Forest model on the training data."""
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test data."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")
    return mse, r2

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data(r"C:\Users\dsharma08k\Personal\Unified Mentor\Uber Trips Analysis\data\featured_uber_trips.csv")
    
    model = train_model(X_train, y_train)
    
    mse, r2 = evaluate_model(model, X_test, y_test)
    
    with open(r'C:\Users\dsharma08k\Personal\Unified Mentor\Uber Trips Analysis\models\trip_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to models/trip_model.pkl")
