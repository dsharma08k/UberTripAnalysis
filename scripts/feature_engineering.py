import pandas as pd

def add_time_features(data):
    """Add time-based features to the dataset."""
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['weekday'] = data['date'].dt.weekday
    data['hour'] = data['date'].dt.hour
    return data

def create_lagged_features(data, lag=1):
    """Create lagged features for trip demand."""
    data[f'lag_{lag}_trips'] = data['trips'].shift(lag)
    return data

if __name__ == "__main__":
    file_path = r"C:\Users\dsharma08k\Personal\Unified Mentor\Uber Trips Analysis\data\cleaned_uber_trips.csv"
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date']) 
    
    # Adding features
    data = add_time_features(data)
    data = create_lagged_features(data, lag=1) 
    
    data.to_csv(r'C:\Users\dsharma08k\Personal\Unified Mentor\Uber Trips Analysis\data\featured_uber_trips.csv', index=False)
    print("Features added and data saved.")
