import pandas as pd

def load_data(file_path):
    """Load Uber trip data from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """Clean the Uber trip data."""
    data['date'] = pd.to_datetime(data['date'])
    data = data.dropna()
    data = data.drop_duplicates()

    return data

if __name__ == "__main__":
    file_path = r"C:\Users\dsharma08k\Personal\Unified Mentor\Uber Trips Analysis\data\uber_trips.csv"
    data = load_data(file_path)
    cleaned_data = clean_data(data)
    cleaned_data.to_csv(r'C:\Users\dsharma08k\Personal\Unified Mentor\Uber Trips Analysis\data\cleaned_uber_trips.csv', index=False)
    print("Data cleaned and saved.")
