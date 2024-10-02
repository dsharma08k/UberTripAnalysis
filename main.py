import os
from scripts import data_cleaning, feature_engineering, train_model, visualization

def run_pipeline():
    """Run the entire Uber Trip Analysis pipeline."""
    
    # Data Cleaning
    print("Cleaning data...")
    data = data_cleaning.load_data(r'C:\Users\dsharma08k\Personal\Unified Mentor\Uber Trips Analysis\data\uber_trips.csv')
    clean_data = data_cleaning.clean_data(data)
    clean_data.to_csv(r'C:\Users\dsharma08k\Personal\Unified Mentor\Uber Trips Analysis\data\cleaned_uber_trips.csv', index=False) 
    print("Cleaned data saved.")

    # Feature Engineering
    print("Adding features...")
    featured_data = feature_engineering.add_time_features(clean_data)
    featured_data = feature_engineering.create_lagged_features(featured_data)
    featured_data.to_csv(r'C:\Users\dsharma08k\Personal\Unified Mentor\Uber Trips Analysis\data\featured_uber_trips.csv', index=False)
    print("Featured data saved.")

    # Model Training (Trip Demand Prediction)
    print("Training trip demand model...")
    X_train, X_test, y_train, y_test = train_model.load_and_split_data(r'C:\Users\dsharma08k\Personal\Unified Mentor\Uber Trips Analysis\data\featured_uber_trips.csv')
    model = train_model.train_model(X_train, y_train)
    mse, r2 = train_model.evaluate_model(model, X_test, y_test)

    # Visualizations
    print("Generating visualizations...")
    os.makedirs('visualizations', exist_ok=True)
    visualization.generate_all_visualizations(featured_data, y_test, model.predict(X_test))
    
    print("Pipeline complete. Visualizations saved to 'visualizations' folder.")

if __name__ == "__main__":
    run_pipeline()
