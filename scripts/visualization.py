import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_trips_over_time(data):
    """Plot the number of trips over time."""
    plt.figure(figsize=(10, 6))
    data.groupby('date')['trips'].sum().plot()
    plt.title('Trips Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Trips')
    plt.savefig(r'C:\Users\dsharma08k\Personal\Unified Mentor\Uber Trips Analysis\visualizations\trips_over_time.png')
    plt.close()

def plot_active_vehicles_over_time(data):
    """Plot the number of active vehicles over time."""
    plt.figure(figsize=(10, 6))
    data.groupby('date')['active_vehicles'].sum().plot()
    plt.title('Active Vehicles Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Active Vehicles')
    plt.savefig(r'C:\Users\dsharma08k\Personal\Unified Mentor\Uber Trips Analysis\visualizations\active_vehicles_over_time.png')
    plt.close()

def plot_predictions_vs_actual(y_test, y_pred):
    """Plot predictions vs actual trips."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel('Actual Trips')
    plt.ylabel('Predicted Trips')
    plt.title('Actual vs Predicted Trips')
    plt.savefig(r'C:\Users\dsharma08k\Personal\Unified Mentor\Uber Trips Analysis\visualizations\predictions_vs_actual.png')
    plt.close()

def generate_all_visualizations(data, y_test=None, y_pred=None):
    """Generate all visualizations and save them to the visualizations folder."""
    plot_trips_over_time(data)
    plot_active_vehicles_over_time(data)
    if y_test is not None and y_pred is not None:
        plot_predictions_vs_actual(y_test, y_pred)
