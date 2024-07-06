# Importing necessary libraries for data manipulation, visualization, and analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

"""
fetch_ohlc_data:
This function fetches the historical open, high, low, and close (OHLC) data for a given ticker over a specified number
of days. It's important when providing the raw data needed to analyze the market movements.
"""


def fetch_ohlc_data(ticker_symbol, days=100, interval='1d'):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = yf.download(ticker_symbol, start=start_date, end=end_date, interval=interval)
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker_symbol}")
    df.reset_index(inplace=True)
    return df


"""
fetch_intraday_data:
Retrieves intraday trading data for a specific date. This is essential for understanding the intraday price dynamics
and supports higher-resolution analysis such as identifying specific intraday patterns or trends.
"""


def fetch_intraday_data(ticker_symbol, date, interval='1h'):
    ticker = yf.Ticker(ticker_symbol)
    start_date = date.strftime('%Y-%m-%d')
    end_date = (date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    df = ticker.history(start=start_date, end=end_date, interval=interval)

    if df.empty:
        raise ValueError(f"No intraday data found for {ticker_symbol} on {date}")

    df.reset_index(inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df[df['Datetime'].dt.date == date.date()]
    df = df.set_index('Datetime').between_time('09:30', '16:00').reset_index()

    if df.empty:
        raise ValueError(f"No intraday data found for {ticker_symbol} on {date} within market hours")

    return df


"""
normalize_ohlc:
Normalizes the OHLC data relative to the opening price. This transformation is key for comparing price movements
across different trading sessions, regardless of the absolute price levels. This helps normalize all data. The charts
open at the 0 x-axis and end at the 0 x-axis. Each OHLC line don't finish in relativity to each other, rather at 0 to
normalize the data and create a chart that is easier to read.
"""


def normalize_ohlc(df):
    normalized_df = df.copy()
    normalized_df['Move_High'] = df['High'] - df['Open']
    normalized_df['Move_Low'] = df['Low'] - df['Open']
    normalized_df['Move_Close'] = df['Close'] - df['Open']
    normalized_df['Move_Open'] = 0
    return normalized_df


"""
apply_kmeans_clustering:
Applies K-means clustering to identify common price movement patterns. The KMeans clustering isn't actually useful
from what I have tested, but it gives an idea of how we can use this chart. A more effective way to do this would
be to find the Apex support and resistance points from initial or secondary moves.
"""


def apply_kmeans_clustering(high_moves, low_moves, n_clusters=5):
    data = np.concatenate((high_moves, low_moves)).reshape(-1, 1)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_scaled)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    clusters = scaler.inverse_transform(cluster_centers).flatten()
    return clusters


"""
plot_ohlc_moves:
Visualizes price movements within a trading day using a density map. This function is vital for visual analysis, 
allowing traders to visually identify patterns and potentially profitable trading setups based on intraday price
actions.
"""


def plot_ohlc_moves(df, ticker_symbol, ax):
    ax.clear()
    ax.set_title(f"Density Map for {ticker_symbol}")
    ax.set_xlabel('Normalized Time')
    ax.set_ylabel('Move in Dollars')
    high_moves = df['Move_High'].values
    low_moves = df['Move_Low'].values
    open_price = df['Open'].iloc[-1]

    # Process each trading day's data
    for i in range(len(df)):
        date = df['Date'].iloc[i]
        high_move = df['Move_High'].iloc[i]
        low_move = df['Move_Low'].iloc[i]

        try:
            intraday_data = fetch_intraday_data(ticker_symbol, date)
            if len(intraday_data) < 2:
                raise ValueError("Not enough intraday data to process.")

            high_idx = intraday_data['High'].idxmax()
            low_idx = intraday_data['Low'].idxmin()
            high_time = intraday_data['Datetime'][high_idx]
            low_time = intraday_data['Datetime'][low_idx]

            # Normalize time for plotting
            total_seconds = (intraday_data['Datetime'].iloc[-1] - intraday_data['Datetime'].iloc[0]).total_seconds()
            if total_seconds == 0:
                raise ValueError("Total seconds for intraday data is zero, cannot normalize time.")

            x_high = (high_time - intraday_data['Datetime'].iloc[0]).total_seconds() / total_seconds
            x_low = (low_time - intraday_data['Datetime'].iloc[0]).total_seconds() / total_seconds

            # Create spline curve based on price movements
            if x_low < x_high:
                x_coords = [0, 0.25, 0.75, 1]
                y_coords = [0, low_move, high_move, 0]
            else:
                x_coords = [0, 0.25, 0.75, 1]
                y_coords = [0, high_move, low_move, 0]

            cs = CubicSpline(x_coords, y_coords)
            x_new = np.linspace(0, 1, 100)
            y_new = cs(x_new)

            # Plot the density map
            ax.plot(x_new, y_new, color='black', alpha=0.6, linewidth=1)
            ax.fill_between([0, 1], low_move, high_move, color='#363636', alpha=0.1)
        except ValueError as ve:
            print(f"Error processing date {date}: {ve}")
            continue

    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)

    clusters = apply_kmeans_clustering(high_moves, low_moves)
    supports = sorted([c for c in clusters if c < 0])
    resistances = sorted([c for c in clusters if c > 0])

    # Mark supports and resistances on the chart
    for i, support in enumerate(supports):
        support_price = open_price + support
        ax.axhline(support, color='#00cf0e', linestyle='-', linewidth=0.5, label=f'Support {i+1} at {support_price:.2f}')
        ax.text(0.5, support, f'Support {i+1} at {support_price:.2f}', ha='center', va='top',
                color='#00cf0e', fontsize=7, fontweight='bold')

    for i, resistance in enumerate(resistances):
        resistance_price = open_price + resistance
        ax.axhline(resistance, color='#0088ff', linestyle='-', linewidth=0.5, label=f'Resistance {i+1} at {resistance_price:.2f}')
        ax.text(0.5, resistance, f'Resistance {i+1} at {resistance_price:.2f}', ha='center', va='top',
                color='#0088ff', fontsize=7, fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    ax.figure.canvas.draw()


"""
main:
This function orchestrates the data fetching, normalization, and plotting processes. It's the control center
for executing the defined analysis, handling exceptions, and ensuring that the data visualization 
is accurately represented for the specified ticker and time frame.
"""

def main(ticker_symbol, days, ax):
    try:
        df = fetch_ohlc_data(ticker_symbol, days=days)
        normalized_df = normalize_ohlc(df)
        plot_ohlc_moves(normalized_df, ticker_symbol, ax)
    except ValueError as e:
        print(e)


"""
__main__:
This is the entry point of the script when run directly. It sets up the plot environment and parses command-line
arguments to drive the script's functionality, it allows the user to specify the ticker symbol and the number
of days for analysis directly from the command line.
"""

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) != 3:
        print("Usage: python main_plot.py <ticker_symbol> <days>")
    else:
        ticker_symbol = sys.argv[1]
        days = int(sys.argv[2])

        fig, ax = plt.subplots(figsize=(15, 9))
        main(ticker_symbol, days, ax)
        plt.show()
