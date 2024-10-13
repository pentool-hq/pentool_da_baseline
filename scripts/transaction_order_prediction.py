# scripts/transaction_order_prediction.py

import pandas as pd
import numpy as np
from scipy.stats import expon, gamma, weibull_min, pareto, burr, lognorm, beta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import TimeSeriesSplit
from joblib import Parallel, delayed
from tabulate import tabulate
import logging
import warnings
import plotly.express as px  # For Plotly visualizations

from .config import load_config, validate_selection  # Import configurations and validation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
config = load_config()

# Set Parameters from config
BUY_YT = config.get('BUY_YT', False)
SELL_YT = config.get('SELL_YT', True)
BUY_PT = config.get('BUY_PT', False)
SELL_PT = config.get('SELL_PT', False)
YT_OR_PT_AMOUNT = config.get('YT_OR_PT_AMOUNT', 0.01)

# Suppress Warnings
warnings.filterwarnings('ignore')

def preprocess_data(df, buy):
    """
    Preprocess the transaction data.
    """
    required_columns = ['timestamp', 'input_baseType', 'valuation_acc']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame.")

    # Convert 'timestamp' to datetime and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp')

    # Define 'buy_sell' column: PT = 1, others = 0
    df['buy_sell'] = (df['input_baseType'] == 'PT').astype(int)

    # Filter orders based on user selection and amount
    AMOUNT = YT_OR_PT_AMOUNT
    if buy == 1:
        df_filtered = df[
            (df['buy_sell'] == 1) &
            (df['valuation_acc'] >= AMOUNT)
        ].copy()
    else:
        df_filtered = df[
            (df['buy_sell'] == 0) &
            (df['valuation_acc'] >= AMOUNT)
        ].copy()

    return df_filtered

def extract_features(df):
    """
    Extract statistical features from the transaction data.
    """
    # Define time window (4-hour intervals)
    df['time_window'] = df['timestamp'].dt.floor('4H')

    # Group by time window
    grouped = df.groupby('time_window')

    # Initialize feature DataFrame
    feature_df = pd.DataFrame()
    feature_df['order_count'] = grouped.size()
    feature_df['mean_inter_arrival'] = grouped['timestamp'].apply(lambda x: x.diff().dt.total_seconds().mean())
    feature_df['std_inter_arrival'] = grouped['timestamp'].apply(lambda x: x.diff().dt.total_seconds().std())
    feature_df['hour'] = feature_df.index.hour
    feature_df['weekday'] = feature_df.index.weekday
    feature_df['min_inter_arrival'] = grouped['timestamp'].apply(lambda x: x.diff().dt.total_seconds().min())
    feature_df['max_inter_arrival'] = grouped['timestamp'].apply(lambda x: x.diff().dt.total_seconds().max())
    feature_df['median_inter_arrival'] = grouped['timestamp'].apply(lambda x: x.diff().dt.total_seconds().median())
    feature_df['skew_inter_arrival'] = grouped['timestamp'].apply(lambda x: x.diff().dt.total_seconds().skew())
    feature_df['kurtosis_inter_arrival'] = grouped['timestamp'].apply(lambda x: x.diff().dt.total_seconds().kurtosis())
    feature_df['total_time_span'] = grouped['timestamp'].apply(lambda x: (x.max() - x.min()).total_seconds())

    # Additional Features
    # Fee-related Features
    if 'implicitSwapFeeSy' in df.columns and 'explicitSwapFeeSy' in df.columns:
        feature_df['total_swap_fee_sy'] = grouped[['implicitSwapFeeSy', 'explicitSwapFeeSy']].sum().sum(axis=1)
        feature_df['avg_swap_fee_sy'] = grouped[['implicitSwapFeeSy', 'explicitSwapFeeSy']].mean().mean(axis=1)
    else:
        logging.warning("Fee-related columns not found. Skipping fee feature engineering.")
        feature_df['total_swap_fee_sy'] = 0
        feature_df['avg_swap_fee_sy'] = 0

    # User Interaction Features
    if 'user' in df.columns:
        feature_df['unique_users'] = grouped['user'].nunique()
    else:
        logging.warning("Column 'user' not found. Skipping user interaction features.")
        feature_df['unique_users'] = 0

    # Action-Based Features
    if 'action' in df.columns:
        feature_df['swap_pt_count'] = grouped['action'].apply(lambda x: (x == 'SWAP_PT').sum())
    else:
        logging.warning("Column 'action' not found. Skipping action-based features.")
        feature_df['swap_pt_count'] = 0

    # Market-Based Features
    if 'market_symbol' in df.columns and 'market_expiry' in df.columns:
        feature_df['unique_market_symbols'] = grouped['market_symbol'].nunique()
        feature_df['days_until_expiry'] = grouped['market_expiry'].apply(lambda x: (pd.to_datetime(x).max() - pd.Timestamp.utcnow()).days)
    else:
        logging.warning("Market-related columns not found. Skipping market-based features.")
        feature_df['unique_market_symbols'] = 0
        feature_df['days_until_expiry'] = 0

    # Valuation Features
    if 'valuation_usd' in df.columns:
        feature_df['total_valuation_usd'] = grouped['valuation_usd'].sum()
        feature_df['avg_valuation_usd'] = grouped['valuation_usd'].mean()
    else:
        logging.warning("Column 'valuation_usd' not found. Skipping valuation features.")
        feature_df['total_valuation_usd'] = 0
        feature_df['avg_valuation_usd'] = 0

    # Address-Type Features
    if 'input_baseType' in df.columns and 'output_baseType' in df.columns:
        feature_df['unique_input_baseType'] = grouped['input_baseType'].nunique()
        feature_df['unique_output_baseType'] = grouped['output_baseType'].nunique()
    else:
        logging.warning("Address-Type columns not found. Skipping address-type features.")
        feature_df['unique_input_baseType'] = 0
        feature_df['unique_output_baseType'] = 0

    # Handle missing values
    feature_df.fillna(method='ffill', inplace=True)
    feature_df.fillna(method='bfill', inplace=True)

    # Additional Feature Engineering
    feature_df['rolling_mean_inter_arrival'] = feature_df['mean_inter_arrival'].rolling(window=3).mean()
    feature_df['rolling_std_inter_arrival'] = feature_df['mean_inter_arrival'].rolling(window=3).std()
    feature_df['lag_order_count'] = feature_df['order_count'].shift(1)
    feature_df['lag_mean_inter_arrival'] = feature_df['mean_inter_arrival'].shift(1)

    # Fill new missing values after feature engineering
    feature_df.fillna(method='ffill', inplace=True)
    feature_df.fillna(method='bfill', inplace=True)

    return feature_df

def reduce_dimensionality(features_scaled, variance_threshold=0.95):
    """
    Apply PCA for dimensionality reduction to improve clustering performance.
    """
    pca = PCA(n_components=variance_threshold, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    logging.info(f"PCA reduced features to {features_pca.shape[1]} dimensions explaining {variance_threshold*100}% variance.")
    return features_pca, pca

def determine_optimal_k(features_scaled, k_min=2, k_max=10):
    """
    Determine the optimal number of clusters using Silhouette and Calinski-Harabasz methods.
    """
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    K_range = range(k_min, k_max + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features_scaled)
        inertias.append(kmeans.inertia_)

        if k > 1:
            sil_score = silhouette_score(features_scaled, labels)
            silhouette_scores.append(sil_score)
            calinski_score = calinski_harabasz_score(features_scaled, labels)
            calinski_scores.append(calinski_score)
            davies_score = davies_bouldin_score(features_scaled, labels)
            davies_scores.append(davies_score)
        else:
            silhouette_scores.append(None)
            calinski_scores.append(None)
            davies_scores.append(None)

    # Determine optimal K based on Silhouette Score
    optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
    logging.info(f"Optimal number of clusters determined: {optimal_k_silhouette} (Silhouette Score)")

    return optimal_k_silhouette

def fit_distributions_parallel(inter_arrival_times, distributions):
    """
    Fit distributions in parallel to speed up the process.
    """
    def fit_distribution(name, dist, data):
        try:
            params = dist.fit(data)
            expected_inter_arrival = dist.mean(*params)
            if not np.isfinite(expected_inter_arrival) or expected_inter_arrival <= 0:
                return None
            log_likelihood = np.sum(dist.logpdf(data, *params))
            k_params = len(params)
            bic = k_params * np.log(len(data)) - 2 * log_likelihood
            return (name, {'params': params, 'bic': bic})
        except Exception as e:
            logging.warning(f"Error fitting {name}: {e}")
            return None

    results = Parallel(n_jobs=-1)(
        delayed(fit_distribution)(name, dist, inter_arrival_times)
        for name, dist in distributions.items()
    )

    # Filter out failed fits
    fit_results = {name: info for result in results if result is not None for name, info in [result]}
    return fit_results

def plot_clusters(features_pca, clusters, title='Cluster Visualization with KMeans'):
    """
    Plot the clusters using the first two PCA components with Plotly for consistency.
    """
    # Create a DataFrame for Plotly
    plot_df = pd.DataFrame({
        'PCA1': features_pca[:, 0],
        'PCA2': features_pca[:, 1],
        'Cluster': clusters.astype(str)
    })

    # Define color palette
    unique_clusters = plot_df['Cluster'].unique()
    color_palette = px.colors.qualitative.Plotly
    color_map = {cluster: color_palette[i % len(color_palette)] for i, cluster in enumerate(unique_clusters)}

    # Create Plotly scatter plot
    fig = px.scatter(
        plot_df,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        color_discrete_map=color_map,
        title=title,
        labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2'},
        hover_data=['Cluster'],
        width=800,
        height=600
    )

    # Update layout for better aesthetics
    fig.update_layout(
        legend_title_text='Cluster',
        title_x=0.5,
        template='plotly_white'
    )

    # Instead of showing the plot, return the figure as JSON for API response if needed
    return fig.to_json()

def predict_next_order_time(current_time, last_order_time, best_dist, best_params):
    """
    Predict the next order arrival time based on the fitted distribution.
    """
    time_since_last_order = (current_time - last_order_time).total_seconds()
    expected_inter_arrival = best_dist.mean(*best_params)
    time_until_next_order = expected_inter_arrival - time_since_last_order

    if time_until_next_order < 0:
        time_until_next_order = expected_inter_arrival

    next_order_time = current_time + pd.Timedelta(seconds=time_until_next_order)
    return next_order_time, time_until_next_order

def waiting_time(df_tran_cleaned):
    """
    Main function to execute the order arrival time prediction pipeline.
    """
    # Validate User Selection
    try:
        BUY = validate_selection(
            buy_yt=BUY_YT,
            sell_yt=SELL_YT,
            buy_pt=BUY_PT,
            sell_pt=SELL_PT
        )
    except ValueError as ve:
        logging.error(ve)
        return {"error": str(ve)}

    # Data Preprocessing
    try:
        df_orders = preprocess_data(df_tran_cleaned, BUY)
    except KeyError as ke:
        logging.error(ke)
        return {"error": str(ke)}

    # Check Data Sufficiency
    MIN_TRANSACTIONS = 100
    MIN_ORDERS = 8

    if df_orders.empty:
        logging.info("No orders above the threshold.")
        return {"message": "No orders above the threshold."}
    elif len(df_tran_cleaned) < MIN_TRANSACTIONS:
        logging.info(f"Not enough transactions in the dataset. At least {MIN_TRANSACTIONS} required.")
        return {"message": f"Not enough transactions in the dataset. At least {MIN_TRANSACTIONS} required."}
    elif len(df_orders) < MIN_ORDERS:
        logging.info(f"Not enough orders above the threshold. At least {MIN_ORDERS} required.")
        return {"message": f"Not enough orders above the threshold. At least {MIN_ORDERS} required."}

    # Feature Extraction
    feature_df = extract_features(df_orders)

    # Save feature columns for consistency
    feature_columns = feature_df.columns.tolist()

    # Standardize Features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_df)

    # Dimensionality Reduction
    features_pca, pca = reduce_dimensionality(features_scaled, variance_threshold=0.95)

    # Determine Optimal Number of Clusters
    optimal_k = determine_optimal_k(features_pca, k_min=2, k_max=10)

    # Perform Clustering using KMeans
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(features_pca)
    feature_df['cluster'] = clusters

    # Cluster Visualization with Plotly
    cluster_plot_json = plot_clusters(features_pca, clusters, title='Cluster Visualization with KMeans')

    # Set 'time_window' as index for merging
    feature_df = feature_df.reset_index().set_index('time_window')

    # Merge Cluster Labels Back to Orders DataFrame
    df_orders = df_orders.merge(feature_df['cluster'], left_on='time_window', right_index=True, how='left')

    # Segment-wise Modeling
    segment_models = {}
    distributions = {
        'Exponential': expon,
        'Gamma': gamma,
        'Weibull': weibull_min,
        'Pareto': pareto,
        'Burr': burr,
        'LogNormal': lognorm,
        'Beta': beta
    }

    for cluster in feature_df['cluster'].unique():
        df_segment = df_orders[df_orders['cluster'] == cluster].copy()
        df_segment = df_segment.sort_values('timestamp')

        # Calculate inter-arrival times
        df_segment['inter_arrival_time'] = df_segment['timestamp'].diff().dt.total_seconds()
        df_segment = df_segment[df_segment['inter_arrival_time'] > 0].dropna(subset=['inter_arrival_time'])

        if len(df_segment) < MIN_ORDERS:
            logging.info(f"Cluster {cluster} has insufficient data. Skipping.")
            continue

        inter_arrival_times = df_segment['inter_arrival_time'].values

        # Fit distributions in parallel
        fit_results = fit_distributions_parallel(inter_arrival_times, distributions)

        # Select the best-fitting distribution based on BIC
        if fit_results:
            best_fit_name, best_fit_info = min(fit_results.items(), key=lambda x: x[1]['bic'])
            best_dist = distributions[best_fit_name]
            best_params = best_fit_info['params']
            best_bic = best_fit_info['bic']

            segment_models[cluster] = {
                'distribution_name': best_fit_name,
                'distribution': best_dist,
                'params': best_params,
                'bic': best_bic
            }
            logging.info(f"Cluster {cluster}: Best fit - {best_fit_name} with BIC={best_bic:.2f}")
        else:
            logging.info(f"Cluster {cluster}: No suitable distribution found.")

    # Prediction
    # Extract current time features
    current_time = pd.Timestamp.utcnow()
    current_features = {
        'order_count': feature_df['order_count'].mean(),
        'mean_inter_arrival': feature_df['mean_inter_arrival'].mean(),
        'std_inter_arrival': feature_df['std_inter_arrival'].mean(),
        'min_inter_arrival': feature_df['min_inter_arrival'].mean(),
        'max_inter_arrival': feature_df['max_inter_arrival'].mean(),
        'median_inter_arrival': feature_df['median_inter_arrival'].mean(),
        'skew_inter_arrival': feature_df['skew_inter_arrival'].mean(),
        'kurtosis_inter_arrival': feature_df['kurtosis_inter_arrival'].mean(),
        'total_time_span': feature_df['total_time_span'].mean(),
        'total_swap_fee_sy': feature_df.get('total_swap_fee_sy', 0).mean(),
        'avg_swap_fee_sy': feature_df.get('avg_swap_fee_sy', 0).mean(),
        'unique_users': feature_df.get('unique_users', 0).mean(),
        'swap_pt_count': feature_df.get('swap_pt_count', 0).mean(),
        'unique_market_symbols': feature_df.get('unique_market_symbols', 0).mean(),
        'days_until_expiry': feature_df.get('days_until_expiry', 0).mean(),
        'total_valuation_usd': feature_df.get('total_valuation_usd', 0).mean(),
        'avg_valuation_usd': feature_df.get('avg_valuation_usd', 0).mean(),
        'unique_input_baseType': feature_df.get('unique_input_baseType', 0).mean(),
        'unique_output_baseType': feature_df.get('unique_output_baseType', 0).mean(),
        'rolling_mean_inter_arrival': feature_df['rolling_mean_inter_arrival'].mean(),
        'rolling_std_inter_arrival': feature_df['rolling_std_inter_arrival'].mean(),
        'lag_order_count': feature_df['lag_order_count'].mean(),
        'lag_mean_inter_arrival': feature_df['lag_mean_inter_arrival'].mean(),
        'hour': current_time.hour,
        'weekday': current_time.weekday()
    }

    # Create DataFrame with consistent feature order
    current_features_df = pd.DataFrame([current_features], columns=feature_columns)

    # Standardize current features
    current_features_scaled = scaler.transform(current_features_df)

    # Apply PCA transformation
    current_features_pca = pca.transform(current_features_scaled)

    # Predict cluster for current time window using KMeans
    current_cluster = kmeans.predict(current_features_pca)[0]
    logging.info(f"Predicted cluster for current time window: {current_cluster}")
    
    if 'cluster' in df_orders.columns:
        df_orders_with_clusters = df_orders.to_dict(orient='records')
    else:
        df_orders_with_clusters = []
    if current_cluster in segment_models:
        model_info = segment_models[current_cluster]
        best_dist = model_info['distribution']
        best_params = model_info['params']

        # Get the last order time in the current cluster
        df_current_cluster = df_orders[df_orders['cluster'] == current_cluster].copy()
        if df_current_cluster.empty:
            logging.info("Not enough data in the current segment to make a prediction.")
            prediction_result = {"message": "Not enough data in the current segment to make a prediction."}
        else:
            last_order_time = df_current_cluster['timestamp'].max()
            next_order_time, time_until_next_order = predict_next_order_time(current_time, last_order_time, best_dist, best_params)

            # Prepare prediction results
            order_type = "buy" if BUY == 0 else "sell"
            prediction_result = {
                "predicted_next_order_time": str(next_order_time),
                "approximate_time_until_next_order": {
                    "seconds": round(time_until_next_order, 2),
                    "minutes": round(time_until_next_order / 60, 2),
                    "hours": round(time_until_next_order / 3600, 2),
                    "days": round(time_until_next_order / 86400, 2)
                },
                "order_type": order_type.capitalize(),
                "model_used": {
                    "cluster": int(current_cluster),
                    "distribution_name": model_info['distribution_name'],
                    "parameters": model_info['params'],
                    "bic": model_info['bic']
                },
                "categorized_transactions": df_orders_with_clusters
            }

            # Optionally, include detailed logging or other information
            logging.info("Prediction completed successfully.")

    else:
        logging.info("Not enough data in the current segment to make a prediction.")
        prediction_result = {"message": "Not enough data in the current segment to make a prediction."}

    return prediction_result
