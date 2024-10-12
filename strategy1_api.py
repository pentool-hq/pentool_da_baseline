from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scripts.asset_retriever import AssetRetriever
from scripts.config import load_config
from scripts.data_acquisition import DataAcquisition, clean_transaction_data
from scripts.yt_calculation import YTCalculation

app = FastAPI()

config = load_config()

class FullResponse(BaseModel):
    df_merged: List[Dict[str, Any]]
    h_range: List[float]
    fair_value_curve: List[float]
    symbol: str
    network: str
    mode: str
    underlying_amount: float

def convert_timestamps_to_hours(timestamps: pd.DatetimeIndex, maturity: pd.Timestamp) -> List[float]:
    return [(maturity - ts).total_seconds() / 3600 for ts in timestamps]

def main_logic():
    # Load configuration
    network = config['network']
    yt_contract = config['yt_contract']
    market_contract = config['market_contract']
    start_time = config['start_time']
    points_per_hour_per_underlying = config['points_per_hour_per_underlying']
    underlying_amount = config['underlying_amount']
    pendle_multiplier = config['pendle_multiplier']
    mode = 'plotly_dark' if config['dark_mode'] else 'plotly_white'

    # Step 1: Retrieve asset information
    asset_retriever = AssetRetriever(network, 'YT', yt_contract)
    try:
        symbol, maturity = asset_retriever.get_asset_details()
    except ValueError as e:
        print(f"Error in retrieving asset details: {e}")
        raise HTTPException(status_code=404, detail=f"Error retrieving asset details: {e}")

    # Convert maturity to pandas Timestamp if it's a string
    if isinstance(maturity, str):
        maturity = pd.to_datetime(maturity)
    
    # Ensure maturity is timezone-aware (UTC)
    if maturity.tzinfo is None:
        maturity = maturity.tz_localize('UTC')

    # Step 2: Fetch market and transaction data
    data_acquisition = DataAcquisition(market_contract, yt_contract, start_time, network)
    df_combined, df_transactions = data_acquisition.run()

    if df_combined.empty or df_transactions.empty:
        raise HTTPException(status_code=404, detail="No data fetched from the API.")

    # Step 3: Clean transaction data
    df_cleaned_transactions = clean_transaction_data(df_transactions)

    if df_cleaned_transactions.empty:
        print("No valid transactions after cleaning.")
        raise HTTPException(status_code=404, detail="No valid transactions after cleaning.")

    # Step 4: Merge cleaned transaction data with combined data
    df_cleaned_transactions['timestamp'] = pd.to_datetime(df_cleaned_transactions['timestamp'], utc=True)
    df_combined['timestamp'] = pd.to_datetime(df_combined['Time'], utc=True)

    df_merged = pd.merge_asof(
        df_cleaned_transactions.sort_values('timestamp'),
        df_combined[['timestamp', 'underlyingApy']],
        on='timestamp',
        direction='backward'
    )

    # Step 5: Perform YT calculations
    calculation = YTCalculation(
        df_merged, df_combined, maturity,
        points_per_hour_per_underlying, underlying_amount, pendle_multiplier
    )
    df_merged, df_combined, h_range, fair_value_curve, weighted_points = calculation.run_calculations()


    h_range = pd.date_range(start=df_merged['timestamp'].iloc[0], end=maturity, freq='H')
    h_range_hours = convert_timestamps_to_hours(h_range, maturity)


    # Step 6: Return the results
    return {
        "df_merged": df_merged.to_dict(orient="records"),
        "h_range": h_range_hours,
        "fair_value_curve": fair_value_curve,
        "symbol": symbol,
        "network": network,
        "mode": mode,
        "underlying_amount": underlying_amount
    }

@app.get("/full_data", response_model=FullResponse)
def get_full_data():
    result = main_logic()
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
