# 📚 Pendle-YT-Timing-Strategy-Analyzer V6 API Usage Guide

Welcome to the **Pendle-YT-Timing-Strategy-Analyzer V6** API Usage Guide. This document is intended to help collaborators understand and effectively utilize the API for fetching data and predicting order arrival times based on transaction data.

## 🚀 Table of Contents

- [📚 Pendle-YT-Timing-Strategy-Analyzer V6 API Usage Guide](#-pendle-yt-timing-strategy-analyzer-v6-api-usage-guide)
  - [🚀 Table of Contents](#-table-of-contents)
  - [🔍 Prerequisites](#-prerequisites)
  - [📦 Installation](#-installation)
  - [🛠 Running the API Server](#-running-the-api-server)
  - [🗺 API Endpoints](#-api-endpoints)
    - [1. GET `/full_data`](#1-get-full_data)
    - [2. GET \& POST `/predict_order_time`](#2-get--post-predict_order_time)
  - [🔧 Example Usage](#-example-usage)
    - [Using `curl`](#using-curl)
      - [1. GET `/full_data`](#1-get-full_data-1)
      - [2. GET `/predict_order_time`](#2-get-predict_order_time)
      - [3. POST `/predict_order_time`](#3-post-predict_order_time)
    - [Using Python's `requests` Library](#using-pythons-requests-library)
      - [1. GET `/full_data`](#1-get-full_data-2)
      - [2. GET `/predict_order_time`](#2-get-predict_order_time-1)
      - [3. POST `/predict_order_time`](#3-post-predict_order_time-1)
  - [🛑 Error Handling](#-error-handling)

---

## 🔍 Prerequisites

Before interacting with the API, ensure you have the following:

- **Python 3.x** installed on your machine.
- **Git** installed for cloning the repository.
- Necessary Python libraries installed (as specified in `requirements.txt`, hahaha,just a joke, we use pipenv now).

## 📦 Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/labrinyang/Pendle-YT-Timing-Strategy-Analyzer.git
   ```

2. **Navigate to the Project Directory**

   ```bash
   cd Pendle-YT-Timing-Strategy-Analyzer
   ```

3. **Set Up a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install Required Python Libraries**

   ```bash
   pip install pipenv
   pipenv install --ignore-pipfile
   ```

## 🛠 Running the API Server

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

- **`--reload`**: Enables auto-reloading of the server upon code changes. Useful during development.

After running the above command, the server will be accessible at `http://0.0.0.0:8000`.

## 🗺 API Endpoints

### 1. GET `/full_data`

**Description**: Retrieves the complete set of processed data, including cleaned transactions, merged data, time ranges, and fair value curves.

**Method**: `GET`

**URL**: `/full_data`

**Parameters**: None

**Response Model**: `FullResponse`

**Response Structure**:

```json
{
  "df_cleaned_transactions": [
    {
      "timestamp": "2023-01-01T00:00:00Z",
      "input_baseType": "SY",
      "valuation_acc": 0.01
    },
    ...
  ],
  "df_merged": [
    {
      "timestamp": "2023-01-01T00:00:00Z",
      "underlyingApy": 0.03
    },
    ...
  ],
  "h_range": [0.0, 1.0, 2.0, ...],
  "fair_value_curve": [0.03, 0.06, 0.09, ...],
  "symbol": "YT-EBTC-26DEC2024",
  "network": "ethereum",
  "mode": "plotly_dark",
  "underlying_amount": 1
}
```

**Field Descriptions**:

- **`df_cleaned_transactions`**: List of cleaned transaction records.
- **`df_merged`**: List of merged data combining transactions with market data.
- **`h_range`**: List representing the hourly range from the start time to maturity, in hours.
- **`fair_value_curve`**: List representing the calculated fair value of YT over time.
- **`symbol`**: Symbol of the Yield Token.
- **`network`**: Blockchain network used (e.g., Ethereum).
- **`mode`**: Visualization mode (e.g., `plotly_dark`).
- **`underlying_amount`**: Amount of underlying assets involved.

### 2. GET & POST `/predict_order_time`

**Description**: Predicts the next order arrival time based on cleaned transaction data. This endpoint supports both `GET` and `POST` methods for flexibility.

**Methods**: `GET`, `POST`

**URL**: `/predict_order_time`

**Parameters**: None

**Response Model**: `CombinedResponse`

**Response Structure**:

```json
{
  "full_data": {
    "df_cleaned_transactions": [
      {
        "id": "1-0xa6741398ac890b8c035c3706b392d21cb04839fa3979c99c3feb6507dfefc6a8-0x0abe2dbbec7a1a5cfdc09f098e63177e6bd8b93726503ce033ea0263eac3a960-140",
        "chainId": 1,
        "txHash": "0x0abe2dbbec7a1a5cfdc09f098e63177e6bd8b93726503ce033ea0263eac3a960",
        "blockNumber": 20957811,
        "timestamp": "2024-10-13T16:21:59Z",
        "action": "SWAP_PT",
        "origin": "PENDLE_MARKET",
        "inputs": "[{'asset': {'id': '1-0xb997b3418935a1df0f914ee901ec83927c1509a0', 'chainId': 1, 'address': '0xb997b3418935a1df0f914ee901ec83927c1509a0', 'symbol': 'PT-EBTC-26DEC2024', 'decimals': 8, 'expiry': '2024-12-26T00:00:00.000Z', 'accentColor': '', 'price': {'usd': 61482.96766103765}, 'priceUpdatedAt': '2024-10-13T16:38:00.000Z', 'name': 'PT ether.fi eBTC 26DEC2024', 'baseType': 'PT', 'types': ['PT'], 'protocol': 'Ether.fi (Bitcoin LRT)', 'underlyingPool': '', 'proSymbol': 'PT eBTC', 'proIcon': 'https://storage.googleapis.com/prod-pendle-bucket-a/images/uploads/685e07f1-e50d-4ff9-bac2-158611f718a6.svg', 'zappable': False, 'simpleName': 'PT ether.fi eBTC 26DEC2024', 'simpleSymbol': 'PT-EBTC-26DEC2024', 'simpleIcon': 'https://storage.googleapis.com/pendle-assets-staging/images/assets/unknown.svg', 'proName': 'PT ether.fi eBTC 26DEC2024'}, 'amount': 0.14386774, 'price': {'usd': 61472.58840770664, 'acc': 0.9881021275716418}}]",
        "outputs": "[{'asset': {'id': '1-0x7acdf2012aac69d70b86677fe91eb66e08961880', 'chainId': 1, 'address': '0x7acdf2012aac69d70b86677fe91eb66e08961880', 'symbol': 'SY-EBTC', 'decimals': 8, 'expiry': None, 'accentColor': '', 'price': {'usd': 62223.18100113741}, 'priceUpdatedAt': '2024-10-13T16:38:00.000Z', 'name': 'SY ether.fi eBTC', 'baseType': 'SY', 'types': ['SY'], 'protocol': 'Ether.fi (Bitcoin LRT)', 'underlyingPool': '', 'proSymbol': 'eBTC', 'proIcon': 'https://storage.googleapis.com/prod-pendle-bucket-a/images/uploads/cccfb931-8fef-4bb4-815b-9cc52517893e.svg', 'zappable': False, 'simpleName': 'SY ether.fi eBTC', 'simpleSymbol': 'SY-EBTC', 'simpleIcon': 'https://storage.googleapis.com/pendle-assets-staging/images/assets/unknown.svg', 'proName': 'SY ether.fi eBTC'}, 'amount': 0.14209897, 'price': {'usd': 62212.78822542521, 'acc': 1}}]",
        "user": "0x7acdf2012aac69d70b86677fe91eb66e08961880",
        "implicitSwapFeeSy": 0,
        "explicitSwapFeeSy": 0.00005704,
        "impliedApy": 0.0613975357302738,
        "gasUsed": 391874,
        "market_chainId": 1,
        "market_address": "0x36d3ca43ae7939645c306e26603ce16e39a89192",
        "market_symbol": "PENDLE-LPT",
        "market_expiry": "2024-12-26T00:00:00.000Z",
        "market_name": "PENDLE-LPT",
        "input_address": "0xb997b3418935a1df0f914ee901ec83927c1509a0",
        "input_baseType": "PT",
        "output_address": "0x7acdf2012aac69d70b86677fe91eb66e08961880",
        "output_baseType": "SY",
        "valuation_usd": 8843.92236616695,
        "valuation_acc": 0.142156019982924
      },
      ...
    ],
    "df_merged": [
      {
        "timestamp": "2023-01-01T00:00:00Z",
        "underlyingApy": 0.03
      },
      ...
    ],
    "h_range": [0.0, 1.0, 2.0, ...],
    "fair_value_curve": [0.03, 0.06, 0.09, ...],
    "symbol": "YT-EBTC-26DEC2024",
    "network": "ethereum",
    "mode": "plotly_dark",
    "underlying_amount": 1
  },
  "prediction": {
    "predicted_next_order_time": "2024-12-27T02:30:00+00:00",
    "approximate_time_until_next_order": {
      "seconds": 9000.0,
      "minutes": 150.0,
      "hours": 2.5,
      "days": 0.10416666666666667
    },
    "order_type": "Sell",
    "model_used": {
      "cluster": 1,
      "distribution_name": "Gamma",
      "parameters": [2.0, 1.5],
      "bic": 1234.56
    },
    "categorized_transactions": [
      {
        "timestamp": "2023-01-01T00:00:00Z",
        "input_baseType": "SY",
        "valuation_acc": 0.01,
        "cluster": 1
      },
      ...
    ],
    "message": null,
    "error": null
  }
}
```

**Field Descriptions**:

- **`full_data`**: Contains all the data returned by the `/full_data` endpoint.
- **`prediction`**:
  - **`predicted_next_order_time`**: ISO 8601 timestamp indicating when the next order is expected to be filled.
  - **`approximate_time_until_next_order`**: Breakdown of the estimated wait time until the next order is filled in seconds, minutes, hours, and days.
  - **`order_type`**: Indicates whether the predicted action is a "Buy" or "Sell".
  - **`model_used`**:
    - **`cluster`**: Cluster number used in the prediction model.
    - **`distribution_name`**: Name of the statistical distribution fitted to the data.
    - **`parameters`**: Parameters of the fitted distribution.
    - **`bic`**: Bayesian Information Criterion value for the fitted model.
  - **`categorized_transactions`**: List of transactions with their assigned cluster labels.
  - **`message`**: Informational message (if any).
  - **`error`**: Error message (if any).

## 🔧 Example Usage

### Using `curl`

#### 1. GET `/full_data`

```bash
curl -X GET "http://localhost:8000/full_data" -H "accept: application/json"
```

#### 2. GET `/predict_order_time`

```bash
curl -X GET "http://localhost:8000/predict_order_time" -H "accept: application/json"
```

#### 3. POST `/predict_order_time`

```bash
curl -X POST "http://localhost:8000/predict_order_time" -H "accept: application/json"
```

### Using Python's `requests` Library

First, ensure you have the `requests` library installed:

```bash
pip install requests
```

#### 1. GET `/full_data`

```python
import requests

response = requests.get("http://localhost:8000/full_data")
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Error: {response.status_code} - {response.text}")
```

#### 2. GET `/predict_order_time`

```python
import requests

response = requests.get("http://localhost:8000/predict_order_time")
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Error: {response.status_code} - {response.text}")
```

#### 3. POST `/predict_order_time`

```python
import requests

response = requests.post("http://localhost:8000/predict_order_time")
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Error: {response.status_code} - {response.text}")
```

## 🛑 Error Handling

The API utilizes HTTP status codes to indicate the success or failure of requests:

- **`200 OK`**: The request was successful.
- **`404 Not Found`**: Data not found or invalid asset details.
- **`500 Internal Server Error`**: An unexpected error occurred on the server.

**Response with Error Example**:

```json
{
  "full_data": null,
  "prediction": {
    "predicted_next_order_time": null,
    "approximate_time_until_next_order": null,
    "order_type": null,
    "model_used": null,
    "categorized_transactions": null,
    "message": null,
    "error": "Error retrieving asset details: Invalid contract address."
  }
}
```