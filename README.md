# **Wallet Risk Scoring API**

A Flask-based API that analyzes Ethereum wallet activity, generates behavioral features, uploads them to Kaggle, triggers a remote ML model, and returns real-time risk predictions.

---

## ⭐ Features

* 🔍 Fetches complete Ethereum transaction history (via **Etherscan API**)
* 🛠️ Generates 25+ engineered wallet behavior features
* ☁️ Automatically uploads data to **Kaggle Datasets**
* 🤖 Triggers a **Kaggle Notebook** containing your ML model
* 📊 Executes XGBoost inference remotely (no model hosting required)
* 📥 Downloads prediction results & returns them as JSON
* 🧱 Built for production: retries, polling, logging & error handling included

---

## 📂 Project Structure

```
.
│── app.py                         # Main Flask API
│── requirements.txt
│── .env                           # Environment variables
│── temp_data/                     # Auto-generated dataset folder
│── kaggle_notebook_push/          # Auto-generated notebook bundle
│── README.md
```

---

## 🧰 Tech Stack

* **Flask** (API server)
* **Python 3.10+**
* **Etherscan API**
* **Pandas / NumPy**
* **Kaggle Datasets API**
* **Kaggle Kernels API**
* **XGBoost model** (runs inside Kaggle)

---

## 🔧 Installation

### **1. Clone the repository**

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Configure Kaggle CLI**

```bash
pip install kaggle
```

Place your `kaggle.json` credentials in:

```
Windows: C:\Users\<User>\.kaggle\
Mac/Linux: ~/.kaggle/
```

Verify login:

```bash
kaggle datasets list
```

---

## 🔐 Environment Variables

Create a `.env` file:

```ini
ETHERSCAN_API_KEY=your_etherscan_key

# Must match your Kaggle dataset/kernel identifiers
KAGGLE_DATASET_SLUG=username/dataset-name
KAGGLE_NOTEBOOK_SLUG=username/kernel-name
KAGGLE_MODELS_DATASET_SLUG=username/model-artifacts

# Optional: manually define path to kaggle executable
KAGGLE_EXECUTABLE_PATH=C:\Python312\Scripts\kaggle.exe
```

Example:

```
KAGGLE_DATASET_SLUG=johndoe/eth-wallet-features
```

---

## ▶️ Running the API

Start the Flask server:

```bash
python app.py
```

API available at:

```
http://localhost:5000
```

---

## 📡 API Endpoint

### **POST /analyze**

Fetches TX history → generates features → uploads to Kaggle → runs ML model → returns prediction.

#### **Request**

```json
{
  "address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
}
```

#### **Response**

```json
{
  "address": "0x742d35Cc6634...",
  "prediction": 1,
  "confidence": 0.92
}
```

#### Prediction Meaning

| Label | Meaning                                    |
| ----- | ------------------------------------------ |
| **0** | Low-risk wallet                            |
| **1** | High-risk or suspicious                    |
| **2** | Anomalous behavior / depends on your model |

---

## 🧠 How It Works (Architecture)

```
Client ──► Flask API
         │
         ├── Fetch transactions (Etherscan)
         ├── Feature engineering (Pandas)
         ├── Upload dataset → Kaggle
         ├── Push kernel → Kaggle
         ├── Poll kernel status
         └── Download ML prediction
                 │
                 ▼
              JSON Output
```

This design eliminates the need to host the ML model on your server.

---

## 🚀 Feature Engineering Summary

The API computes more than **25 wallet behavior features**, including:

* total transactions
* sent/received counts
* gas usage patterns
* fee statistics
* average/max transfer values
* time-between-transactions
* partner diversity
* ratios of incoming vs outgoing ETH
* activity duration
* anomaly-friendly metrics

Output is stored as:

```
features.csv
```

---

## 🛡️ Error Handling

The API gracefully detects:

* ❌ Missing or invalid ETH address
* ❌ Etherscan API failures / rate limits
* ❌ Kaggle dataset upload failures
* ❌ Kernel run errors
* ❌ Missing prediction output
* ⏳ Timeouts (dataset → 120s, kernel → 300s)

Example error response:

```json
{
  "error": "Kaggle took too long to process the new data."
}
```

---

## 🧪 Example cURL Request

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d "{\"address\": \"0x742d35Cc6634C0532925a3b844Bc454e4438f44e\"}"
```

---

## 🐞 Troubleshooting

### Kaggle CLI not found

Set the executable manually:

```
KAGGLE_EXECUTABLE_PATH=C:\Python312\Scripts\kaggle.exe
```

### Kaggle stuck in "queued"

You may be out of Kaggle compute quota.

### Etherscan shows empty list

The wallet may have no transactions.

### `prediction_output.csv` missing

Notebook likely crashed — check Kaggle activity logs.

---

## 📈 Future Improvements

* Local fallback ML inference
* Multiple models (risk, anomaly, threat scoring)
* On-chain analytics dashboard
* Realtime Geth/Alchemy integration
* Caching frequently queried addresses

---

## 📝 License

MIT License. Free to use, modify, and distribute.

---
