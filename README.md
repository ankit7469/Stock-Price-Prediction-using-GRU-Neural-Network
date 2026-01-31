# Stock Price Prediction using GRU Neural Network
-------------------------------------------------------------------

This project predicts the **stock closing price** using a **GRU (Gated Recurrent Unit)** deep learning model. The model analyzes **past stock prices**, learns their patterns, and predicts future values.

-----

##  Project Overview

This is a **Time-Series Forecasting** project built using:

* Python
* GRU Neural Networks
* TensorFlow/Keras
* NumPy & Pandas
* Data Scaling (MinMaxScaler)

The model uses the last **N past days (window size)** of stock prices to predict the **next day price**.

---

##  Project Objective

* Understand and implement time-series forecasting
* Analyze past stock values
* Train a GRU model to learn stock price patterns
* Compare predicted values with actual values
* Build a model capable of future prediction

---

##  Dataset

You can download stock data using:

```python
import yfinance as yf

# Example: Reliance Industries (NSE)
df = yf.download('RELIANCE.NS', start='2010-01-01', end='2025-01-01')
```

Dataset columns include:

* Open
* High
* Low
* Close  (Used for prediction)
* Volume

---

##  Technologies Used

* **Python**
* **TensorFlow / Keras**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **scikit-learn** (MinMaxScaler)

---

##  Model Architecture

The model consists of:

* GRU Layer (64 units)
* Dropout (20%)
* GRU Layer (32 units)
* Dropout (20%)
* Dense Output Layer (1 neuron)

---

##  Code Workflow

### ✔ 1. Load the dataset

### ✔ 2. Use only the 'Close' price

### ✔ 3. Normalize using MinMaxScaler

### ✔ 4. Create sequences (windowing)

### ✔ 5. Reshape data for GRU (samples, timesteps, features)

### ✔ 6. Build GRU model and compile 

### ✔ 7. Train the model

### ✔ 8. Predict and inverse transform results

### ✔ 9. Plot real vs predicted values

---

##  Training & Prediction Graph

The project visualizes:

* **Real Stock Prices** (Actual values)
* **Predicted Stock Prices** (Model output)

This helps evaluate how well the GRU model learned the pattern.

---

##  Folder Structure

```
├── GRU_Stock_Prediction.ipynb
├── reliance.csv (optional)
├── README.md
└── plots/
    └── prediction_graph.png
```

---

##  Key Learning Outcomes

* Time-series preparation techniques
* GRU working and gating mechanism
* Scaling and inverse-scaling data
* Sequence modeling
* Evaluating deep learning models visually


##  License

This project is open-source and free to use.

---

##  Author

** Ankit Kashyap"

Feel free to connect or ask questions!
