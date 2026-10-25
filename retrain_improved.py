"""
StockAI — Improved LSTM Training Script v2 (FIXED)
Fixes:
  - auto_adjust=False — uses raw actual traded prices (no split distortion)
  - Consistent with app.py prediction code
  - All 17 stocks (TATAMOTORS removed)

Improvements over v1:
  - Added Volume as 5th feature
  - Added technical indicators: RSI, EMA_10, EMA_20 (8 features total)
  - Deeper model: LSTM(128) → LSTM(64) → LSTM(32) → Dense(32) → Dense(1)
  - Dropout(0.2) layers to prevent overfitting
  - 100 epochs with EarlyStopping (stops when no improvement)
  - ReduceLROnPlateau — learning rate adapts automatically
  - 45-day window (better for volatile Indian stocks)
  - 5 years of training data
  - Batch size 16 for better gradient updates
  - Saves best model only (not last epoch)
  - Huber loss — less sensitive to outliers than MSE

Run: python retrain_improved.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ───────────────────────────────────────────────
WINDOW_SIZE  = 45       # days of history to look back
EPOCHS       = 100      # max epochs (EarlyStopping will stop early)
BATCH_SIZE   = 16
TEST_SPLIT   = 0.10     # 10% for validation
YEARS_DATA   = 5        # years of training data
# ─────────────────────────────────────────────────────────

# 17 stocks — TATAMOTORS removed
STOCKS = {
    "TCS.NS":        "tcs",
    "INFY.NS":       "infy",
    "RELIANCE.NS":   "reliance",
    "HDFCBANK.NS":   "hdfcbank",
    "WIPRO.NS":      "wipro",
    "ICICIBANK.NS":  "icicibank",
    "SBIN.NS":       "sbin",
    "BAJFINANCE.NS": "bajfinance",
    "ADANIENT.NS":   "adanient",
    "AAPL":          "aapl",
    "GOOGL":         "googl",
    "MSFT":          "msft",
    "AMZN":          "amzn",
    "TSLA":          "tsla",
    "META":          "meta",
    "NVDA":          "nvda",
    "NFLX":          "nflx",
}


def compute_rsi(series, period=14):
    """Compute RSI indicator."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs  = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def prepare_features(data):
    """
    Build 8-feature dataframe:
    Open, High, Low, Close, Volume, RSI, EMA10, EMA20
    """
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['RSI']   = compute_rsi(df['Close'])
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df.dropna(inplace=True)
    return df


def build_model(n_features):
    """Improved LSTM architecture with dropout."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(WINDOW_SIZE, n_features)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    return model


def train_stock(ticker, name):
    print(f"\n{'='*55}")
    print(f"  Training: {ticker} ({name})")
    print(f"{'='*55}")

    # ── 1. Download data (auto_adjust=False = raw prices) ──
    try:
        end   = pd.Timestamp.today()
        start = end - pd.DateOffset(years=YEARS_DATA)
        data  = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False   # ✅ FIXED — raw actual traded prices
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty:
            print(f"  ❌ No data returned. Skipping.")
            return False
        if len(data) < WINDOW_SIZE + 50:
            print(f"  ❌ Not enough data ({len(data)} rows). Skipping.")
            return False
        print(f"  ✓ Downloaded {len(data)} rows ({start.date()} → {end.date()})")
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return False

    # ── 2. Feature engineering ──
    try:
        df = prepare_features(data)
        if len(df) < WINDOW_SIZE + 20:
            print(f"  ❌ Not enough rows after feature calc ({len(df)}). Skipping.")
            return False
        print(f"  ✓ Features: {list(df.columns)} | Rows: {len(df)}")
        n_features = df.shape[1]
    except Exception as e:
        print(f"  ❌ Feature engineering failed: {e}")
        return False

    # ── 3. Scale ──
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values)

    # ── 4. Build sequences ──
    X, y      = [], []
    close_idx = df.columns.get_loc('Close')
    for i in range(WINDOW_SIZE, len(scaled)):
        X.append(scaled[i - WINDOW_SIZE:i])
        y.append(scaled[i, close_idx])
    X, y = np.array(X), np.array(y)

    split   = int(len(X) * (1 - TEST_SPLIT))
    X_train = X[:split];  X_val = X[split:]
    y_train = y[:split];  y_val = y[split:]
    print(f"  ✓ Train: {len(X_train)} | Val: {len(X_val)} sequences")

    # ── 5. Build model ──
    model = build_model(n_features)

    # ── 6. Callbacks ──
    model_path = f"{name}_lstm_model.h5"
    callbacks  = [
        EarlyStopping(
            monitor='val_loss', patience=12,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            model_path, monitor='val_loss',
            save_best_only=True, verbose=0
        )
    ]

    # ── 7. Train ──
    print(f"  ⏳ Training (max {EPOCHS} epochs, early stopping enabled)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0
    )

    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_loss  = min(history.history['val_loss'])
    best_mae   = history.history['val_mae'][best_epoch - 1]
    print(f"  ✓ Best epoch: {best_epoch} | Val Loss: {best_loss:.6f} | Val MAE: {best_mae:.6f}")

    # ── 8. Accuracy estimate on validation set ──
    try:
        preds_scaled = model.predict(X_val, verbose=0).flatten()

        dummy              = np.zeros((len(preds_scaled), n_features))
        dummy[:, close_idx] = preds_scaled
        pred_prices        = scaler.inverse_transform(dummy)[:, close_idx]

        dummy2              = np.zeros((len(y_val), n_features))
        dummy2[:, close_idx] = y_val
        actual_prices       = scaler.inverse_transform(dummy2)[:, close_idx]

        pct_errors = np.abs(pred_prices - actual_prices) / actual_prices * 100
        accuracy   = float(np.mean(100 - pct_errors))
        mae        = float(np.mean(np.abs(pred_prices - actual_prices)))
        print(f"  ✓ Estimated accuracy : {accuracy:.2f}%")
        print(f"  ✓ Mean Abs Error (₹) : {mae:.2f}")
    except Exception as e:
        print(f"  ⚠ Could not estimate accuracy: {e}")

    # ── 9. Save scaler ──
    scaler_path = f"{name}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Saved: {model_path} + {scaler_path}")

    return True


def main():
    print("\n" + "="*55)
    print("  🚀 StockAI LSTM Trainer v2 (FIXED)")
    print(f"  Window : {WINDOW_SIZE} days | Epochs : {EPOCHS} max")
    print(f"  Data   : {YEARS_DATA} years | Features : 8")
    print(f"  Model  : LSTM(128)→LSTM(64)→LSTM(32)→Dense(32)→Dense(1)")
    print(f"  Loss   : Huber | auto_adjust : False ✅")
    print("="*55 + "\n")

    results = {}
    for ticker, name in STOCKS.items():
        success         = train_stock(ticker, name)
        results[ticker] = '✅' if success else '❌'

    print("\n" + "="*55)
    print("  TRAINING SUMMARY")
    print("="*55)
    for ticker, status in results.items():
        print(f"  {status}  {ticker}")
    success_count = sum(1 for v in results.values() if v == '✅')
    print(f"\n  {success_count}/{len(STOCKS)} models trained successfully")
    print("="*55)
    print("\n✅ Done! Copy the new .h5 and .pkl files to your project folder.")
    print("   These models now use raw prices — consistent with app.py\n")


if __name__ == "__main__":
    main()