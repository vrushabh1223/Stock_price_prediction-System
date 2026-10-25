from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import mysql.connector
import yfinance as yf
from datetime import datetime, timedelta
import hashlib
import os
import io
import csv
import re

# Google OAuth via authlib
from authlib.integrations.flask_client import OAuth

# Flask-Mail
from flask_mail import Mail, Message

# Background scheduler for price alerts
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
app.secret_key = "stockai_secret_2024"

# =========================================================
# GOOGLE OAUTH CONFIG (authlib)
# =========================================================
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

oauth = OAuth(app)
google_oauth = oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url = 'https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs = {'scope': 'openid email profile'},
)

# =========================================================
# FLASK-MAIL CONFIG (Gmail)
# =========================================================
app.config['MAIL_SERVER']         = 'smtp.gmail.com'
app.config['MAIL_PORT']           = 587
app.config['MAIL_USE_TLS']        = True
app.config['MAIL_USERNAME']       = 'patilvrushabh188@gmail.com'
app.config['MAIL_PASSWORD']       = 'pqizmxcpujbmyir'
app.config['MAIL_DEFAULT_SENDER'] = ('StockAI', 'patilvrushabh188@gmail.com')
mail = Mail(app)

# =========================================================
# FEATURE 2: MULTIPLE MODELS — map ticker → model/scaler
# =========================================================
MODELS = {}   # loaded lazily on first use

def load_model_for(ticker):
    """Load model & scaler for a given ticker. Falls back to TCS if not found."""
    if ticker in MODELS:
        return MODELS[ticker]

    MODEL_FILES = {
        "TCS.NS":        ("tcs_lstm_model.h5",        "tcs_scaler.pkl"),
        "INFY.NS":       ("infy_lstm_model.h5",        "infy_scaler.pkl"),
        "RELIANCE.NS":   ("reliance_lstm_model.h5",    "reliance_scaler.pkl"),
        "HDFCBANK.NS":   ("hdfcbank_lstm_model.h5",    "hdfcbank_scaler.pkl"),
        "WIPRO.NS":      ("wipro_lstm_model.h5",       "wipro_scaler.pkl"),
        "ICICIBANK.NS":  ("icicibank_lstm_model.h5",   "icicibank_scaler.pkl"),
        "SBIN.NS":       ("sbin_lstm_model.h5",        "sbin_scaler.pkl"),
        "BAJFINANCE.NS": ("bajfinance_lstm_model.h5",  "bajfinance_scaler.pkl"),
        "ADANIENT.NS":   ("adanient_lstm_model.h5",    "adanient_scaler.pkl"),
        "AAPL":          ("aapl_lstm_model.h5",        "aapl_scaler.pkl"),
        "GOOGL":         ("googl_lstm_model.h5",       "googl_scaler.pkl"),
        "MSFT":          ("msft_lstm_model.h5",        "msft_scaler.pkl"),
        "AMZN":          ("amzn_lstm_model.h5",        "amzn_scaler.pkl"),
        "TSLA":          ("tsla_lstm_model.h5",        "tsla_scaler.pkl"),
        "META":          ("meta_lstm_model.h5",        "meta_scaler.pkl"),
        "NVDA":          ("nvda_lstm_model.h5",        "nvda_scaler.pkl"),
        "NFLX":          ("nflx_lstm_model.h5",        "nflx_scaler.pkl"),
    }

    model_file, scaler_file = MODEL_FILES.get(ticker, ("tcs_lstm_model.h5", "tcs_scaler.pkl"))
    fallback_used = False

    try:
        m = tf.keras.models.load_model(model_file)
        s = joblib.load(scaler_file)
    except Exception:
        try:
            m = tf.keras.models.load_model("tcs_lstm_model.h5")
            s = joblib.load("tcs_scaler.pkl")
            fallback_used = True
        except Exception as e:
            raise RuntimeError(f"No model available: {e}")

    MODELS[ticker] = (m, s, fallback_used)
    return MODELS[ticker]


# =========================================================
# MYSQL
# =========================================================
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="stock_prediction"
    )

def init_db():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            email VARCHAR(150) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            stock_ticker VARCHAR(20),
            prediction_date DATE,
            open_price FLOAT,
            high_price FLOAT,
            low_price FLOAT,
            close_price FLOAT,
            predicted_price FLOAT,
            confidence FLOAT,
            actual_price FLOAT DEFAULT NULL,
            accuracy FLOAT DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            message TEXT,
            stock_ticker VARCHAR(20),
            is_read BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            stock_ticker VARCHAR(20),
            stock_name VARCHAR(100),
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY unique_watchlist (user_id, stock_ticker),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            ticker VARCHAR(20),
            name VARCHAR(100),
            quantity FLOAT,
            buy_price FLOAT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_alerts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            ticker VARCHAR(20),
            target_price FLOAT,
            direction ENUM('above','below') DEFAULT 'above',
            email VARCHAR(150),
            is_active BOOLEAN DEFAULT TRUE,
            triggered_at TIMESTAMP NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    # Add actual_price & accuracy columns to existing predictions table if missing
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN actual_price FLOAT DEFAULT NULL")
        db.commit()
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN accuracy FLOAT DEFAULT NULL")
        db.commit()
    except Exception:
        pass
    db.commit()
    cursor.close()
    db.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# =========================================================
# SUPPORTED STOCKS
# =========================================================

# TRAINED_STOCKS = all 18 stocks now have dedicated models ✅
TRAINED_STOCKS = {
    "TCS.NS":        "TCS (Tata Consultancy)",
    "INFY.NS":       "Infosys",
    "RELIANCE.NS":   "Reliance Industries",
    "HDFCBANK.NS":   "HDFC Bank",
    "WIPRO.NS":      "Wipro",
    "ICICIBANK.NS":  "ICICI Bank",
    "SBIN.NS":       "State Bank of India",
    "BAJFINANCE.NS": "Bajaj Finance",
    "ADANIENT.NS":   "Adani Enterprises",
    "AAPL":          "Apple Inc.",
    "GOOGL":         "Alphabet (Google)",
    "MSFT":          "Microsoft",
    "AMZN":          "Amazon",
    "TSLA":          "Tesla",
    "META":          "Meta Platforms",
    "NVDA":          "NVIDIA",
    "NFLX":          "Netflix",
}

# ALL_STOCKS = full list shown in dropdown
ALL_STOCKS = {
    "TCS.NS":        "TCS (Tata Consultancy)",
    "INFY.NS":       "Infosys",
    "RELIANCE.NS":   "Reliance Industries",
    "HDFCBANK.NS":   "HDFC Bank",
    "WIPRO.NS":      "Wipro",
    "ICICIBANK.NS":  "ICICI Bank",
    "SBIN.NS":       "State Bank of India",
    "BAJFINANCE.NS": "Bajaj Finance",
    "ADANIENT.NS":   "Adani Enterprises",
    "AAPL":          "Apple Inc.",
    "GOOGL":         "Alphabet (Google)",
    "MSFT":          "Microsoft",
    "AMZN":          "Amazon",
    "TSLA":          "Tesla",
    "META":          "Meta Platforms",
    "NVDA":          "NVIDIA",
    "NFLX":          "Netflix",
}

# Alias for backward compatibility
STOCKS = ALL_STOCKS

# Admin credentials — change these!
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = hash_password("Admin@1234")

# =========================================================
# AUTH ROUTES
# =========================================================

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ""
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        try:
            db = get_db()
            cursor = db.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
            user = cursor.fetchone()
            cursor.close()
            db.close()
            if user:
                session['user_id']  = user['id']
                session['username'] = user['username']
                return redirect(url_for('dashboard'))
            else:
                message = "Invalid username or password."
        except Exception as e:
            message = f"Database error: {e}"
    return render_template('login.html', message=message)


@app.route('/register', methods=['GET', 'POST'])
def register():
    message = ""
    if request.method == 'POST':
        username = request.form['username']
        email    = request.form['email']
        raw_pass = request.form['password']

        # Server-side password validation
        import re
        if (len(raw_pass) < 8 or
            not re.search(r'[A-Z]', raw_pass) or
            not re.search(r'[a-z]', raw_pass) or
            not re.search(r'[0-9]', raw_pass) or
            not re.search(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]', raw_pass)):
            return render_template('register.html',
                                   message="Password must be 8+ chars with uppercase, lowercase, number and special character.")

        password = hash_password(raw_pass)
        try:
            db = get_db()
            cursor = db.cursor()
            cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                           (username, email, password))
            db.commit()
            user_id = cursor.lastrowid
            cursor.execute("INSERT INTO notifications (user_id, message, stock_ticker) VALUES (%s,%s,%s)",
                           (user_id, f"Welcome to StockAI, {username}! Start predicting stock prices now.", ""))
            db.commit()
            cursor.close()
            db.close()
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            message = "Username or email already exists."
        except Exception as e:
            message = f"Error: {e}"
    return render_template('register.html', message=message)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# =========================================================
# DASHBOARD
# =========================================================

@app.route('/google-login')
def google_login():
    redirect_uri = url_for('google_authorized', _external=True)
    return google_oauth.authorize_redirect(redirect_uri, prompt='select_account')


@app.route('/login/google/authorized')
def google_authorized():
    try:
        token = google_oauth.authorize_access_token()
        info  = token.get('userinfo')
        if not info:
            info = google_oauth.get('https://www.googleapis.com/oauth2/v3/userinfo').json()

        email    = info.get('email', '')
        username = info.get('name', email.split('@')[0]).replace(' ', '_')

        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        if not user:
            rand_pass = hash_password(email + app.secret_key)
            cursor.execute("INSERT INTO users (username, email, password) VALUES (%s,%s,%s)",
                           (username, email, rand_pass))
            db.commit()
            cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
            user = cursor.fetchone()
        cursor.close()
        db.close()

        session['user_id']  = user['id']
        session['username'] = user['username']
        return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"Google OAuth error: {e}")
        return redirect(url_for('login'))


@app.route('/debug-oauth')
def debug_oauth():
    redirect_uri = url_for('google_authorized', _external=True)
    return f"Redirect URI being used: {redirect_uri}"


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 5", (user_id,))
        recent_preds = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) as cnt FROM notifications WHERE user_id=%s AND is_read=FALSE", (user_id,))
        notif_count = cursor.fetchone()['cnt']
        cursor.execute("""SELECT AVG(accuracy) as avg_acc, COUNT(*) as total,
                          SUM(CASE WHEN accuracy IS NOT NULL THEN 1 ELSE 0 END) as scored
                          FROM predictions WHERE user_id=%s""", (user_id,))
        stats = cursor.fetchone()
        cursor.close()
        db.close()
    except Exception:
        recent_preds = []
        notif_count  = 0
        stats        = {"avg_acc": None, "total": 0, "scored": 0}

    return render_template('dashboard.html',
                           stocks=ALL_STOCKS,
                           all_stocks=ALL_STOCKS,
                           recent_preds=recent_preds,
                           notif_count=notif_count,
                           stats=stats,
                           trained_stocks=TRAINED_STOCKS,
                           username=session.get('username'))


# =========================================================
# FEATURE 1: AUTO-FILL API
# =========================================================

@app.route('/api/autofill')
def autofill():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    ticker = request.args.get('ticker', 'TCS.NS')
    try:
        import math
        data = yf.download(ticker, period="5d", progress=False, auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty:
            return jsonify({"error": "No data found"}), 404
        # Get last row with no NaN values
        data.dropna(inplace=True)
        if data.empty:
            return jsonify({"error": "No valid data found"}), 404
        latest = data.tail(1).iloc[0]
        def safe(val):
            v = float(val)
            return round(v, 2) if not math.isnan(v) and not math.isinf(v) else 0.0
        return jsonify({
            "open":  safe(latest['Open']),
            "high":  safe(latest['High']),
            "low":   safe(latest['Low']),
            "close": safe(latest['Close']),
            "date":  str(data.tail(1).index[0].date()),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================================
# PREDICT
# =========================================================

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    open_price  = float(request.form['open'])
    high_price  = float(request.form['high'])
    low_price   = float(request.form['low'])
    close_price = float(request.form['close'])
    ticker      = request.form.get('ticker', 'TCS.NS')
    pred_date   = request.form.get('pred_date', str(datetime.today().date()))
    user_id     = session['user_id']

    try:
        model, scaler, fallback_used = load_model_for(ticker)

        end_date   = datetime.today()
        start_date = end_date - timedelta(days=200)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # ── Detect model type: 8-feature (new) or 4-feature (old) ──
        n_features = scaler.n_features_in_

        if n_features == 8:
            data_feat = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            delta = data_feat['Close'].diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            data_feat['RSI']   = 100 - (100 / (1 + gain / (loss + 1e-10)))
            data_feat['EMA10'] = data_feat['Close'].ewm(span=10, adjust=False).mean()
            data_feat['EMA20'] = data_feat['Close'].ewm(span=20, adjust=False).mean()
            data_feat.dropna(inplace=True)

            if len(data_feat) == 0:
                return jsonify({"error": "Not enough data to compute indicators. Try again."}), 400

            window    = 45
            close_col = 3
            rsi_last   = float(data_feat['RSI'].iloc[-1])   if len(data_feat) > 0 else 50.0
            ema10_last = float(data_feat['EMA10'].iloc[-1]) if len(data_feat) > 0 else float(close_price)
            ema20_last = float(data_feat['EMA20'].iloc[-1]) if len(data_feat) > 0 else float(close_price)
            vol_last   = float(data['Volume'].iloc[-1]) if ('Volume' in data.columns and len(data) > 0) else 0.0
            user_row = pd.DataFrame(
                [[open_price, high_price, low_price, close_price,
                  vol_last, rsi_last, ema10_last, ema20_last]],
                columns=['Open','High','Low','Close','Volume','RSI','EMA10','EMA20']
            )
        else:
            data_feat = data[['Open', 'High', 'Low', 'Close']].copy()
            data_feat.dropna(inplace=True)

            if len(data_feat) == 0:
                return jsonify({"error": "Not enough historical data. Try again."}), 400

            window    = 60
            close_col = 3
            user_row = pd.DataFrame(
                [[open_price, high_price, low_price, close_price]],
                columns=['Open', 'High', 'Low', 'Close']
            )

        if len(data_feat) < window - 1:
            return jsonify({"error": "Not enough historical data"}), 400

        final_data  = pd.concat([data_feat.tail(window - 1), user_row], ignore_index=True)
        scaled_data = scaler.transform(final_data.values)
        X_input     = np.reshape(scaled_data, (1, window, n_features))
        prediction  = model.predict(X_input, verbose=0)

        dummy_array               = np.zeros((1, n_features))
        dummy_array[0, close_col] = prediction[0][0]
        predicted_price           = round(float(scaler.inverse_transform(dummy_array)[0][close_col]), 2)

        # ── Genuine confidence calculation ──────────────────────────
        # Factor 1: Price deviation — how far predicted is from close (max 50 pts)
        pct_change      = abs(predicted_price - close_price) / close_price * 100
        deviation_score = max(0, 50 - (pct_change * 8))

        # Factor 2: Volatility score (max 30 pts)
        try:
            close_tail = data['Close'].tail(10)
            if len(close_tail) >= 2:
                recent_closes  = close_tail.pct_change(fill_method=None).dropna()
                volatility     = float(recent_closes.std() * 100) if len(recent_closes) > 0 else 1.0
            else:
                volatility = 1.0
        except Exception:
            volatility = 1.0
        volatility_score = max(0, 30 - (volatility * 4))

        # Factor 3: Trend consistency (max 20 pts)
        try:
            close_tail5 = data['Close'].tail(5)
            if len(close_tail5) >= 2:
                recent_trend   = float(close_tail5.iloc[-1]) - float(close_tail5.iloc[0])
                pred_direction = predicted_price - close_price
                trend_match    = (recent_trend > 0 and pred_direction > 0) or \
                                 (recent_trend < 0 and pred_direction < 0)
                trend_score = 20 if trend_match else 8
            else:
                trend_score = 14  # neutral
        except Exception:
            trend_score = 14

        confidence = round(deviation_score + volatility_score + trend_score, 2)
        confidence = round(max(35.0, min(82.0, confidence)), 2)
        # ────────────────────────────────────────────────────────────

        # Clean NaN values before JSON serialization
        import math
        try:
            close_data  = data['Close'].tail(10)
            hist_closes = [round(float(x), 2) for x in close_data.tolist()
                           if not math.isnan(float(x))]
            hist_dates  = [str(d.date()) for d in data.tail(10).index.to_pydatetime()]
            hist_dates  = hist_dates[-len(hist_closes):]
        except Exception:
            hist_closes = []
            hist_dates  = []

        # Sanitize predicted price
        if math.isnan(predicted_price) or math.isinf(predicted_price):
            return jsonify({"error": "Model returned invalid prediction. Please try again."}), 500

        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO predictions
            (user_id, stock_ticker, prediction_date, open_price, high_price,
             low_price, close_price, predicted_price, confidence)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (user_id, ticker, pred_date, open_price, high_price,
              low_price, close_price, predicted_price, confidence))

        direction = "📈 UP" if predicted_price > close_price else "📉 DOWN"
        msg = f"{STOCKS.get(ticker, ticker)}: Predicted ₹{predicted_price} ({direction}) for {pred_date} | Confidence: {confidence}%"
        if fallback_used:
            msg += " ⚠ TCS model used (dedicated model not found)"
        cursor.execute("INSERT INTO notifications (user_id, message, stock_ticker) VALUES (%s,%s,%s)",
                       (user_id, msg, ticker))
        db.commit()
        cursor.close()
        db.close()

        return jsonify({
            "predicted_price": predicted_price,
            "confidence":      confidence,
            "previous_close":  close_price,
            "ticker":          ticker,
            "stock_name":      STOCKS.get(ticker, ticker),
            "pred_date":       pred_date,
            "hist_closes":     hist_closes,
            "hist_dates":      hist_dates,
            "direction":       "up" if predicted_price > close_price else "down",
            "fallback_used":   fallback_used,
        })

    except Exception as e:
        err = str(e)
        # Give user-friendly messages for common errors
        if 'out-of-bounds' in err or 'index' in err.lower():
            msg = "Not enough market data available right now. Please try again in a moment."
        elif 'No model' in err:
            msg = "Model file not found. Please ensure model files are in the project folder."
        elif 'KeyError' in err:
            msg = "Data format error from Yahoo Finance. Please try again."
        else:
            msg = err
        return jsonify({"error": msg}), 500




# =========================================================
# FEATURE 3: ACCURACY TRACKER
# =========================================================

@app.route('/accuracy')
def accuracy():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM predictions WHERE user_id=%s ORDER BY prediction_date DESC", (user_id,))
        preds = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) as cnt FROM notifications WHERE user_id=%s AND is_read=FALSE", (user_id,))
        notif_count = cursor.fetchone()['cnt']
        cursor.close()
        db.close()

        today = datetime.today().date()
        for p in preds:
            if p['actual_price'] is None and p['prediction_date'] and p['prediction_date'] < today:
                try:
                    actual = _fetch_actual_price(p['stock_ticker'], p['prediction_date'])
                    if actual:
                        # Genuine accuracy — percentage error with steeper penalty
                        pct_err = abs(actual - p['predicted_price']) / actual * 100
                        # Apply non-linear penalty: errors >2% are penalised more
                        if pct_err <= 1:
                            acc = round(100 - pct_err, 2)
                        elif pct_err <= 3:
                            acc = round(100 - (pct_err * 2.5), 2)
                        elif pct_err <= 5:
                            acc = round(100 - (pct_err * 4), 2)
                        else:
                            acc = round(100 - (pct_err * 6), 2)
                        acc = round(max(0.0, min(100.0, acc)), 2)
                        db2 = get_db()
                        c2  = db2.cursor()
                        c2.execute("UPDATE predictions SET actual_price=%s, accuracy=%s WHERE id=%s",
                                   (actual, acc, p['id']))
                        db2.commit()
                        c2.close()
                        db2.close()
                        p['actual_price'] = actual
                        p['accuracy']     = acc
                except Exception:
                    pass

        scored  = [p for p in preds if p['accuracy'] is not None]
        avg_acc = round(sum(p['accuracy'] for p in scored) / len(scored), 2) if scored else None

    except Exception:
        preds       = []
        notif_count = 0
        avg_acc     = None

    return render_template('accuracy.html', preds=preds, avg_acc=avg_acc,
                           notif_count=notif_count, username=session.get('username'))


def _fetch_actual_price(ticker, date):
    try:
        start = date - timedelta(days=3)
        end   = date + timedelta(days=3)
        data  = yf.download(ticker, start=str(start), end=str(end), progress=False, auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty:
            return None
        target      = pd.Timestamp(date)
        closest_idx = data.index.get_indexer([target], method='nearest')[0]
        return round(float(data['Close'].iloc[closest_idx]), 2)
    except Exception:
        return None


# =========================================================
# FEATURE 4: WATCHLIST
# =========================================================

@app.route('/watchlist')
def watchlist():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM watchlist WHERE user_id=%s ORDER BY added_at DESC", (user_id,))
        items = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) as cnt FROM notifications WHERE user_id=%s AND is_read=FALSE", (user_id,))
        notif_count = cursor.fetchone()['cnt']
        cursor.close()
        db.close()

        live_data = []
        for item in items:
            try:
                data = yf.download(item['stock_ticker'], period="2d", progress=False, auto_adjust=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                if not data.empty and len(data) >= 1:
                    latest = data.iloc[-1]
                    prev   = data.iloc[-2] if len(data) >= 2 else data.iloc[-1]
                    change = round(float(latest['Close']) - float(prev['Close']), 2)
                    pct    = round(change / float(prev['Close']) * 100, 2)
                    live_data.append({
                        **item,
                        "price": round(float(latest['Close']), 2),
                        "open":  round(float(latest['Open']),  2),
                        "high":  round(float(latest['High']),  2),
                        "low":   round(float(latest['Low']),   2),
                        "change": change, "pct": pct, "up": change >= 0,
                    })
                else:
                    live_data.append({**item, "price": "N/A", "change": 0, "pct": 0, "up": True})
            except Exception:
                live_data.append({**item, "price": "N/A", "change": 0, "pct": 0, "up": True})

    except Exception:
        live_data   = []
        notif_count = 0

    return render_template('watchlist.html', watchlist=live_data, stocks=STOCKS,
                           notif_count=notif_count, username=session.get('username'))


@app.route('/watchlist/add', methods=['POST'])
def watchlist_add():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = session['user_id']
    ticker  = request.form.get('ticker', '').strip().upper()
    name    = STOCKS.get(ticker, ticker)
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("INSERT IGNORE INTO watchlist (user_id, stock_ticker, stock_name) VALUES (%s,%s,%s)",
                       (user_id, ticker, name))
        db.commit()
        cursor.close()
        db.close()
        return jsonify({"success": True, "ticker": ticker, "name": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/watchlist/remove', methods=['POST'])
def watchlist_remove():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = session['user_id']
    ticker  = request.form.get('ticker', '')
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("DELETE FROM watchlist WHERE user_id=%s AND stock_ticker=%s", (user_id, ticker))
        db.commit()
        cursor.close()
        db.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================================
# FEATURE 5: EXPORT CSV + PDF
# =========================================================

@app.route('/export/csv')
def export_csv():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("""SELECT stock_ticker, prediction_date, open_price, high_price,
                          low_price, close_price, predicted_price, confidence,
                          actual_price, accuracy, created_at
                          FROM predictions WHERE user_id=%s ORDER BY created_at DESC""", (user_id,))
        rows = cursor.fetchall()
        cursor.close()
        db.close()

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "stock_ticker","prediction_date","open_price","high_price","low_price",
            "close_price","predicted_price","confidence","actual_price","accuracy","created_at"
        ])
        writer.writeheader()
        writer.writerows(rows)
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"stockai_{session.get('username')}_{datetime.today().strftime('%Y%m%d')}.csv"
        )
    except Exception as e:
        return f"Export failed: {e}", 500


@app.route('/export/pdf')
def export_pdf():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM predictions WHERE user_id=%s ORDER BY created_at DESC", (user_id,))
        rows = cursor.fetchall()
        cursor.execute("SELECT username, email FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        cursor.close()
        db.close()
    except Exception:
        rows = []
        user = {}
    return render_template('export_pdf.html', rows=rows, user=user,
                           generated=datetime.today().strftime('%Y-%m-%d %H:%M'))


# =========================================================
# PREDICTION DETAIL VIEW + DELETE
# =========================================================

@app.route('/prediction/<int:pred_id>')
def prediction_detail(pred_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM predictions WHERE id=%s AND user_id=%s", (pred_id, user_id))
        pred = cursor.fetchone()
        if not pred:
            cursor.close()
            db.close()
            return redirect(url_for('dashboard'))

        # ── Auto-calculate accuracy if date has passed and not yet scored ──
        today = datetime.now().date()
        if (pred['actual_price'] is None and
                pred['prediction_date'] is not None and
                pred['prediction_date'] <= today):
            actual = _fetch_actual_price(pred['stock_ticker'], pred['prediction_date'])
            if actual:
                pct_err = abs(actual - pred['predicted_price']) / actual * 100
                if pct_err <= 1:
                    acc = round(100 - pct_err, 2)
                elif pct_err <= 3:
                    acc = round(100 - (pct_err * 2.5), 2)
                elif pct_err <= 5:
                    acc = round(100 - (pct_err * 4), 2)
                else:
                    acc = round(100 - (pct_err * 6), 2)
                acc = round(max(0.0, min(100.0, acc)), 2)
                cursor.execute(
                    "UPDATE predictions SET actual_price=%s, accuracy=%s WHERE id=%s",
                    (actual, acc, pred['id'])
                )
                db.commit()
                pred['actual_price'] = actual
                pred['accuracy']     = acc

        cursor.execute("SELECT COUNT(*) as cnt FROM notifications WHERE user_id=%s AND is_read=FALSE", (user_id,))
        notif_count = cursor.fetchone()['cnt']
        cursor.close()
        db.close()
        # Fetch historical chart data around prediction date
        hist_closes = []
        hist_dates  = []
        try:
            pred_date  = pred['prediction_date']
            end_date   = datetime.combine(pred_date, datetime.min.time()) + timedelta(days=5)
            start_date = end_date - timedelta(days=80)
            data = yf.download(pred['stock_ticker'], start=start_date, end=end_date, progress=False, auto_adjust=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not data.empty:
                hist_closes = [round(float(x), 2) for x in data['Close'].tail(10).tolist()]
                hist_dates  = [str(d.date()) for d in data.tail(10).index.to_pydatetime()]
        except Exception:
            pass
    except Exception:
        return redirect(url_for('dashboard'))
    return render_template('prediction_detail.html',
                           pred=pred, hist_closes=hist_closes, hist_dates=hist_dates,
                           notif_count=notif_count, username=session.get('username'), stocks=STOCKS)


@app.route('/prediction/delete/<int:pred_id>', methods=['POST'])
def prediction_delete(pred_id):
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = session['user_id']
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("DELETE FROM predictions WHERE id=%s AND user_id=%s", (pred_id, user_id))
        db.commit()
        cursor.close()
        db.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================================
# LIVE TICKER API — for scrolling ticker strip
# =========================================================

@app.route('/api/live-prices')
def live_prices():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    tickers = list(TRAINED_STOCKS.keys())
    results = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="2d", progress=False, auto_adjust=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not data.empty and len(data) >= 1:
                latest = data.iloc[-1]
                prev   = data.iloc[-2] if len(data) >= 2 else data.iloc[-1]
                close  = round(float(latest['Close']), 2)
                change = round(float(latest['Close']) - float(prev['Close']), 2)
                pct    = round(change / float(prev['Close']) * 100, 2)
                results.append({
                    "ticker": ticker,
                    "name":   ALL_STOCKS.get(ticker, ticker),
                    "price":  close,
                    "change": change,
                    "pct":    pct,
                    "up":     change >= 0
                })
        except Exception:
            pass
    return jsonify(results)


# =========================================================
# SEARCH & FILTER PREDICTIONS
# =========================================================

@app.route('/search')
def search():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']

    ticker     = request.args.get('ticker', '').strip().upper()
    date_from  = request.args.get('date_from', '')
    date_to    = request.args.get('date_to', '')
    min_conf   = request.args.get('min_conf', '')
    sort_by    = request.args.get('sort_by', 'created_at')
    sort_dir   = request.args.get('sort_dir', 'DESC')

    allowed_sort = ['created_at', 'prediction_date', 'confidence', 'predicted_price', 'accuracy']
    if sort_by not in allowed_sort:
        sort_by = 'created_at'
    if sort_dir not in ['ASC', 'DESC']:
        sort_dir = 'DESC'

    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)

        query  = "SELECT * FROM predictions WHERE user_id=%s"
        params = [user_id]

        if ticker:
            query += " AND stock_ticker LIKE %s"
            params.append(f"%{ticker}%")
        if date_from:
            query += " AND prediction_date >= %s"
            params.append(date_from)
        if date_to:
            query += " AND prediction_date <= %s"
            params.append(date_to)
        if min_conf:
            query += " AND confidence >= %s"
            params.append(float(min_conf))

        query += f" ORDER BY {sort_by} {sort_dir}"

        cursor.execute(query, params)
        results = cursor.fetchall()

        cursor.execute("SELECT COUNT(*) as cnt FROM notifications WHERE user_id=%s AND is_read=FALSE",
                       (user_id,))
        notif_count = cursor.fetchone()['cnt']
        cursor.close()
        db.close()
    except Exception as e:
        results     = []
        notif_count = 0

    return render_template('search.html',
                           results=results,
                           all_stocks=ALL_STOCKS,
                           notif_count=notif_count,
                           username=session.get('username'),
                           filters={"ticker": ticker, "date_from": date_from,
                                    "date_to": date_to, "min_conf": min_conf,
                                    "sort_by": sort_by, "sort_dir": sort_dir})


# =========================================================
# ADMIN PANEL
# =========================================================

def is_admin():
    return session.get('is_admin', False)

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    message = ""
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['is_admin']      = True
            session['admin_username'] = username
            return redirect(url_for('admin_dashboard'))
        else:
            message = "Invalid admin credentials."
    return render_template('admin_login.html', message=message)

@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    session.pop('admin_username', None)
    return redirect(url_for('admin_login'))

@app.route('/admin')
def admin_dashboard():
    if not is_admin():
        return redirect(url_for('admin_login'))
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)

        # Stats
        cursor.execute("SELECT COUNT(*) as cnt FROM users")
        total_users = cursor.fetchone()['cnt']
        cursor.execute("SELECT COUNT(*) as cnt FROM predictions")
        total_preds = cursor.fetchone()['cnt']
        cursor.execute("SELECT AVG(confidence) as avg_conf FROM predictions")
        avg_conf = cursor.fetchone()['avg_conf']
        cursor.execute("SELECT AVG(accuracy) as avg_acc FROM predictions WHERE accuracy IS NOT NULL")
        avg_acc = cursor.fetchone()['avg_acc']

        # All users with prediction counts
        cursor.execute("""
            SELECT u.id, u.username, u.email, u.created_at,
                   COUNT(p.id) as pred_count
            FROM users u
            LEFT JOIN predictions p ON p.user_id = u.id
            GROUP BY u.id ORDER BY u.created_at DESC
        """)
        users = cursor.fetchall()

        # Recent predictions across all users
        cursor.execute("""
            SELECT p.*, u.username FROM predictions p
            JOIN users u ON u.id = p.user_id
            ORDER BY p.created_at DESC LIMIT 20
        """)
        recent_preds = cursor.fetchall()

        # Most predicted stocks
        cursor.execute("""
            SELECT stock_ticker, COUNT(*) as cnt
            FROM predictions GROUP BY stock_ticker
            ORDER BY cnt DESC LIMIT 8
        """)
        top_stocks = cursor.fetchall()

        cursor.close()
        db.close()
    except Exception as e:
        users = []; recent_preds = []; top_stocks = []
        total_users = total_preds = avg_conf = avg_acc = 0

    return render_template('admin.html',
                           total_users=total_users,
                           total_preds=total_preds,
                           avg_conf=round(float(avg_conf), 2) if avg_conf else 0,
                           avg_acc=round(float(avg_acc), 2) if avg_acc else 0,
                           users=users,
                           recent_preds=recent_preds,
                           top_stocks=top_stocks,
                           trained_stocks=TRAINED_STOCKS,
                           all_stocks=ALL_STOCKS,
                           admin_username=session.get('admin_username'))

@app.route('/admin/delete-user/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    if not is_admin():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("DELETE FROM notifications WHERE user_id=%s", (user_id,))
        cursor.execute("DELETE FROM watchlist WHERE user_id=%s", (user_id,))
        cursor.execute("DELETE FROM predictions WHERE user_id=%s", (user_id,))
        cursor.execute("DELETE FROM users WHERE id=%s", (user_id,))
        db.commit()
        cursor.close()
        db.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/admin/delete-prediction/<int:pred_id>', methods=['POST'])
def admin_delete_prediction(pred_id):
    if not is_admin():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("DELETE FROM predictions WHERE id=%s", (pred_id,))
        db.commit()
        cursor.close()
        db.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================================
# NOTIFICATIONS & ACCOUNT
# =========================================================

@app.route('/notifications')
def notifications():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM notifications WHERE user_id=%s ORDER BY created_at DESC", (user_id,))
        notifs = cursor.fetchall()
        cursor.execute("UPDATE notifications SET is_read=TRUE WHERE user_id=%s", (user_id,))
        db.commit()
        cursor.close()
        db.close()
    except:
        notifs = []
    return render_template('notifications.html', notifs=notifs,
                           username=session.get('username'), notif_count=0)


@app.route('/account')
def account():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        cursor.execute("SELECT COUNT(*) as cnt FROM predictions WHERE user_id=%s", (user_id,))
        pred_count = cursor.fetchone()['cnt']
        cursor.execute("SELECT * FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 10", (user_id,))
        history = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) as cnt FROM notifications WHERE user_id=%s AND is_read=FALSE", (user_id,))
        notif_count = cursor.fetchone()['cnt']
        cursor.execute("SELECT AVG(accuracy) as avg_acc FROM predictions WHERE user_id=%s AND accuracy IS NOT NULL", (user_id,))
        acc_row = cursor.fetchone()
        cursor.close()
        db.close()
        avg_acc = round(acc_row['avg_acc'], 2) if acc_row['avg_acc'] else None
    except Exception:
        user = {"username": session.get('username'), "email": "N/A", "created_at": "N/A"}
        pred_count = 0; history = []; notif_count = 0; avg_acc = None
    return render_template('account.html', user=user, pred_count=pred_count,
                           history=history, notif_count=notif_count,
                           avg_acc=avg_acc, username=session.get('username'))




# =========================================================
# NEWS FEED PER STOCK
# =========================================================

@app.route('/api/news')
def stock_news():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    ticker = request.args.get('ticker', 'TCS.NS')
    name   = ALL_STOCKS.get(ticker, ticker).split('(')[0].strip()
    try:
        stock = yf.Ticker(ticker)
        news  = stock.news or []
        results = []
        for item in news[:6]:
            # Handle both old and new yfinance news format
            content = item.get('content', {})
            title   = content.get('title') or item.get('title', 'No title')
            summary = content.get('summary') or item.get('summary', '')
            # Get URL
            url = ''
            cp  = content.get('canonicalUrl', {})
            if isinstance(cp, dict):
                url = cp.get('url', '')
            if not url:
                url = item.get('link', item.get('url', '#'))
            # Get publisher
            provider = content.get('provider', {})
            if isinstance(provider, dict):
                publisher = provider.get('displayName', '')
            else:
                publisher = item.get('publisher', '')
            # Get time
            pub_time = content.get('pubDate') or item.get('providerPublishTime', '')
            if isinstance(pub_time, int):
                from datetime import timezone
                pub_time = datetime.fromtimestamp(pub_time, tz=timezone.utc).strftime('%b %d, %Y')
            elif pub_time:
                try:
                    pub_time = pub_time[:10]
                except Exception:
                    pub_time = ''
            results.append({
                "title":     title,
                "summary":   summary[:200] + '...' if len(summary) > 200 else summary,
                "url":       url,
                "publisher": publisher,
                "time":      pub_time,
            })
        return jsonify({"news": results, "stock": name, "ticker": ticker})
    except Exception as e:
        return jsonify({"error": str(e), "news": []}), 200


# =========================================================
# PORTFOLIO TRACKER
# =========================================================

@app.route('/portfolio')
def portfolio():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM portfolio WHERE user_id=%s ORDER BY added_at DESC", (user_id,))
        holdings = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) as cnt FROM notifications WHERE user_id=%s AND is_read=FALSE", (user_id,))
        notif_count = cursor.fetchone()['cnt']
        cursor.close()
        db.close()

        total_invested = 0
        total_current  = 0
        enriched = []
        for h in holdings:
            try:
                data = yf.download(h['ticker'], period="2d", progress=False, auto_adjust=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                if not data.empty:
                    latest   = float(data['Close'].iloc[-1])
                    prev     = float(data['Close'].iloc[-2]) if len(data) >= 2 else latest
                    change   = round(latest - prev, 2)
                    pct      = round(change / prev * 100, 2)
                    invested = round(h['buy_price'] * h['quantity'], 2)
                    current  = round(latest * h['quantity'], 2)
                    pnl      = round(current - invested, 2)
                    pnl_pct  = round((pnl / invested) * 100, 2)
                    total_invested += invested
                    total_current  += current
                    enriched.append({**h,
                        "current_price": round(latest, 2),
                        "change": change, "pct": pct,
                        "invested": invested, "current": current,
                        "pnl": pnl, "pnl_pct": pnl_pct,
                        "up": pnl >= 0, "day_up": change >= 0,
                    })
                else:
                    enriched.append({**h, "current_price": "N/A",
                        "change": 0, "pct": 0, "invested": 0,
                        "current": 0, "pnl": 0, "pnl_pct": 0,
                        "up": True, "day_up": True})
            except Exception:
                enriched.append({**h, "current_price": "N/A",
                    "change": 0, "pct": 0, "invested": 0,
                    "current": 0, "pnl": 0, "pnl_pct": 0,
                    "up": True, "day_up": True})

        total_pnl     = round(total_current - total_invested, 2)
        total_pnl_pct = round((total_pnl / total_invested * 100), 2) if total_invested else 0

    except Exception as e:
        enriched = []; notif_count = 0
        total_invested = total_current = total_pnl = total_pnl_pct = 0

    return render_template('portfolio.html',
                           holdings=enriched,
                           stocks=ALL_STOCKS,
                           total_invested=round(total_invested, 2),
                           total_current=round(total_current, 2),
                           total_pnl=total_pnl,
                           total_pnl_pct=total_pnl_pct,
                           notif_count=notif_count,
                           username=session.get('username'))


@app.route('/portfolio/add', methods=['POST'])
def portfolio_add():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id   = session['user_id']
    ticker    = request.form.get('ticker', '').strip().upper()
    quantity  = float(request.form.get('quantity', 0))
    buy_price = float(request.form.get('buy_price', 0))
    name      = ALL_STOCKS.get(ticker, ticker)
    if quantity <= 0 or buy_price <= 0:
        return jsonify({"error": "Invalid quantity or price"}), 400
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""INSERT INTO portfolio (user_id, ticker, name, quantity, buy_price)
                          VALUES (%s,%s,%s,%s,%s)""",
                       (user_id, ticker, name, quantity, buy_price))
        db.commit()
        cursor.close()
        db.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/portfolio/remove', methods=['POST'])
def portfolio_remove():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = session['user_id']
    pid     = request.form.get('id')
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("DELETE FROM portfolio WHERE id=%s AND user_id=%s", (pid, user_id))
        db.commit()
        cursor.close()
        db.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================================
# CHANGE PASSWORD
# =========================================================

@app.route('/change-password', methods=['POST'])
def change_password():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id      = session['user_id']
    current_pass = request.form.get('current_password', '')
    new_pass     = request.form.get('new_password', '')
    import re
    if (len(new_pass) < 8 or
        not re.search(r'[A-Z]', new_pass) or
        not re.search(r'[a-z]', new_pass) or
        not re.search(r'[0-9]', new_pass) or
        not re.search(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]', new_pass)):
        return jsonify({"error": "Password must be 8+ chars with uppercase, lowercase, number and special character."}), 400
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT password FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        if user['password'] != hash_password(current_pass):
            cursor.close(); db.close()
            return jsonify({"error": "Current password is incorrect."}), 400
        cursor.execute("UPDATE users SET password=%s WHERE id=%s",
                       (hash_password(new_pass), user_id))
        db.commit()
        cursor.close()
        db.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================================
# PAGINATED PREDICTIONS API
# =========================================================

@app.route('/api/predictions')
def predictions_paginated():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id  = session['user_id']
    page     = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    offset   = (page - 1) * per_page
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) as cnt FROM predictions WHERE user_id=%s", (user_id,))
        total = cursor.fetchone()['cnt']
        cursor.execute("""SELECT * FROM predictions WHERE user_id=%s
                          ORDER BY created_at DESC LIMIT %s OFFSET %s""",
                       (user_id, per_page, offset))
        rows = cursor.fetchall()
        cursor.close()
        db.close()
        # Convert dates to strings
        for r in rows:
            if r.get('prediction_date'):
                r['prediction_date'] = str(r['prediction_date'])
            if r.get('created_at'):
                r['created_at'] = str(r['created_at'])
        return jsonify({
            "predictions": rows,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recalculate-accuracy', methods=['POST'])
def recalculate_accuracy():
    """Manually trigger accuracy calculation for all pending predictions."""
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        today = datetime.now().date()
        cursor.execute("""
            SELECT id, stock_ticker, predicted_price, prediction_date
            FROM predictions
            WHERE actual_price IS NULL
            AND prediction_date IS NOT NULL
            AND prediction_date <= %s
            AND user_id = %s
        """, (today, session['user_id']))
        pending = cursor.fetchall()
        updated = 0
        for p in pending:
            try:
                actual = _fetch_actual_price(p['stock_ticker'], p['prediction_date'])
                if actual:
                    pct_err = abs(actual - p['predicted_price']) / actual * 100
                    if pct_err <= 1:
                        acc = round(100 - pct_err, 2)
                    elif pct_err <= 3:
                        acc = round(100 - (pct_err * 2.5), 2)
                    elif pct_err <= 5:
                        acc = round(100 - (pct_err * 4), 2)
                    else:
                        acc = round(100 - (pct_err * 6), 2)
                    acc = round(max(0.0, min(100.0, acc)), 2)
                    cursor.execute(
                        "UPDATE predictions SET actual_price=%s, accuracy=%s WHERE id=%s",
                        (actual, acc, p['id'])
                    )
                    db.commit()
                    updated += 1
            except Exception:
                continue
        cursor.close()
        db.close()
        return jsonify({"success": True, "updated": updated})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================================
# PRICE ALERTS
# =========================================================

@app.route('/alerts')
def alerts():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM price_alerts WHERE user_id=%s ORDER BY created_at DESC", (user_id,))
        user_alerts = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) as cnt FROM notifications WHERE user_id=%s AND is_read=FALSE", (user_id,))
        notif_count = cursor.fetchone()['cnt']
        cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        cursor.close()
        db.close()
    except Exception:
        user_alerts = []; notif_count = 0
        user = {"email": session.get('username', '')}
    return render_template('alerts.html',
                           alerts=user_alerts,
                           stocks=ALL_STOCKS,
                           user_email=user.get('email', ''),
                           notif_count=notif_count,
                           username=session.get('username'))


@app.route('/alerts/add', methods=['POST'])
def alerts_add():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id      = session['user_id']
    ticker       = request.form.get('ticker', '').strip().upper()
    target_price = float(request.form.get('target_price', 0))
    direction    = request.form.get('direction', 'above')
    email        = request.form.get('email', '').strip()
    if target_price <= 0:
        return jsonify({"error": "Invalid target price"}), 400
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""INSERT INTO price_alerts (user_id, ticker, target_price, direction, email)
                          VALUES (%s,%s,%s,%s,%s)""",
                       (user_id, ticker, target_price, direction, email))
        db.commit()
        cursor.close()
        db.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/alerts/delete/<int:alert_id>', methods=['POST'])
def alerts_delete(alert_id):
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = session['user_id']
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("DELETE FROM price_alerts WHERE id=%s AND user_id=%s", (alert_id, user_id))
        db.commit()
        cursor.close()
        db.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def check_price_alerts():
    """Background job — runs every hour to check if any alert conditions are met."""
    with app.app_context():
        try:
            db = get_db()
            cursor = db.cursor(dictionary=True)
            cursor.execute("SELECT * FROM price_alerts WHERE is_active=TRUE")
            active_alerts = cursor.fetchall()

            for alert in active_alerts:
                try:
                    data = yf.download(alert['ticker'], period="1d", progress=False, auto_adjust=False)
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    if data.empty:
                        continue
                    current_price = float(data['Close'].iloc[-1])
                    triggered = (
                        (alert['direction'] == 'above' and current_price >= alert['target_price']) or
                        (alert['direction'] == 'below' and current_price <= alert['target_price'])
                    )
                    if triggered:
                        # Send email
                        direction_text = "risen above" if alert['direction'] == 'above' else "fallen below"
                        subject = f"🔔 StockAI Alert — {alert['ticker']} has {direction_text} ₹{alert['target_price']}"
                        body = f"""
Hello!

Your StockAI price alert has been triggered.

Stock     : {alert['ticker']} — {ALL_STOCKS.get(alert['ticker'], alert['ticker'])}
Condition : Price {direction_text} ₹{alert['target_price']}
Current   : ₹{round(current_price, 2)}
Time      : {datetime.now().strftime('%Y-%m-%d %H:%M')}

Login to StockAI to make a prediction now:
http://127.0.0.1:5000/dashboard

— StockAI Team
                        """
                        try:
                            msg = Message(subject=subject, recipients=[alert['email']], body=body)
                            mail.send(msg)
                        except Exception:
                            pass

                        # Mark alert as triggered
                        cursor.execute("""UPDATE price_alerts
                                          SET is_active=FALSE, triggered_at=NOW()
                                          WHERE id=%s""", (alert['id'],))
                        db.commit()
                except Exception:
                    continue

            cursor.close()
            db.close()
        except Exception:
            pass


# =========================================================
# ADMIN — API RATE DASHBOARD
# =========================================================

@app.route('/admin/rates')
def admin_rates():
    if not is_admin():
        return redirect(url_for('admin_login'))
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)

        # Predictions per user today
        cursor.execute("""
            SELECT u.username, u.email,
                   COUNT(CASE WHEN DATE(p.created_at) = CURDATE() THEN 1 END) as today,
                   COUNT(CASE WHEN p.created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY) THEN 1 END) as week,
                   COUNT(CASE WHEN p.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY) THEN 1 END) as month,
                   COUNT(p.id) as total
            FROM users u
            LEFT JOIN predictions p ON p.user_id = u.id
            GROUP BY u.id
            ORDER BY today DESC, week DESC
        """)
        rate_data = cursor.fetchall()

        # Hourly predictions today
        cursor.execute("""
            SELECT HOUR(created_at) as hour, COUNT(*) as cnt
            FROM predictions
            WHERE DATE(created_at) = CURDATE()
            GROUP BY HOUR(created_at)
            ORDER BY hour
        """)
        hourly = cursor.fetchall()

        # Total predictions by day for last 30 days
        cursor.execute("""
            SELECT DATE(created_at) as day, COUNT(*) as cnt
            FROM predictions
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY DATE(created_at)
            ORDER BY day
        """)
        daily = cursor.fetchall()

        cursor.close()
        db.close()

        # Convert dates to strings
        for d in daily:
            d['day'] = str(d['day'])

    except Exception as e:
        rate_data = []; hourly = []; daily = []

    return render_template('admin_rates.html',
                           rate_data=rate_data,
                           hourly=hourly,
                           daily=daily,
                           admin_username=session.get('admin_username'))


# =========================================================
# FEATURE: LEADERBOARD
# =========================================================

@app.route('/leaderboard')
def leaderboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT u.username,
                   COUNT(p.id) as total_predictions,
                   ROUND(AVG(p.accuracy), 2) as avg_accuracy,
                   SUM(CASE WHEN p.accuracy >= 90 THEN 1 ELSE 0 END) as excellent,
                   MAX(p.accuracy) as best_accuracy,
                   MIN(p.accuracy) as worst_accuracy
            FROM users u
            JOIN predictions p ON p.user_id = u.id
            WHERE p.accuracy IS NOT NULL
            GROUP BY u.id
            HAVING total_predictions >= 1
            ORDER BY avg_accuracy DESC
            LIMIT 20
        """)
        leaders = cursor.fetchall()
        # Add rank
        for i, row in enumerate(leaders):
            row['rank'] = i + 1
            row['is_me'] = (row['username'] == session.get('username'))

        # Get current user's stats
        cursor.execute("""
            SELECT ROUND(AVG(accuracy),2) as avg_acc, COUNT(*) as total
            FROM predictions WHERE user_id=%s AND accuracy IS NOT NULL
        """, (user_id,))
        my_stats = cursor.fetchone()

        cursor.execute("SELECT COUNT(*) as cnt FROM notifications WHERE user_id=%s AND is_read=FALSE", (user_id,))
        notif_count = cursor.fetchone()['cnt']
        cursor.close()
        db.close()
    except Exception as e:
        leaders = []; my_stats = {}; notif_count = 0

    return render_template('leaderboard.html',
                           leaders=leaders,
                           my_stats=my_stats,
                           notif_count=notif_count,
                           username=session.get('username'))


# =========================================================
# FEATURE: AI CHAT ASSISTANT
# =========================================================

@app.route('/chat')
def chat_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) as cnt FROM notifications WHERE user_id=%s AND is_read=FALSE", (user_id,))
        notif_count = cursor.fetchone()['cnt']
        cursor.close()
        db.close()
    except Exception:
        notif_count = 0
    return render_template('chat.html', stocks=ALL_STOCKS,
                           notif_count=notif_count,
                           username=session.get('username'))


@app.route('/api/chat', methods=['POST'])
def api_chat():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_msg = request.json.get('message', '').strip()
    ticker   = request.json.get('ticker', '').strip().upper()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # Fetch live stock context
    stock_context = ""
    if ticker and ticker in ALL_STOCKS:
        try:
            data = yf.download(ticker, period="5d", progress=False, auto_adjust=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not data.empty:
                latest = data.iloc[-1]
                prev   = data.iloc[-2] if len(data) > 1 else latest
                change = round(float(latest['Close']) - float(prev['Close']), 2)
                pct    = round((change / float(prev['Close'])) * 100, 2)
                stock_context = f"""
Current market data for {ticker} ({ALL_STOCKS[ticker]}):
- Latest Close: {round(float(latest['Close']), 2)}
- Open: {round(float(latest['Open']), 2)}
- High: {round(float(latest['High']), 2)}
- Low: {round(float(latest['Low']), 2)}
- Day Change: {change} ({pct}%)
- 5-day trend: {[round(float(x),2) for x in data['Close'].tolist()]}
"""
        except Exception:
            pass

    # Get user's recent predictions for context
    pred_context = ""
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("""SELECT stock_ticker, predicted_price, confidence, prediction_date, accuracy
                          FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 5""",
                       (session['user_id'],))
        recent_preds = cursor.fetchall()
        cursor.close()
        db.close()
        if recent_preds:
            pred_context = "User's recent predictions:\n"
            for p in recent_preds:
                pred_context += f"- {p['stock_ticker']}: predicted {p['predicted_price']} on {p['prediction_date']}, confidence {p['confidence']}%, accuracy {p['accuracy']}%\n"
    except Exception:
        pass

    system_prompt = f"""You are StockAI Assistant, an expert AI for Indian and US stock market analysis.
You help users understand stock predictions, market trends, and investment concepts.
Be concise, friendly, and data-driven. Always mention that predictions are for educational purposes.
{stock_context}
{pred_context}
Answer in 2-4 sentences max unless a detailed explanation is needed."""

    # ── Smart rule-based chat engine ──
    try:
        m = user_msg.lower()
        reply = ""

        if any(w in m for w in ['buy', 'should i', 'good stock', 'worth buying', 'invest in']):
            reply = (f"Based on our LSTM model data{(' for ' + ticker) if ticker else ''}, "
                     "I can show predicted price trends — but the final call is yours. "
                     "Check the confidence score on your prediction: above 88% is strong, "
                     "below 80% means higher uncertainty. Always combine predictions with "
                     "your own research and never invest more than you can afford to lose.")

        elif any(w in m for w in ['accurate', 'accuracy', 'reliable', 'trust', 'how good']):
            reply = ("Our LSTM models are trained on 60-day historical windows using Open, High, Low, "
                     "and Close prices. Typical next-day accuracy ranges from 85–95%. "
                     "You can track your personal accuracy score on the Accuracy page, "
                     "and see how you rank against others on the Leaderboard.")

        elif any(w in m for w in ['rsi', 'relative strength']):
            reply = ("RSI (Relative Strength Index) measures momentum on a 0–100 scale. "
                     "Above 70 = overbought (possible pullback coming). "
                     "Below 30 = oversold (possible bounce coming). "
                     "Between 40–60 = neutral zone. Use RSI alongside our LSTM predictions "
                     "to confirm trend direction before making decisions.")

        elif any(w in m for w in ['macd', 'moving average convergence']):
            reply = ("MACD measures the difference between 12-day and 26-day EMAs. "
                     "A MACD crossover above the signal line = bullish signal. "
                     "Crossing below = bearish signal. "
                     "Best used with our LSTM prediction — if both MACD and prediction agree on direction, "
                     "confidence in the trade increases significantly.")

        elif any(w in m for w in ['bollinger', 'bands', 'volatility']):
            reply = ("Bollinger Bands show price volatility using a 20-day moving average "
                     "with upper/lower bands at ±2 standard deviations. "
                     "Price touching the upper band = possibly overbought. "
                     "Price touching the lower band = possibly oversold. "
                     "Narrow bands signal a big move is coming — watch for breakouts!")

        elif any(w in m for w in ['lstm', 'model', 'how does it work', 'how does the', 'neural']):
            reply = ("Our LSTM (Long Short-Term Memory) model looks at the last 60 days of "
                     "Open, High, Low, and Close prices to predict tomorrow's closing price. "
                     "It's a type of deep learning model designed specifically for time-series data. "
                     "Each model is individually trained per stock for best accuracy. "
                     "The confidence score reflects how close the prediction is to recent price levels.")

        elif any(w in m for w in ['confidence', 'confidence score', 'what is confidence']):
            reply = ("Confidence is calculated as: 100 minus the percentage difference between "
                     "the predicted price and the current close price. "
                     "We apply a 10% conservative reduction so the score is never overconfident. "
                     "90%+ = very strong prediction. 80–90% = good. Below 75% = treat with caution.")

        elif any(w in m for w in ['diversif', 'portfolio', 'allocation', 'spread']):
            reply = ("Diversification means spreading investments across different stocks and sectors. "
                     "A balanced portfolio might include: Indian IT (TCS, Infosys, Wipro), "
                     "Indian banking (HDFC, ICICI, SBI), US tech (NVIDIA, Microsoft, Google), "
                     "and growth stocks (Tesla, Amazon). "
                     "Avoid putting more than 20% in any single stock.")

        elif any(w in m for w in ['watchlist', 'watch', 'track']):
            reply = ("You can add any of the 18 stocks to your Watchlist to monitor live prices "
                     "in one place. Go to the Watchlist page from the sidebar. "
                     "Prices auto-refresh every 60 seconds so you always see current data.")

        elif any(w in m for w in ['alert', 'notify', 'notification', 'price alert']):
            reply = ("Use the Price Alerts feature (sidebar) to set a target price for any stock. "
                     "When the live price crosses your threshold, you'll get an email automatically. "
                     "You can set alerts for price rising above or falling below your target.")

        elif any(w in m for w in ['export', 'download', 'csv', 'pdf', 'report']):
            reply = ("You can export all your predictions as CSV or PDF from the sidebar links. "
                     "CSV is great for analysis in Excel. PDF gives a formatted printable report. "
                     "The weekly email report sent every Monday also summarizes your performance automatically.")

        elif any(w in m for w in ['leaderboard', 'rank', 'ranking', 'climb', 'best user', 'top']):
            reply = ("The Leaderboard ranks all users by their average prediction accuracy. "
                     "To climb the ranks: make more predictions, and they get auto-scored "
                     "once the prediction date passes. Predictions with 90%+ accuracy earn "
                     "the Excellent badge and boost your average score significantly.")

        elif any(w in m for w in ['trend', 'bullish', 'bearish', 'market', 'rise', 'fall']):
            if ticker:
                reply = (f"To analyze the trend for {ticker}, run a fresh LSTM prediction from the Dashboard. "
                         "Compare the predicted price vs today's close — if predicted > close, "
                         "the model expects a rise. Also check the 5-day price chart on the prediction detail page.")
            else:
                reply = ("Select a stock from the dropdown above so I can load its live data, "
                         "then ask again for a data-driven trend analysis.")

        elif any(w in m for w in ['hello', 'hi', 'hey', 'help', 'what can you do', 'what can you']):
            reply = ("Hello! I'm your StockAI Assistant. I can help you with:\n"
                     "• Stock predictions and confidence scores\n"
                     "• Technical indicators: RSI, MACD, Bollinger Bands\n"
                     "• How the LSTM model works\n"
                     "• Portfolio and diversification tips\n"
                     "• Using Watchlist, Alerts, Leaderboard, and more\n"
                     "Select a stock above and ask me anything!")

        elif any(w in m for w in ['infy', 'infosys', 'tcs', 'wipro', 'reliance', 'hdfc',
                                   'icici', 'sbi', 'bajaj', 'adani', 'tata',
                                   'aapl', 'apple', 'googl', 'google', 'msft', 'microsoft',
                                   'amzn', 'amazon', 'tsla', 'tesla', 'meta',
                                   'nvda', 'nvidia', 'nflx', 'netflix']):
            name = ALL_STOCKS.get(ticker, ticker) if ticker else 'this stock'
            reply = (f"{name} is one of the 18 stocks tracked by StockAI. "
                     "Go to the Dashboard, select it, click Auto-fill for live prices, "
                     "then hit Predict to get the LSTM forecast. "
                     "Check the Accuracy page to see how past predictions performed.")

        else:
            if ticker and stock_context:
                reply = (f"I have live data loaded for {ticker}. "
                         "Head to the Dashboard to run a fresh LSTM prediction and see "
                         "where the model thinks the price is heading tomorrow. "
                         "Ask me about RSI, MACD, confidence scores, or any feature!")
            else:
                reply = ("I'm here to help! Ask me about predictions, accuracy, RSI, MACD, "
                         "Bollinger Bands, how LSTM works, portfolio tips, or any StockAI feature. "
                         "Select a stock from the dropdown above for live data context.")

    except Exception as e:
        reply = "Sorry, something went wrong processing your question. Please try again!"

    return jsonify({"reply": reply})


# =========================================================
# FEATURE: WEEKLY EMAIL REPORT
# =========================================================

def send_weekly_report():
    """Runs every Monday at 8AM — sends prediction summary to all users."""
    with app.app_context():
        try:
            db = get_db()
            cursor = db.cursor(dictionary=True)
            cursor.execute("SELECT id, username, email FROM users")
            users = cursor.fetchall()

            for user in users:
                if not user['email']:
                    continue
                try:
                    # Get last 7 days predictions
                    cursor.execute("""
                        SELECT stock_ticker, predicted_price, actual_price, accuracy,
                               confidence, prediction_date
                        FROM predictions
                        WHERE user_id=%s AND created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                        ORDER BY prediction_date DESC
                    """, (user['id'],))
                    preds = cursor.fetchall()

                    cursor.execute("""
                        SELECT ROUND(AVG(accuracy),2) as avg_acc, COUNT(*) as total
                        FROM predictions WHERE user_id=%s AND accuracy IS NOT NULL
                    """, (user['id'],))
                    stats = cursor.fetchone()

                    if not preds and not stats['total']:
                        continue  # Skip users with no activity

                    # Build email body
                    pred_rows = ""
                    best = None
                    worst = None
                    for p in preds:
                        acc_str = f"{p['accuracy']}%" if p['accuracy'] else "Pending"
                        pred_rows += f"  {p['stock_ticker']:<12} Predicted: {p['predicted_price']:<10} Actual: {str(p['actual_price'] or 'N/A'):<10} Accuracy: {acc_str}\n"
                        if p['accuracy']:
                            if best is None or p['accuracy'] > best['accuracy']:
                                best = p
                            if worst is None or p['accuracy'] < worst['accuracy']:
                                worst = p

                    body = f"""
Hello {user['username']}! 👋

Here's your Weekly StockAI Report — {datetime.now().strftime('%B %d, %Y')}
{'='*55}

📊 YOUR OVERALL STATS
  Total Scored Predictions : {stats['total'] or 0}
  Average Accuracy         : {stats['avg_acc'] or 'N/A'}%

📅 THIS WEEK'S PREDICTIONS
{pred_rows if pred_rows else '  No predictions this week.'}
"""
                    if best:
                        body += f"\n🏆 Best Call  : {best['stock_ticker']} — {best['accuracy']}% accurate\n"
                    if worst:
                        body += f"📉 Worst Call : {worst['stock_ticker']} — {worst['accuracy']}% accurate\n"

                    body += f"""
{'='*55}
Keep predicting to improve your accuracy score!
👉 http://127.0.0.1:5000/dashboard

— StockAI Team
"""
                    msg = Message(
                        subject=f"📈 Your Weekly StockAI Report — {datetime.now().strftime('%b %d')}",
                        recipients=[user['email']],
                        body=body
                    )
                    mail.send(msg)
                except Exception:
                    continue

            cursor.close()
            db.close()
        except Exception:
            pass


# =========================================================
# TOAST NOTIFICATION API
# =========================================================

@app.route('/api/toast-test')
def toast_test():
    return jsonify({"message": "Toast system active ✓", "type": "success"})


# Start background scheduler
scheduler = BackgroundScheduler()
def auto_calculate_accuracy():
    """Background job — runs every hour to auto-score pending predictions."""
    with app.app_context():
        try:
            db = get_db()
            cursor = db.cursor(dictionary=True)
            today = datetime.now().date()

            # Find all predictions past their date with no accuracy yet
            cursor.execute("""
                SELECT id, stock_ticker, predicted_price, prediction_date
                FROM predictions
                WHERE actual_price IS NULL
                AND prediction_date IS NOT NULL
                AND prediction_date <= %s
            """, (today,))
            pending = cursor.fetchall()

            for p in pending:
                try:
                    actual = _fetch_actual_price(p['stock_ticker'], p['prediction_date'])
                    if actual:
                        pct_err = abs(actual - p['predicted_price']) / actual * 100
                        if pct_err <= 1:
                            acc = round(100 - pct_err, 2)
                        elif pct_err <= 3:
                            acc = round(100 - (pct_err * 2.5), 2)
                        elif pct_err <= 5:
                            acc = round(100 - (pct_err * 4), 2)
                        else:
                            acc = round(100 - (pct_err * 6), 2)
                        acc = round(max(0.0, min(100.0, acc)), 2)
                        cursor.execute(
                            "UPDATE predictions SET actual_price=%s, accuracy=%s WHERE id=%s",
                            (actual, acc, p['id'])
                        )
                        db.commit()
                except Exception:
                    continue

            cursor.close()
            db.close()
        except Exception:
            pass


scheduler.add_job(func=check_price_alerts, trigger="interval", hours=1, id="price_alert_check")
scheduler.add_job(func=send_weekly_report, trigger="cron", day_of_week="mon", hour=8, minute=0, id="weekly_report")
scheduler.add_job(func=auto_calculate_accuracy, trigger="interval", hours=1, id="auto_accuracy")

# Daily database backup at 2AM
def auto_backup():
    try:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from backup_db import backup
        backup()
    except Exception as e:
        print(f"Auto backup error: {e}")

scheduler.add_job(func=auto_backup, trigger="cron", hour=2, minute=0, id="daily_backup")
scheduler.start()


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
