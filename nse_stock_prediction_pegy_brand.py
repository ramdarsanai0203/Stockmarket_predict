# NSE/YFinance PEGY updater with Normalized Brand Value + Quarterly Gross Profit
# Reads: CM_52_wk_High_low_26082025.xlsx
# Writes: CM_52_wk_High_low_26082025_PE_Filled_v6.xlsx

import time
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from sklearn.linear_model import LinearRegression

INPUT_FILE = "CM_52_wk_High_low_09092025.xlsx"
OUTPUT_FILE = "CM_52_wk_High_low_09092025_PE_Filled_v1.xlsx"
TICKER_COL = "SYMBOL"
QUARTERLY_COLS = ["Jun 2024", "Sep 2024", "Dec 2024", "Mar 2025", "Jun 2025"]

# ----------------------------- Helpers -----------------------------

def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def get_nifty_pe_via_nse():
    """Fetch NIFTY 50 PE from NSE API."""
    url = "https://www.nseindia.com/api/allIndices"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        with requests.Session() as s:
            s.headers.update(headers)
            r = s.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "data" in data:
                for idx in data["data"]:
                    if idx.get("index", "").strip().upper() in ("NIFTY 50","NIFTY50"):
                        return safe_float(idx.get("pe"))
    except Exception as e:
        print("⚠️ NSE API failed:", e)
    return None

def get_nifty_pe_via_yf():
    try:
        idx = yf.Ticker("^NSEI")
        info = idx.get_info()
        return safe_float(info.get("trailingPE"))
    except Exception as e:
        print("⚠️ yfinance NIFTY failed:", e)
    return None

def yf_fetch_price_pe_div_brand(symbol):
    """Fetch price, PE, dividend yield (%), and marketCap (brand proxy)."""
    sym_ns = symbol.upper()
    if not sym_ns.endswith(".NS"):
        sym_ns += ".NS"
    price, pe, div_yield, brand_value = None, None, None, None
    try:
        t = yf.Ticker(sym_ns)
        # Price
        fi = getattr(t, "fast_info", None)
        if fi:
            if isinstance(fi, dict):
                price = fi.get("last_price") or fi.get("lastClose")
            else:
                price = getattr(fi, "last_price", None) or getattr(fi, "lastClose", None)
        if price is None:
            hist = t.history(period="10d", interval="1d", actions=False)
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        # Fundamentals
        info = t.get_info()
        pe = safe_float(info.get("trailingPE"))
        dy = info.get("dividendYield")
        if dy is not None:
            div_yield = safe_float(dy) * 100
        mc = info.get("marketCap")
        brand_value = safe_float(mc)
    except Exception as e:
        print(f"⚠️ yfinance failed for {symbol}: {e}")
    return price, pe, div_yield, brand_value

def get_quarterly_gross_profit(symbol):
    """Fetch last 5 quarters of Gross Profit for ticker."""
    sym_ns = symbol.upper()
    if not sym_ns.endswith(".NS"):
        sym_ns += ".NS"
    try:
        t = yf.Ticker(sym_ns)
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and "Gross Profit" in qf.index:
            gp = qf.loc["Gross Profit"].to_dict()
            result = []
            for col in QUARTERLY_COLS:
                val = None
                for k, v in gp.items():
                    if isinstance(k, pd.Timestamp) and k.strftime("%b %Y") == col:
                        val = safe_float(v)
                result.append(val)
            return result
    except Exception as e:
        print(f"⚠️ Gross Profit fetch failed for {symbol}: {e}")
    return [np.nan]*len(QUARTERLY_COLS)

def compute_growth(values):
    vals = [v for v in values if pd.notna(v)]
    if len(vals) < 2: return None
    first, last = vals[0], vals[-1]
    n_quarters = len(vals)
    years = (n_quarters - 1) / 4.0
    if years > 0 and first > 0 and last > 0:
        return ((last/first)**(1/years)-1)*100
    return (last-first)/abs(first)*100 if first != 0 else None

def predict_next(values):
    vals = [v for v in values if pd.notna(v)]
    if not vals: return None
    if len(vals) < 3: return vals[-1]
    X = np.arange(len(vals)).reshape(-1,1)
    y = np.array(vals).reshape(-1,1)
    model = LinearRegression().fit(X,y)
    return float(model.predict([[len(vals)]])[0][0])

# ----------------------------- Main -----------------------------

def main():
    df = pd.read_excel(INPUT_FILE, engine="openpyxl")

    # Ensure columns exist
    for c in QUARTERLY_COLS:
        if c not in df.columns:
            df[c] = np.nan
    required = ["CA_CurrentPrice","CB_PE","CC_SectorPE","PEGY_Ratio",
                "BZ_BuySellSignal","Predicted Stock value","Brand_Value"]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    nifty_pe = get_nifty_pe_via_nse() or get_nifty_pe_via_yf()
    print("NIFTY PE:", nifty_pe)

    all_pes, all_brands, cache = [], [], {}
    for sym in df[TICKER_COL].astype(str):
        price, pe, div_yield, brand = yf_fetch_price_pe_div_brand(sym)
        cache[sym] = (price, pe, div_yield, brand)
        if pe: all_pes.append(pe)
        if brand: all_brands.append(brand)
        time.sleep(0.1)

    if nifty_pe is None and all_pes:
        nifty_pe = float(np.nanmedian(all_pes))

    # Normalize Brand Values to 0–100
    brand_min, brand_max = (min(all_brands), max(all_brands)) if all_brands else (None, None)
    def normalize_brand(val):
        if brand_min is None or brand_max is None or val is None: return None
        if brand_max == brand_min: return 100.0
        return (val - brand_min) / (brand_max - brand_min) * 100

    for sym in df[TICKER_COL].astype(str):
        price, pe, div_yield, brand = cache.get(sym, (None,None,None,None))

        # Fetch and fill Gross Profit values
        qvals = get_quarterly_gross_profit(sym)
        df.loc[df[TICKER_COL]==sym, QUARTERLY_COLS] = qvals

        # Compute growth from gross profit
        growth = compute_growth(qvals)
        pred_val = predict_next(qvals)

        pe_val = safe_float(pe)
        nifty_val = safe_float(nifty_pe)

        # Always compute PEGY if PE exists
        if pe_val is not None:
            g = growth if growth is not None else 0.0
            d = div_yield if div_yield is not None else 0.01
            denom = g + d
            pegy = pe_val / denom if denom != 0 else np.nan
        else:
            pegy = np.nan

        norm_brand = normalize_brand(brand)

        # Decision logic: PEGY + Brand value
        if pegy is not None and not np.isnan(pegy):
            if pegy < 1:
                signal = "BUY" if (norm_brand is None or norm_brand > 30) else "HOLD"
            elif pegy < 2:
                if norm_brand and norm_brand >= 50: signal = "BUY"
                elif norm_brand and norm_brand <= 30: signal = "SELL"
                else: signal = "HOLD"
            else:
                signal = "SELL" if (norm_brand is None or norm_brand < 50) else "HOLD"
        elif pe_val is not None and nifty_val is not None:
            signal = "BUY" if pe_val < nifty_val else "SELL"
        else:
            signal = "HOLD"

        # Write row
        df.loc[df[TICKER_COL]==sym,
               ["CA_CurrentPrice","CB_PE","CC_SectorPE","PEGY_Ratio",
                "BZ_BuySellSignal","Predicted Stock value","Brand_Value"]] = [
            price, pe_val, nifty_val, pegy, signal, pred_val, norm_brand
        ]

    df.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
    print("✅ Saved to", OUTPUT_FILE)

if __name__=="__main__":
    main()
