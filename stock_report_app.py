# stock_report_app.py
# Run with: streamlit run stock_report_app.py

import pandas as pd
import streamlit as st

# Load your Excel file
FILE_PATH = "CM_52_wk_High_low_09092025_PE_Filled_v1.xlsx"
df = pd.read_excel(FILE_PATH, engine="openpyxl")

# Function to generate report
def stock_report(symbol: str):
    row = df[df["SYMBOL"].str.upper() == symbol.upper()]
    if row.empty:
        return f"❌ Symbol '{symbol}' not found in the dataset."
    row = row.iloc[0].to_dict()

    report = f"📊 **Stock Report for {row['SYMBOL']}**\n\n"
    report += f"- Series: {row['SERIES']}\n"
    report += f"- 52 Week High: {row['Adjusted_52_Week_High']} (Date: {row['52_Week_High_Date']})\n"
    report += f"- 52 Week Low: {row['Adjusted_52_Week_Low']} (Date: {row['52_Week_Low_DT']})\n\n"

    report += "🧾 **Quarterly Gross Profit**\n"
    for q in ["Jun 2024","Sep 2024","Dec 2024","Mar 2025","Jun 2025"]:
        report += f"- {q}: {row[q]}\n"

    report += "\n💰 **Financials**\n"
    report += f"- Current Price: {row['CA_CurrentPrice']}\n"
    report += f"- PE Ratio: {row['CB_PE']}\n"
    report += f"- Sector PE (NIFTY 50): {row['CC_SectorPE']}\n"
    report += f"- PEGY Ratio: {row['PEGY_Ratio']}\n"
    report += f"- Buy/Sell Signal: {row['BZ_BuySellSignal']}\n"
    report += f"- Predicted Stock Value: {row['Predicted Stock value']}\n"
    report += f"- Brand Value: {row['Brand_Value']}\n"

    return report

# ---------------- Streamlit UI ----------------

st.title("📊 Stock Report Viewer")

# All available symbols
all_symbols = sorted(df["SYMBOL"].dropna().unique())

# Search filter box
search_text = st.text_input("🔎 Search symbol:")
if search_text:
    filtered_symbols = [s for s in all_symbols if search_text.upper() in s.upper()]
else:
    filtered_symbols = all_symbols

# Dropdown of filtered results
if filtered_symbols:
    selected_symbol = st.selectbox("Choose a Stock Symbol:", filtered_symbols)
    if selected_symbol:
        report = stock_report(selected_symbol)
        st.markdown(report)
else:
    st.warning("No symbols match your search.")
