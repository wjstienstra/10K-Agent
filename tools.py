from sec_edgar_downloader import Downloader
import os

def fetch_multi_year_10k(ticker, amount=3):
    """
    Downloadt de laatste X jaarverslagen (10-K) van de SEC EDGAR database.
    """
    # De SEC vereist een identificatie (Naam + Email)
    dl = Downloader("MijnBeleggingsApp", "jouw@email.com")
    
    print(f"📥 Downloaden van de laatste {amount} 10-K's voor {ticker}...")
    
    # Download de rapporten. Dit maakt een map: sec-edgar-filings/{ticker}/10-K/
    dl.get("10-K", ticker, limit=amount, download_details=True)
    
    # We geven het pad terug waar de bestanden staan voor de ingestion
    target_dir = os.path.join("sec-edgar-filings", ticker, "10-K")
    return target_dir

import requests
import pandas as pd
import streamlit as st

@st.cache_data(ttl=86400) # Cache voor 24 uur
def get_all_sec_tickers():
    url = "https://www.sec.gov/files/company_tickers.json"
    # De SEC vereist een User-Agent header, anders krijg je een 403
    headers = {"User-Agent": "MijnBeleggingsApp (mijn@email.nl)"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # Transformeer JSON naar een lijst van strings: "AAPL (Apple Inc.)"
        ticker_list = [
            f"{v['ticker']} - {v['title']}" 
            for k, v in data.items()
        ]
        return sorted(ticker_list)
    return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"] # Fallback