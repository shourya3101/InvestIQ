import sys
import os

# ===== STARTUP INFO =====
print("=" * 60)
print("Script started")
print("=" * 60)
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")
print("=" * 60)

# ===== STEP 1: IMPORT =====
print("\n[STEP 1] About to import MarketDataManager and get_market_data...")
try:
    from market_data_manager import MarketDataManager, get_market_data
    print("[STEP 1] SUCCESS: Import completed")
except Exception as e:
    print(f"[STEP 1] ERROR: Import failed - {e}")
    raise

# ===== STEP 2: CREATE INSTANCE =====
print("\n[STEP 2] About to create MarketDataManager instance...")
try:
    mdm = MarketDataManager()
    print("[STEP 2] SUCCESS: MarketDataManager instance created")
except Exception as e:
    print(f"[STEP 2] ERROR: Failed to create instance - {e}")
    raise

# ===== STEP 3: LIVE MODE - FETCH DATA =====
print("\n[STEP 3] About to fetch live data from yfinance (ticker=AAPL)...")
try:
    df_live = get_market_data(mdm, mode="live", ticker="AAPL")
    print("[STEP 3] SUCCESS: Live data fetched successfully")
    print("LIVE MODE:", df_live.head(), "\n")
except Exception as e:
    print(f"[STEP 3] ERROR: Failed to fetch live data - {e}")
    raise

# ===== STEP 4: SAVE CSV =====
print("\n[STEP 4] About to save CSV file (sample_bloomberg_export.csv)...")
try:
    df_live.to_csv("sample_bloomberg_export.csv", index=False)
    print("[STEP 4] SUCCESS: CSV file saved")
except Exception as e:
    print(f"[STEP 4] ERROR: Failed to save CSV - {e}")
    raise

# ===== STEP 5: OFFLINE MODE - LOAD CSV =====
print("\n[STEP 5] About to load CSV file in offline mode...")
try:
    df_offline = get_market_data(mdm, mode="offline", filepath="sample_bloomberg_export.csv")
    print("[STEP 5] SUCCESS: CSV file loaded successfully")
    print("OFFLINE MODE:", df_offline.head())
except Exception as e:
    print(f"[STEP 5] ERROR: Failed to load CSV - {e}")
    raise

print("\n" + "=" * 60)
print("Script completed successfully!")
print("=" * 60)
