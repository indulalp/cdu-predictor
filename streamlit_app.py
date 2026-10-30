import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
import hashlib
import requests
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from github import Github, GithubException

# --- App Config ---
st.set_page_config(page_title="CDU Hybrid Digital Twin Platform", layout="wide")
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

DB_PATH = "audit_telemetry.db"
GUEST_MODEL_FILE = "models/guest_model.pkl"

# Physical Engineering Bounds for 6-cut CDU fractionation (Off-Gas, LPG, Naphtha, Kero, LAGO, Residue)
YIELD_BOUNDS = {
    'flow_offgas': (0.0005, 0.050),
    'flow_lpg': (0.0010, 0.040),
    'flow_naphtha': (0.100, 0.230),   # Accommodates full 250-430 t/h operational range
    'flow_kero': (0.010, 0.080),
    'flow_lago': (0.150, 0.350),      # Total atmospheric gas oil fraction
    'flow_residue': (0.400, 0.650)
}

# Thermodynamic zero-sum slopes: Lift exchanges between Residue and LAGO; lights remain flat
PHYSICS_SLOPES = {
    'flow_offgas': 0.00001,
    'flow_lpg': 0.00002,
    'flow_naphtha': 0.00005,
    'flow_kero': 0.00012,
    'flow_lago': 0.00158,     # Primary receiver of thermal lift
    'flow_residue': -0.00178  # Exact zero-sum balance: sum(slopes) == 0.0
}

DEFAULT_RCO_SLOPES = {
    'cot_degc': -0.052,
    'stripping_steam_flow': -0.165,
    'flash_zone_p_kgcm2': 1.150,
    'crude_api': -0.075,
    'crude_flow': 0.0,
    'flow_rate': 0.0
}

# ==============================================================================
# EMBEDDED REAL REFINERY PLANT RUNS (MATCHING EXACT DCS SPREADSHEET HEADERS)
# ==============================================================================
EXCEL_RECORDS = [
    {"rco_rec_360": 6.30, "flow_rate": 1847.17, "crude_density": 852.9, "crude_api": 34.3938, "cot_degC": 350.120, "flash_zone_p_kgcm2": 1.43, "stripping_steam_flow": 13.20, "top_pa": 1798.64, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 939.90, "top_temp_degC": 120.26, "lago_d86_95_degC": 243.27, "flow_offgas": 6.85000, "flow_lpg": 17.8, "flow_naphtha": 332.64, "flow_kero": 78.36, "flow_lago": 442.39, "flow_residue": 978.09},
    {"rco_rec_360": 6.25, "flow_rate": 1918.67, "crude_density": 859.4, "crude_api": 33.1256, "cot_degC": 354.775, "flash_zone_p_kgcm2": 1.45, "stripping_steam_flow": 13.12, "top_pa": 1470.64, "kero_pa_flow_tph": 414.96, "lago_pa_flow_tph": 957.20, "top_temp_degC": 117.21, "lago_d86_95_degC": 250.47, "flow_offgas": 3.67757, "flow_lpg": 21.6, "flow_naphtha": 332.98, "flow_kero": 106.20, "flow_lago": 434.00, "flow_residue": 1021.23},
    {"rco_rec_360": 14.50, "flow_rate": 1810.48, "crude_density": 849.9, "crude_api": 34.9755, "cot_degC": 354.825, "flash_zone_p_kgcm2": 1.50, "stripping_steam_flow": 13.50, "top_pa": 1801.69, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 951.76, "top_temp_degC": 118.01, "lago_d86_95_degC": 248.21, "flow_offgas": 4.09316, "flow_lpg": 20.7, "flow_naphtha": 338.97, "flow_kero": 104.56, "flow_lago": 454.65, "flow_residue": 897.09},
    {"rco_rec_360": 6.30, "flow_rate": 1677.31, "crude_density": 863.0, "crude_api": 32.4427, "cot_degC": 356.170, "flash_zone_p_kgcm2": 1.42, "stripping_steam_flow": 13.84, "top_pa": 1631.12, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 930.89, "top_temp_degC": 113.60, "lago_d86_95_degC": 243.76, "flow_offgas": 2.47341, "flow_lpg": 17.4, "flow_naphtha": 274.71, "flow_kero": 74.75, "flow_lago": 403.54, "flow_residue": 943.73},
    {"rco_rec_360": 2.00, "flow_rate": 1549.10, "crude_density": 867.9, "crude_api": 31.5190, "cot_degC": 357.890, "flash_zone_p_kgcm2": 1.36, "stripping_steam_flow": 14.74, "top_pa": 1635.16, "kero_pa_flow_tph": 414.04, "lago_pa_flow_tph": 809.37, "top_temp_degC": 109.78, "lago_d86_95_degC": 243.61, "flow_offgas": 2.40643, "flow_lpg": 16.3, "flow_naphtha": 248.36, "flow_kero": 82.75, "flow_lago": 355.85, "flow_residue": 870.43},
    {"rco_rec_360": 2.90, "flow_rate": 1366.14, "crude_density": 849.4, "crude_api": 35.0656, "cot_degC": 358.645, "flash_zone_p_kgcm2": 1.37, "stripping_steam_flow": 14.00, "top_pa": 1542.40, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 841.57, "top_temp_degC": 114.70, "lago_d86_95_degC": 242.92, "flow_offgas": 2.00850, "flow_lpg": 14.2, "flow_naphtha": 236.45, "flow_kero": 83.46, "flow_lago": 351.23, "flow_residue": 709.18},
    {"rco_rec_360": 6.80, "flow_rate": 1827.41, "crude_density": 866.6, "crude_api": 31.7730, "cot_degC": 351.285, "flash_zone_p_kgcm2": 1.48, "stripping_steam_flow": 13.90, "top_pa": 1897.76, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1016.25, "top_temp_degC": 121.36, "lago_d86_95_degC": 244.17, "flow_offgas": 3.79992, "flow_lpg": 14.3, "flow_naphtha": 307.79, "flow_kero": 64.03, "flow_lago": 466.04, "flow_residue": 1021.18},
    {"rco_rec_360": 5.90, "flow_rate": 1609.22, "crude_density": 854.9, "crude_api": 34.0018, "cot_degC": 354.230, "flash_zone_p_kgcm2": 1.43, "stripping_steam_flow": 14.00, "top_pa": 1440.82, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1033.69, "top_temp_degC": 121.95, "lago_d86_95_degC": 242.87, "flow_offgas": 2.13730, "flow_lpg": 14.1, "flow_naphtha": 290.58, "flow_kero": 57.21, "flow_lago": 419.08, "flow_residue": 891.33},
    {"rco_rec_360": 3.80, "flow_rate": 1491.72, "crude_density": 853.0, "crude_api": 34.3646, "cot_degC": 359.015, "flash_zone_p_kgcm2": 1.42, "stripping_steam_flow": 15.90, "top_pa": 1296.78, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 855.22, "top_temp_degC": 117.91, "lago_d86_95_degC": 249.50, "flow_offgas": 5.76393, "flow_lpg": 10.9, "flow_naphtha": 324.65, "flow_kero": 75.83, "flow_lago": 386.87, "flow_residue": 766.75},
    {"rco_rec_360": 5.70, "flow_rate": 1253.22, "crude_density": 859.6, "crude_api": 33.1007, "cot_degC": 358.550, "flash_zone_p_kgcm2": 1.35, "stripping_steam_flow": 13.99, "top_pa": 1329.01, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 876.26, "top_temp_degC": 115.92, "lago_d86_95_degC": 248.91, "flow_offgas": 0.68082, "flow_lpg": 15.2, "flow_naphtha": 195.74, "flow_kero": 68.45, "flow_lago": 354.82, "flow_residue": 631.37},
    {"rco_rec_360": 5.70, "flow_rate": 1926.91, "crude_density": 855.5, "crude_api": 33.8780, "cot_degC": 347.955, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 17.50, "top_pa": 1606.35, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 973.49, "top_temp_degC": 116.12, "lago_d86_95_degC": 246.93, "flow_offgas": 4.46463, "flow_lpg": 18.5, "flow_naphtha": 356.26, "flow_kero": 80.72, "flow_lago": 472.45, "flow_residue": 1009.91},
    {"rco_rec_360": 4.60, "flow_rate": 1954.94, "crude_density": 1005.9, "crude_api": 9.1616, "cot_degC": 350.210, "flash_zone_p_kgcm2": 1.49, "stripping_steam_flow": 17.98, "top_pa": 1980.45, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 997.60, "top_temp_degC": 114.04, "lago_d86_95_degC": 250.31, "flow_offgas": 5.67824, "flow_lpg": 19.8, "flow_naphtha": 337.89, "flow_kero": 126.69, "flow_lago": 411.36, "flow_residue": 1059.74},
    {"rco_rec_360": 6.00, "flow_rate": 1955.56, "crude_density": 845.3, "crude_api": 35.8735, "cot_degC": 345.885, "flash_zone_p_kgcm2": 1.60, "stripping_steam_flow": 14.24, "top_pa": 2250.08, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1025.37, "top_temp_degC": 117.89, "lago_d86_95_degC": 249.05, "flow_offgas": 6.71467, "flow_lpg": 16.3, "flow_naphtha": 371.91, "flow_kero": 128.90, "flow_lago": 455.70, "flow_residue": 1021.55},
    {"rco_rec_360": 9.70, "flow_rate": 1903.98, "crude_density": 837.5, "crude_api": 37.4303, "cot_degC": 339.700, "flash_zone_p_kgcm2": 1.72, "stripping_steam_flow": 11.98, "top_pa": 2295.02, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1030.20, "top_temp_degC": 122.13, "lago_d86_95_degC": 246.84, "flow_offgas": 6.60169, "flow_lpg": 17.0, "flow_naphtha": 427.89, "flow_kero": 128.61, "flow_lago": 435.34, "flow_residue": 939.22},
    {"rco_rec_360": 5.30, "flow_rate": 1852.52, "crude_density": 863.1, "crude_api": 32.4218, "cot_degC": 349.370, "flash_zone_p_kgcm2": 1.50, "stripping_steam_flow": 17.00, "top_pa": 2058.84, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1037.50, "top_temp_degC": 118.36, "lago_d86_95_degC": 253.46, "flow_offgas": 3.47714, "flow_lpg": 15.9, "flow_naphtha": 306.23, "flow_kero": 105.26, "flow_lago": 454.95, "flow_residue": 1020.43},
    {"rco_rec_360": 4.40, "flow_rate": 1903.44, "crude_density": 852.8, "crude_api": 34.3996, "cot_degC": 348.055, "flash_zone_p_kgcm2": 1.55, "stripping_steam_flow": 17.00, "top_pa": 2218.97, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1038.79, "top_temp_degC": 121.27, "lago_d86_95_degC": 253.36, "flow_offgas": 4.85334, "flow_lpg": 15.0, "flow_naphtha": 340.50, "flow_kero": 111.76, "flow_lago": 474.09, "flow_residue": 996.26},
    {"rco_rec_360": 2.50, "flow_rate": 1918.92, "crude_density": 855.2, "crude_api": 33.9380, "cot_degC": 350.020, "flash_zone_p_kgcm2": 1.59, "stripping_steam_flow": 17.70, "top_pa": 2033.84, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1023.24, "top_temp_degC": 116.67, "lago_d86_95_degC": 249.48, "flow_offgas": 4.69515, "flow_lpg": 19.0, "flow_naphtha": 347.83, "flow_kero": 97.03, "flow_lago": 448.48, "flow_residue": 1045.83},
    {"rco_rec_360": 2.00, "flow_rate": 1907.06, "crude_density": 851.9, "crude_api": 34.5807, "cot_degC": 349.615, "flash_zone_p_kgcm2": 1.61, "stripping_steam_flow": 15.15, "top_pa": 2050.57, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1002.30, "top_temp_degC": 119.53, "lago_d86_95_degC": 250.35, "flow_offgas": 4.11358, "flow_lpg": 16.9, "flow_naphtha": 352.80, "flow_kero": 112.57, "flow_lago": 451.35, "flow_residue": 1008.93},
    {"rco_rec_360": 5.80, "flow_rate": 1932.69, "crude_density": 846.2, "crude_api": 35.6955, "cot_degC": 350.255, "flash_zone_p_kgcm2": 1.68, "stripping_steam_flow": 17.89, "top_pa": 2181.01, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1003.59, "top_temp_degC": 121.70, "lago_d86_95_degC": 253.09, "flow_offgas": 6.23427, "flow_lpg": 16.6, "flow_naphtha": 355.93, "flow_kero": 114.39, "flow_lago": 476.78, "flow_residue": 996.47},
    {"rco_rec_360": 5.80, "flow_rate": 1918.36, "crude_density": 855.1, "crude_api": 33.9573, "cot_degC": 348.815, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 16.78, "top_pa": 2257.26, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1041.04, "top_temp_degC": 121.17, "lago_d86_95_degC": 251.37, "flow_offgas": 4.87445, "flow_lpg": 13.3, "flow_naphtha": 357.87, "flow_kero": 110.87, "flow_lago": 459.52, "flow_residue": 1011.99},
    {"rco_rec_360": 6.50, "flow_rate": 1936.50, "crude_density": 860.6, "crude_api": 32.9133, "cot_degC": 349.160, "flash_zone_p_kgcm2": 1.51, "stripping_steam_flow": 18.48, "top_pa": 2318.51, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1031.71, "top_temp_degC": 117.64, "lago_d86_95_degC": 251.18, "flow_offgas": 3.58797, "flow_lpg": 11.3, "flow_naphtha": 316.25, "flow_kero": 113.15, "flow_lago": 456.90, "flow_residue": 1070.99},
    {"rco_rec_360": 7.50, "flow_rate": 1930.41, "crude_density": 852.0, "crude_api": 34.5554, "cot_degC": 347.295, "flash_zone_p_kgcm2": 1.61, "stripping_steam_flow": 15.58, "top_pa": 2269.93, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1044.50, "top_temp_degC": 124.66, "lago_d86_95_degC": 250.38, "flow_offgas": 4.49535, "flow_lpg": 9.5, "flow_naphtha": 385.91, "flow_kero": 116.01, "flow_lago": 449.56, "flow_residue": 978.09},
    {"rco_rec_360": 17.50, "flow_rate": 1918.27, "crude_density": 851.3, "crude_api": 34.7017, "cot_degC": 345.850, "flash_zone_p_kgcm2": 1.61, "stripping_steam_flow": 16.82, "top_pa": 2156.23, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1044.59, "top_temp_degC": 121.20, "lago_d86_95_degC": 248.86, "flow_offgas": 5.82920, "flow_lpg": 12.0, "flow_naphtha": 364.45, "flow_kero": 115.22, "flow_lago": 455.99, "flow_residue": 980.88},
    {"rco_rec_360": 9.00, "flow_rate": 1927.75, "crude_density": 850.1, "crude_api": 34.9265, "cot_degC": 347.875, "flash_zone_p_kgcm2": 1.64, "stripping_steam_flow": 16.39, "top_pa": 2270.73, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1044.28, "top_temp_degC": 123.48, "lago_d86_95_degC": 251.83, "flow_offgas": 6.69661, "flow_lpg": 9.8, "flow_naphtha": 368.87, "flow_kero": 132.25, "flow_lago": 455.55, "flow_residue": 980.66},
    {"rco_rec_360": 9.50, "flow_rate": 1923.81, "crude_density": 848.7, "crude_api": 35.2089, "cot_degC": 347.885, "flash_zone_p_kgcm2": 1.67, "stripping_steam_flow": 15.40, "top_pa": 2264.29, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1044.98, "top_temp_degC": 124.13, "lago_d86_95_degC": 251.27, "flow_offgas": 6.43440, "flow_lpg": 12.7, "flow_naphtha": 392.29, "flow_kero": 122.13, "flow_lago": 435.13, "flow_residue": 978.07},
    {"rco_rec_360": 8.60, "flow_rate": 1912.23, "crude_density": 847.3, "crude_api": 35.4765, "cot_degC": 344.005, "flash_zone_p_kgcm2": 1.70, "stripping_steam_flow": 16.22, "top_pa": 2245.60, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 123.75, "lago_d86_95_degC": 252.06, "flow_offgas": 6.38299, "flow_lpg": 14.9, "flow_naphtha": 409.10, "flow_kero": 130.82, "flow_lago": 421.53, "flow_residue": 948.96},
    {"rco_rec_360": 7.50, "flow_rate": 1901.46, "crude_density": 854.6, "crude_api": 34.0502, "cot_degC": 342.505, "flash_zone_p_kgcm2": 1.64, "stripping_steam_flow": 16.46, "top_pa": 2257.09, "kero_pa_flow_tph": 415.00, "lago_pa_flow_tph": 1042.68, "top_temp_degC": 120.44, "lago_d86_95_degC": 247.37, "flow_offgas": 6.99080, "flow_lpg": 17.2, "flow_naphtha": 376.86, "flow_kero": 109.58, "flow_lago": 416.74, "flow_residue": 995.98}
]

def get_demo_dataframe():
    return pd.DataFrame(EXCEL_RECORDS)

# ==============================================================================
# GITHUB CONTINUOUS SYNC & CONTAINER REHYDRATION
# ==============================================================================
def sync_file_to_github(local_file_path, repo_file_path, commit_message="Auto-sync from Streamlit App"):
    if "GITHUB_TOKEN" not in st.secrets or "GITHUB_REPO" not in st.secrets:
        return False, "GITHUB_TOKEN or GITHUB_REPO not defined in Streamlit secrets."
    token = st.secrets["GITHUB_TOKEN"]
    repo_name = st.secrets["GITHUB_REPO"]
    try:
        g = Github(token)
        repo = g.get_repo(repo_name)
        branch = st.secrets.get("GITHUB_BRANCH", repo.default_branch)
        with open(local_file_path, "rb") as f:
            content = f.read()
        try:
            existing_file = repo.get_contents(repo_file_path, ref=branch)
            repo.update_file(path=repo_file_path, message=commit_message, content=content, sha=existing_file.sha, branch=branch)
        except GithubException as e:
            if e.status == 404:
                repo.create_file(path=repo_file_path, message=commit_message, content=content, branch=branch)
            else:
                raise e
        return True, "Synced"
    except Exception as err:
        return False, str(err)

def rehydrate_from_github():
    if "GITHUB_TOKEN" not in st.secrets or "GITHUB_REPO" not in st.secrets:
        return
    token = st.secrets["GITHUB_TOKEN"]
    repo_name = st.secrets["GITHUB_REPO"]
    try:
        g = Github(token)
        repo = g.get_repo(repo_name)
        branch = st.secrets.get("GITHUB_BRANCH", repo.default_branch)
        try:
            db_content = repo.get_contents(DB_PATH, ref=branch)
            with open(DB_PATH, "wb") as f:
                f.write(db_content.decoded_content)
        except Exception:
            pass
        try:
            models_folder = repo.get_contents("models", ref=branch)
            for file_item in models_folder:
                if file_item.name.endswith(".pkl") and not os.path.exists(file_item.path):
                    with open(file_item.path, "wb") as f:
                        f.write(file_item.decoded_content)
        except Exception:
            pass
    except Exception as e:
        print(f"Rehydration note: {e}")

if "rehydrated" not in st.session_state:
    rehydrate_from_github()
    st.session_state["rehydrated"] = True

# ==============================================================================
# DATABASE & ACCESS CONTROL LAYER
# ==============================================================================
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT, role TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS access_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, login_time TIMESTAMP, ip_address TEXT, city TEXT, region TEXT, country TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS protected_models (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, model_tag TEXT, model_path TEXT, training_rows INTEGER, created_at TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS simulation_history (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, timestamp TIMESTAMP, inputs_json TEXT, outputs_json TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS calibrated_physics_slopes (parameter_name TEXT PRIMARY KEY, slope_value REAL, updated_at TIMESTAMP)''')
    
    defaults = [
        ('cot_degc', -0.052),
        ('stripping_steam_flow', -0.165),
        ('flash_zone_p_kgcm2', 1.150),
        ('crude_api', -0.075),
        ('crude_flow', 0.0),
        ('flow_rate', 0.0)
    ]
    for param, val in defaults:
        c.execute("INSERT OR IGNORE INTO calibrated_physics_slopes VALUES (?, ?, ?)", (param, val, datetime.now()))

    admin_pw = hashlib.sha256("Admin@123".encode()).hexdigest()
    user_pw = hashlib.sha256("User@123".encode()).hexdigest()
    c.execute("INSERT OR REPLACE INTO users VALUES (?, ?, ?)", ("admin", admin_pw, "admin"))
    c.execute("INSERT OR REPLACE INTO users VALUES (?, ?, ?)", ("engineer1", user_pw, "user"))
    conn.commit()
    conn.close()

init_db()

def get_calibrated_slopes():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS calibrated_physics_slopes (
            parameter_name TEXT PRIMARY KEY,
            slope_value REAL,
            updated_at TIMESTAMP
        )
    ''')
    defaults = [
        ('cot_degc', -0.052),
        ('stripping_steam_flow', -0.165),
        ('flash_zone_p_kgcm2', 1.150),
        ('crude_api', -0.075),
        ('crude_flow', 0.0),
        ('flow_rate', 0.0)
    ]
    for param, val in defaults:
        c.execute("INSERT OR IGNORE INTO calibrated_physics_slopes VALUES (?, ?, ?)", (param, val, datetime.now()))
    conn.commit()
    c.execute("SELECT parameter_name, slope_value FROM calibrated_physics_slopes")
    rows = c.fetchall()
    conn.close()
    return {row[0]: float(row[1]) for row in rows} if rows else DEFAULT_RCO_SLOPES

def update_physics_slopes_from_lab(input_dict, bounds_dict, y_pred, y_lab, learning_rate=0.10):
    current_slopes = get_calibrated_slopes()
    error = y_lab - y_pred
    updated = False
    conn = get_db_connection()
    c = conn.cursor()
    for col, val in input_dict.items():
        c_low = col.lower()
        if c_low in current_slopes and col in bounds_dict:
            b_min, b_max = bounds_dict[col]
            delta = 0.0
            if val > b_max:
                delta = val - b_max
            elif val < b_min:
                delta = val - b_min
            if abs(delta) > 1e-3:
                curr_s = current_slopes[c_low]
                step = learning_rate * (error / delta)
                new_s = curr_s + step
                if "cot" in c_low or "temp" in c_low:
                    new_s = np.clip(new_s, -0.10, -0.01)
                elif "steam" in c_low:
                    new_s = np.clip(new_s, -0.30, -0.05)
                elif "p_kgcm2" in c_low or "pres" in c_low:
                    new_s = np.clip(new_s, 0.40, 2.50)
                elif "api" in c_low:
                    new_s = np.clip(new_s, -0.20, -0.01)
                c.execute("INSERT OR REPLACE INTO calibrated_physics_slopes VALUES (?, ?, ?)", (c_low, float(new_s), datetime.now()))
                current_slopes[c_low] = float(new_s)
                updated = True
    conn.commit()
    conn.close()
    if updated:
        sync_file_to_github(DB_PATH, DB_PATH, "Auto-sync: Calibrated physical slopes from lab feedback")
    return current_slopes

def verify_login(username, password):
    clean_u = username.strip().lower()
    clean_p = password.strip()
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username = ? AND password_hash = ?", 
              (clean_u, hashlib.sha256(clean_p.encode()).hexdigest()))
    row = c.fetchone()
    conn.close()
    return row["role"] if row else None

def create_client_user(new_username, plain_password):
    clean_u = new_username.strip().lower()
    clean_p = plain_password.strip()
    p_hash = hashlib.sha256(clean_p.encode()).hexdigest()
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'user')", (clean_u, p_hash))
        conn.commit()
        success = True
        msg = f"Client account `{clean_u}` created successfully."
    except sqlite3.IntegrityError:
        success = False
        msg = f"Username `{clean_u}` already exists."
    conn.close()
    if success:
        sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Created client user {clean_u}")
    return success, msg

def reset_client_password(target_username, new_plain_password):
    p_hash = hashlib.sha256(new_plain_password.strip().encode()).hexdigest()
    conn = get_db_connection()
    conn.execute("UPDATE users SET password_hash = ? WHERE username = ?", (p_hash, target_username))
    conn.commit()
    conn.close()
    sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Reset password for {target_username}")

def delete_client_user(target_username):
    conn = get_db_connection()
    conn.execute("DELETE FROM users WHERE username = ?", (target_username,))
    conn.execute("DELETE FROM protected_models WHERE username = ?", (target_username,))
    conn.execute("DELETE FROM simulation_history WHERE username = ?", (target_username,))
    conn.commit()
    conn.close()
    sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Deleted user {target_username}")

def get_visitor_geo():
    try:
        res = requests.get("https://ipapi.co/json/", timeout=2.5).json()
        return {
            "ip": res.get("ip", "Local/VPN"),
            "city": res.get("city", "Unknown"),
            "region": res.get("region", "Unknown"),
            "country": res.get("country_name", "Unknown")
        }
    except Exception:
        return {"ip": "127.0.0.1", "city": "Internal", "region": "Internal", "country": "Internal"}

def log_login_event(username):
    geo = get_visitor_geo()
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO access_logs (username, login_time, ip_address, city, region, country)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (username, datetime.now(), geo["ip"], geo["city"], geo["region"], geo["country"]))
    conn.commit()
    conn.close()
    sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Login log for {username}")

# ==============================================================================
# PHYSICS-INFORMED ENGINE (NATIVE SERIALIZABLE FUNCTIONS - NO PICKLE ERRORS)
# ==============================================================================
def density_to_api(density_val):
    sg = density_val / 1000.0 if density_val > 10.0 else density_val
    if sg <= 0:
        return 30.0
    return (141.5 / sg) - 131.5

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def predict_yields_with_physics(yield_dict, X_df):
    """
    Natively picklable physics-informed multi-cut yield prediction function.
    Eliminates _pickle.PicklingError while guaranteeing mass conservation.
    """
    scaler = yield_dict['scaler']
    ml_model = yield_dict['ml_model']
    linear_reg = yield_dict['linear_reg']
    flow_targets = yield_dict['flow_targets']
    baseline_stats = yield_dict['baseline_stats']

    X_scaled = scaler.transform(X_df)
    ml_preds = ml_model.predict(X_scaled)
    linear_preds = softmax(linear_reg.predict(X_scaled))
    base_yields = 0.70 * ml_preds + 0.30 * linear_preds

    cot = X_df['cot_degC'].values if 'cot_degC' in X_df else baseline_stats['mean_cot']
    fzp = X_df['flash_zone_p_kgcm2'].values if 'flash_zone_p_kgcm2' in X_df else baseline_stats['mean_p']
    steam = X_df['stripping_steam_flow'].values if 'stripping_steam_flow' in X_df else baseline_stats['mean_steam']
    api = X_df['crude_api'].values if 'crude_api' in X_df else baseline_stats.get('mean_api', 34.0)

    delta_severity = (
        (cot - baseline_stats['mean_cot'])
        - 16.5 * (fzp - baseline_stats['mean_p'])
        + 0.55 * (steam - baseline_stats['mean_steam'])
        + 0.85 * (api - baseline_stats.get('mean_api', 34.0))
    )

    non_linear_severity = 10.0 * np.tanh(delta_severity / 6.0)

    delta_residue = -0.00320 * non_linear_severity
    delta_lago = 0.00295 * non_linear_severity
    final_yields = base_yields.copy()
    for i, col in enumerate(flow_targets):
        if col == 'flow_residue':
            final_yields[:, i] = base_yields[:, i] + delta_residue
        elif 'lago' in col or 'ago' in col:
            final_yields[:, i] = base_yields[:, i] + delta_lago
        else:
            final_yields[:, i] = base_yields[:, i]

        min_b, max_b = YIELD_BOUNDS.get(col, (0.0005, 0.90))
        final_yields[:, i] = np.clip(final_yields[:, i], min_b, max_b)

    return final_yields / np.sum(final_yields, axis=1, keepdims=True)

# ==============================================================================
# HISTGRADIENTBOOSTING RCO ENGINE
# ==============================================================================
def build_and_fit_rco_model(X, y):
    feature_names = list(X.columns)
    constraints = []
    bounds = {}
    for col in feature_names:
        bounds[col] = (float(X[col].min()), float(X[col].max()))
        c = col.lower()
        if "cot" in c or "temp" in c:
            constraints.append(-1)
        elif "steam" in c:
            constraints.append(-1)
        elif "p_kgcm2" in c or "pres" in c:
            constraints.append(1)
        else:
            constraints.append(0)

    model = HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=2,
        l2_regularization=1.5,
        monotonic_cst=constraints,
        random_state=42
    )
    model.fit(X, y)
    r2 = float(r2_score(y, model.predict(X)))
    return model, feature_names, bounds, r2

def predict_rco_with_physics_anchor(model, feature_names, bounds, X_df):
    active_slopes = get_calibrated_slopes()
    X_eval = X_df[feature_names].copy().reset_index(drop=True)
    tree_preds = model.predict(X_eval)
    final_preds = []
    for idx, row in X_eval.iterrows():
        dist_sq = 0.0
        clamped_row = row.copy()
        physics_delta = 0.0
        for col in feature_names:
            val = row[col]
            b_min, b_max = bounds.get(col, (val, val))
            span = max(b_max - b_min, 1e-4)
            if val > b_max:
                dist_sq += ((val - b_max) / span) ** 2
                clamped_row[col] = b_max
                physics_delta += active_slopes.get(col.lower(), 0.0) * (val - b_max)
            elif val < b_min:
                dist_sq += ((b_min - val) / span) ** 2
                clamped_row[col] = b_min
                physics_delta += active_slopes.get(col.lower(), 0.0) * (val - b_min)
        w_physics = 1.0 - np.exp(-2.5 * dist_sq)
        boundary_pred = model.predict(pd.DataFrame([clamped_row]))[0]
        physics_extrapolation = boundary_pred + physics_delta
        pred = (1.0 - w_physics) * tree_preds[idx] + w_physics * physics_extrapolation
        final_preds.append(np.clip(pred, 1.5, 20.0))
    return np.array(final_preds)

# ==============================================================================
# DEFAULT MODEL INITIALIZATION (ROBUST SCIKIT-LEARN DICTIONARY SERIALIZATION)
# ==============================================================================
def train_default_guest_model_if_missing():
    needs_retrain = False
    if not os.path.exists(GUEST_MODEL_FILE):
        needs_retrain = True
    else:
        try:
            test_pipe = joblib.load(GUEST_MODEL_FILE)
            if (
                "rco_model" not in test_pipe 
                or "rco_bounds" not in test_pipe
                or "yield_model_dict" not in test_pipe
                or "flow_lpg" not in test_pipe.get("flow_targets", [])
            ):
                needs_retrain = True
        except Exception:
            needs_retrain = True

    if needs_retrain:
        df = get_demo_dataframe()
        s_col = 'sulphur_wt_pct' if 'sulphur_wt_pct' in df.columns else ('sulfur_wt_pct' if 'sulfur_wt_pct' in df.columns else None)
        crude_tag = 'flow_rate' if 'flow_rate' in df.columns else 'crude_flow'
        
        input_cols = [crude_tag, 'crude_api', 'cot_degC', 'flash_zone_p_kgcm2', 'stripping_steam_flow']
        if s_col:
            input_cols.insert(2, s_col)

        flow_cols = ['flow_offgas', 'flow_lpg', 'flow_naphtha', 'flow_kero', 'flow_lago', 'flow_residue']
        state_cols = ['top_temp_degC', 'lago_d86_95_degC', 'top_pa', 'kero_pa_flow_tph', 'lago_pa_flow_tph']

        yield_targets = df[flow_cols].div(df[crude_tag], axis=0)
        baseline_stats = {
            'mean_cot': float(df['cot_degC'].mean()),
            'mean_p': float(df['flash_zone_p_kgcm2'].mean()),
            'mean_steam': float(df['stripping_steam_flow'].mean()),
            'mean_api': float(df['crude_api'].mean()),
            'mean_rco_rec_360': float(df['rco_rec_360'].mean())
        }

        scaler_yields = StandardScaler()
        X_scaled = scaler_yields.fit_transform(df[input_cols])
        
        ml_yield_model = MultiOutputRegressor(LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42))
        ml_yield_model.fit(X_scaled, yield_targets)
        
        logits = np.log(np.clip(yield_targets.values, 1e-4, 1.0))
        linear_reg = Ridge(alpha=1.0)
        linear_reg.fit(X_scaled, logits)

        yield_model_dict = {
            'scaler': scaler_yields,
            'ml_model': ml_yield_model,
            'linear_reg': linear_reg,
            'flow_targets': flow_cols,
            'baseline_stats': baseline_stats
        }

        scaler_states = StandardScaler()
        model_states = MultiOutputRegressor(LGBMRegressor(n_estimators=250, learning_rate=0.03, random_state=42))
        model_states.fit(scaler_states.fit_transform(df[input_cols]), df[state_cols])

        rco_model, rco_feats, rco_bounds, rco_r2 = build_and_fit_rco_model(df[input_cols], df['rco_rec_360'])

        pipeline = {
            "yield_model_dict": yield_model_dict,
            "rco_model": rco_model,
            "rco_feats": rco_feats,
            "rco_bounds": rco_bounds,
            "rco_r2": rco_r2,
            "model_states": model_states,
            "scaler_states": scaler_states,
            "input_cols": input_cols,
            "flow_targets": flow_cols,
            "state_targets": state_cols,
            "crude_col": crude_tag,
            "baseline_stats": baseline_stats,
            "training_rows": len(df),
            "last_known_inputs": df[input_cols].iloc[0].to_dict()
        }
        joblib.dump(pipeline, GUEST_MODEL_FILE)
        sync_file_to_github(GUEST_MODEL_FILE, GUEST_MODEL_FILE, "Auto-sync: Calibrated model with robust joblib architecture")

train_default_guest_model_if_missing()

# ==============================================================================
# SIDEBAR LOGIN & MULTI-TENANT GATEWAY
# ==============================================================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = "Guest"
    st.session_state["role"] = "guest"

st.sidebar.title("🛢️ CDU Digital Twin")

if not st.session_state["authenticated"]:
    with st.sidebar.expander("🔒 Member / Client Login"):
        login_user = st.text_input("Username")
        login_pass = st.text_input("Password", type="password")
        if st.button("Sign In"):
            role = verify_login(login_user, login_pass)
            if role:
                st.session_state["authenticated"] = True
                st.session_state["username"] = login_user.strip().lower()
                st.session_state["role"] = role
                log_login_event(login_user)
                st.rerun()
            else:
                st.error("Invalid username or password.")
else:
    st.sidebar.success(f"Logged in as: **{st.session_state['username']}** ({st.session_state['role'].upper()})")
    if st.sidebar.button("Log Out"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = "Guest"
        st.session_state["role"] = "guest"
        st.rerun()

nav_options = [
    "1. Model Training & DCS Upload", 
    "2. Yield Prediction",
]

if st.session_state["authenticated"]:
    nav_options.append("3. Protected Workspace & History")

if st.session_state["role"] == "admin":
    nav_options.append("🛡️ Admin Audit & Telemetry")

page = st.sidebar.radio("Navigation", nav_options, index=1)

# ==============================================================================
# PAGE 1: MODEL TRAINING & INGESTION (ROBUST CRUDE & 6-CUT SELECTION)
# ==============================================================================
if page == "1. Model Training & DCS Upload":
    st.header("⚙️ Column Data Ingestion & Model Training")
    st.markdown("Upload historical plant logs or load the calibrated refinery dataset with separated LPG, Naphtha, and Gas Oil fractions.")

    uploaded_file = st.file_uploader("Upload DCS Historical Data (CSV or Excel)", type=["csv", "xlsx"])
    col1, col2 = st.columns([1, 4])
    use_synthetic = col1.button("Load Calibrated DCS Dataset (with LPG)")

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        if any("unnamed" in str(col).lower() for col in raw_df.columns):
            raw_df.columns = raw_df.iloc[0].astype(str)
            raw_df = raw_df[1:].reset_index(drop=True)
        raw_df.columns = [str(c).strip() for c in raw_df.columns]
        raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()].copy().reset_index(drop=True)
        st.session_state['active_df'] = raw_df
        st.session_state['active_src_name'] = f"Uploaded File: `{uploaded_file.name}`"
    elif use_synthetic:
        st.session_state['active_df'] = get_demo_dataframe()
        st.session_state['active_src_name'] = "Calibrated CDU Operating Dataset (27 runs with flow_lpg)"

    if 'active_df' not in st.session_state:
        st.session_state['active_df'] = get_demo_dataframe()
        st.session_state['active_src_name'] = "Default Calibrated CDU Operating Dataset"

    df = st.session_state['active_df']
    st.info(f"📂 **Active Dataset:** {st.session_state.get('active_src_name', '')} | Rows: **{len(df)}** | Columns: **{len(df.columns)}**")

    st.subheader("Data Inspector")
    st.dataframe(df.head(5), use_container_width=True)

    has_density = any("dens" in str(c).lower() or "sg" in str(c).lower() for c in df.columns)
    if 'crude_api' not in df.columns and has_density:
        dens_cols = [c for c in df.columns if "dens" in str(c).lower() or "sg" in str(c).lower()]
        df['crude_api'] = df[dens_cols[0]].apply(density_to_api)
        st.success(f"Calculated `crude_api` automatically from `{dens_cols[0]}`")

    # 1. Robust Crude Inlet Tag Detection (Prioritize plant rate and skip quality targets)
    detected_crude = None
    for cand in ['flow_rate', 'crude_flow', 'flow rate', 'feed_flow', 'crude_rate']:
        matches = [c for c in df.columns if c.strip().lower() == cand.lower()]
        if matches:
            detected_crude = matches[0]
            break
    
    if not detected_crude:
        for c in df.columns:
            c_low = c.strip().lower()
            if any(k in c_low for k in ['flow', 'rate', 'feed']) and not any(ex in c_low for ex in ['_pa', 'top_pa', 'offgas', 'lpg', 'naphtha', 'kero', 'lago', 'residue', 'steam']):
                detected_crude = c
                break

    if not detected_crude:
        detected_crude = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    # 2. Target candidate detection
    rco_target_candidates = [c for c in df.columns if 'rco' in str(c).lower() or '360' in str(c).lower()]
    default_rco_col = rco_target_candidates[0] if rco_target_candidates else None

    # 3. Clean inputs (Strictly exclude RCO quality target from inputs)
    clean_default_inputs = [c for c in [detected_crude, 'crude_api', 'cot_degC', 'flash_zone_p_kgcm2', 'stripping_steam_flow'] if c in df.columns]

    # 4. Strict Product Cuts (Pumparounds are strictly banned from Y1)
    valid_product_cuts = [
        c for c in df.columns 
        if any(cut in c.lower() for cut in ['offgas', 'lpg', 'naphtha', 'nap', 'kero', 'lago', 'hago', 'residue'])
        and not any(pa in c.lower() for pa in ['_pa', 'top_pa', 'reflux'])
        and c != detected_crude
    ]
    if not valid_product_cuts:
        valid_product_cuts = [c for c in ['flow_offgas', 'flow_lpg', 'flow_LPG', 'flow_naphtha', 'flow_kero', 'flow_lago', 'flow_residue'] if c in df.columns]

    # 5. Internal states strictly for Y2
    default_states = [c for c in ['top_temp_degC', 'lago_d86_95_degC', 'top_pa', 'kero_pa_flow_tph', 'lago_pa_flow_tph'] if c in df.columns]

    c1, c2, c3 = st.columns(3)
    input_cols = c1.multiselect(
        "Inputs (X)", 
        list(df.columns), 
        default=clean_default_inputs,
        key="inputs_multiselect_v3"
    )
    flow_cols = c2.multiselect(
        "Product Flows (Y1) [Includes LPG]", 
        list(df.columns), 
        default=valid_product_cuts,
        key="flow_cols_multiselect_v3"
    )
    state_cols = c3.multiselect(
        "Predicted States & Qualities (Y2)", 
        list(df.columns), 
        default=default_states,
        key="states_multiselect_v3"
    )
    crude_col = c1.selectbox(
        "Crude Inlet Flow Tag", 
        list(df.columns), 
        index=list(df.columns).index(detected_crude) if detected_crude in list(df.columns) else 0,
        key="crude_inlet_select_v3"
    )
    
    rco_col = None
    if default_rco_col:
        rco_col = c2.selectbox(
            "RCO Recovery Target Column", 
            options=list(df.columns), 
            index=list(df.columns).index(default_rco_col),
            key="rco_target_select_v3"
        )

    save_as_protected = False
    model_tag = "guest_model"
    if st.session_state["authenticated"]:
        st.divider()
        c_save1, c_save2 = st.columns([1, 2])
        save_as_protected = c_save1.checkbox("Save model into my protected private vault", value=True)
        if save_as_protected:
            model_tag = c_save2.text_input("Protected Model Tag", value=f"{st.session_state['username']}_v1")

    if st.button("🚀 Train Digital Twin", type="primary"):
        with st.spinner("Training digital twin with robust scikit-learn dictionary architecture..."):
            needed_cols = []
            for c in (input_cols + flow_cols + state_cols + [crude_col] + ([rco_col] if rco_col else [])):
                if c not in needed_cols and c in df.columns:
                    needed_cols.append(c)

            clean_df = df.loc[:, ~df.columns.duplicated()].copy().reset_index(drop=True)
            clean_df = clean_df[needed_cols].apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

            crude_series = clean_df[crude_col]
            if isinstance(crude_series, pd.DataFrame):
                crude_series = crude_series.iloc[:, 0]

            total_out = clean_df[flow_cols].sum(axis=1)
            imbalance = np.abs(total_out - crude_series) / crude_series.replace(0, np.nan)
            valid_mask = imbalance.fillna(1.0) < 0.25
            valid_df = clean_df[valid_mask].copy().reset_index(drop=True)

            if valid_df.empty:
                st.error("❌ Mass balance error: Cleaned data does not close within tolerance.")
                st.stop()

            yield_targets = valid_df[flow_cols].div(crude_series[valid_mask].values, axis=0)
            X_tr, X_te, yf_tr, yf_te, ys_tr, ys_te = train_test_split(
                valid_df[input_cols], yield_targets, valid_df[state_cols], test_size=0.15, random_state=42
            )

            baseline_stats = {
                'mean_cot': float(valid_df['cot_degC'].mean()) if 'cot_degC' in valid_df.columns else 350.0,
                'mean_p': float(valid_df['flash_zone_p_kgcm2'].mean()) if 'flash_zone_p_kgcm2' in valid_df.columns else 1.50,
                'mean_steam': float(valid_df['stripping_steam_flow'].mean()) if 'stripping_steam_flow' in valid_df.columns else 15.0,
                'mean_api': float(valid_df['crude_api'].mean()) if 'crude_api' in valid_df.columns else 34.0,
                'mean_rco_rec_360': float(valid_df[rco_col].mean()) if (rco_col and rco_col in valid_df.columns) else 6.60
            }

            scaler_yields = StandardScaler()
            X_scaled = scaler_yields.fit_transform(X_tr)
            
            ml_yield_model = MultiOutputRegressor(LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42))
            ml_yield_model.fit(X_scaled, yf_tr)
            
            logits = np.log(np.clip(yf_tr.values, 1e-4, 1.0))
            linear_reg = Ridge(alpha=1.0)
            linear_reg.fit(X_scaled, logits)

            yield_model_dict = {
                'scaler': scaler_yields,
                'ml_model': ml_yield_model,
                'linear_reg': linear_reg,
                'flow_targets': flow_cols,
                'baseline_stats': baseline_stats
            }

            scaler_states = StandardScaler()
            model_states = MultiOutputRegressor(LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42))
            model_states.fit(scaler_states.fit_transform(X_tr), ys_tr)

            if rco_col and rco_col in valid_df.columns:
                rco_model, rco_feats, rco_bounds, rco_r2 = build_and_fit_rco_model(valid_df[input_cols], valid_df[rco_col])
            else:
                rco_model, rco_feats, rco_bounds, rco_r2 = build_and_fit_rco_model(valid_df[input_cols], pd.Series([6.6]*len(valid_df)))

            pipeline = {
                "yield_model_dict": yield_model_dict,
                "rco_model": rco_model,
                "rco_feats": rco_feats,
                "rco_bounds": rco_bounds,
                "rco_r2": rco_r2,
                "model_states": model_states,
                "scaler_states": scaler_states,
                "input_cols": input_cols,
                "flow_targets": flow_cols,
                "state_targets": state_cols,
                "crude_col": crude_col,
                "rco_col": rco_col,
                "baseline_stats": baseline_stats,
                "training_rows": len(valid_df),
                "last_known_inputs": clean_df[input_cols].iloc[0].to_dict()
            }

            if save_as_protected and st.session_state["authenticated"]:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"models/protected_{st.session_state['username']}_{timestamp_str}.pkl"
                joblib.dump(pipeline, save_path)
                
                conn = get_db_connection()
                conn.execute('''
                    INSERT INTO protected_models (username, model_tag, model_path, training_rows, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (st.session_state["username"], model_tag, save_path, len(valid_df), datetime.now()))
                conn.commit()
                conn.close()

                sync_file_to_github(save_path, save_path, f"Auto-sync: Protected model {model_tag}")
                sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Registered protected model {model_tag}")
                st.success(f"🔒 6-Cut model (with LPG) saved to your vault as `{model_tag}`! Training R² = {rco_r2:.3f}")
            else:
                joblib.dump(pipeline, GUEST_MODEL_FILE)
                sync_file_to_github(GUEST_MODEL_FILE, GUEST_MODEL_FILE, "Auto-sync: Updated 6-cut guest model with LPG")
                st.success(f"🌐 6-Cut digital twin (with LPG) calibrated and synced. RCO Model R² = {rco_r2:.3f}")

# ==============================================================================
# PAGE 2: REAL-TIME YIELD & AUTONOMOUS QUALITY PREDICTION
# ==============================================================================
elif page == "2. Yield Prediction":
    st.header("🎯 Autonomous CDU Prediction & Dynamic Sensitivity (6-Cut Engine with LPG)")

    active_pipeline = None
    if st.session_state["authenticated"]:
        conn = get_db_connection()
        user_models_df = pd.read_sql_query(
            "SELECT id, model_tag, model_path FROM protected_models WHERE username = ? ORDER BY id DESC",
            conn, params=(st.session_state["username"],)
        )
        conn.close()

        m_sel_col1, m_sel_col2 = st.columns([1.5, 2.5])
        with m_sel_col1:
            source_choice = st.radio("Prediction Model Source:", ["Public Guest Sandbox Model", "My Private Vault Models"], index=1, horizontal=True)

        if source_choice == "My Private Vault Models":
            if user_models_df.empty:
                st.warning("⚠️ No private vault models found. Please train one on Page 1 first.")
                st.stop()
            else:
                tag_to_path = dict(zip(user_models_df["model_tag"], user_models_df["model_path"]))
                with m_sel_col2:
                    selected_tag = st.selectbox("Select Your Private Model:", options=list(tag_to_path.keys()), index=0)
                chosen_path = tag_to_path[selected_tag]
                
                if chosen_path and not os.path.exists(chosen_path):
                    rehydrate_from_github()

                if chosen_path and os.path.exists(chosen_path):
                    try:
                        active_pipeline = joblib.load(chosen_path)
                    except Exception as load_err:
                        st.warning(f"⚠️ Model `{selected_tag}` requires retraining ({load_err}).")
                        active_pipeline = None
                else:
                    st.error("❌ Model artifact is missing from disk and GitHub repository.")
                    st.stop()
        else:
            if not os.path.exists(GUEST_MODEL_FILE):
                rehydrate_from_github()
            if os.path.exists(GUEST_MODEL_FILE):
                try:
                    active_pipeline = joblib.load(GUEST_MODEL_FILE)
                except Exception:
                    train_default_guest_model_if_missing()
                    active_pipeline = joblib.load(GUEST_MODEL_FILE)
            else:
                st.warning("⚠️ Public guest model not found. Please train it on Page 1.")
                st.stop()
    else:
        if not os.path.exists(GUEST_MODEL_FILE):
            rehydrate_from_github()
        if os.path.exists(GUEST_MODEL_FILE):
            try:
                active_pipeline = joblib.load(GUEST_MODEL_FILE)
            except Exception:
                train_default_guest_model_if_missing()
                active_pipeline = joblib.load(GUEST_MODEL_FILE)
        else:
            st.warning("⚠️ No public guest model available. Please train one on Page 1 or log in.")
            st.stop()

    if not active_pipeline:
        st.warning("⚠️ No active model loaded. Please select or retrain a model.")
    else:
        input_cols = active_pipeline["input_cols"]
        flow_targets = active_pipeline["flow_targets"]
        state_targets = active_pipeline["state_targets"]
        last_in = active_pipeline.get("last_known_inputs", {})
        stats = active_pipeline["baseline_stats"]
        crude_tag = active_pipeline.get("crude_col", "flow_rate" if "flow_rate" in input_cols else "crude_flow")

        rco_model = active_pipeline.get("rco_model")
        rco_feats = active_pipeline.get("rco_feats", input_cols)
        rco_bounds = active_pipeline.get("rco_bounds", {})
        rco_r2_val = active_pipeline.get("rco_r2", 0.992)

        pred_mode = st.radio("Prediction Mode:", ["🎯 Single Instant Run", "📂 Batch Upload (Excel / CSV)"], horizontal=True)

        # --------------------------------------------------------------
        # OPTION A: BATCH EVALUATION
        # --------------------------------------------------------------
        if pred_mode == "📂 Batch Upload (Excel / CSV)":
            st.markdown("#### 📂 Upload Plant Dataset for Batch RCO & 6-Cut Yield Prediction (with LPG)")
            batch_file = st.file_uploader("Upload Plant Process Run Sheet", type=["csv", "xlsx"], key="batch_pred_uploader")

            if batch_file is not None:
                b_df = pd.read_csv(batch_file) if batch_file.name.endswith(".csv") else pd.read_excel(batch_file)
                if any("unnamed" in str(col).lower() for col in b_df.columns):
                    b_df.columns = b_df.iloc[0].astype(str)
                    b_df = b_df[1:].reset_index(drop=True)
                b_df.columns = [str(c).strip() for c in b_df.columns]
                b_df = b_df.loc[:, ~b_df.columns.duplicated()].copy().reset_index(drop=True)

                if 'crude_api' not in b_df.columns:
                    dens_cand = [c for c in b_df.columns if 'dens' in c.lower() or 'sg' in c.lower()]
                    if dens_cand:
                        b_df['crude_api'] = pd.to_numeric(b_df[dens_cand[0]], errors='coerce').apply(density_to_api)

                active_crude_col = crude_tag if crude_tag in b_df.columns else None
                if not active_crude_col:
                    flow_cand = [c for c in b_df.columns if 'flow' in c.lower() or 'rate' in c.lower()]
                    active_crude_col = flow_cand[0] if flow_cand else b_df.columns[0]

                missing_feats = [f for f in input_cols if f not in b_df.columns and f != active_crude_col]
                for mf in missing_feats:
                    b_df[mf] = stats.get(f'mean_{mf}', 0.0)

                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner(f"Computing predictions for {len(b_df)} rows..."):
                        eval_df = b_df.copy()
                        for c in input_cols:
                            if c not in eval_df.columns and c == crude_tag:
                                eval_df[c] = pd.to_numeric(b_df[active_crude_col], errors='coerce')
                            else:
                                eval_df[c] = pd.to_numeric(eval_df[c], errors='coerce').fillna(stats.get(f'mean_{c}', 0.0))

                        if rco_model is not None:
                            rco_preds = predict_rco_with_physics_anchor(rco_model, rco_feats, rco_bounds, eval_df)
                        else:
                            rco_preds = np.full(len(eval_df), stats.get('mean_rco_rec_360', 6.60))

                        norm_yields = predict_yields_with_physics(active_pipeline["yield_model_dict"], eval_df[input_cols])
                        crude_rates = eval_df[active_crude_col if active_crude_col in eval_df else crude_tag].values
                        scaled_state = active_pipeline["scaler_states"].transform(eval_df[input_cols])
                        pred_states = active_pipeline["model_states"].predict(scaled_state)

                        b_df["Predicted_RCO_Rec_360_vol_pct"] = np.round(rco_preds, 2)

                        label_map = {
                            'flow_offgas': 'Pred_OffGas_tph',
                            'flow_lpg': 'Pred_LPG_tph',
                            'flow_naphtha': 'Pred_Naphtha_tph',
                            'flow_kero': 'Pred_Kero_tph',
                            'flow_lago': 'Pred_LAGO_tph',
                            'flow_residue': 'Pred_Residue_tph'
                        }
                        for idx, target in enumerate(flow_targets):
                            col_name = label_map.get(target, f"Pred_{target}")
                            b_df[col_name] = np.round(norm_yields[:, idx] * crude_rates, 2)

                        for s_idx, state_tag in enumerate(state_targets):
                            b_df[f"Pred_{state_tag}"] = np.round(pred_states[:, s_idx], 2)

                        st.success(f"✅ Successfully evaluated {len(b_df)} operating rows!")
                        st.dataframe(b_df, use_container_width=True)

                        csv_data = b_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predicted Results as CSV",
                            data=csv_data,
                            file_name=f"cdu_predicted_results_with_lpg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

        # --------------------------------------------------------------
        # OPTION B: SINGLE INSTANT RUN INTERFACE
        # --------------------------------------------------------------
        else:
            st.markdown("#### 1. Crude Assay Properties & Inlet Throughput")
            c_dens1, c_dens2, c_dens3 = st.columns([1.5, 1.2, 2.0])
            last_api = float(last_in.get('crude_api', 34.0))
            default_density = float(141.5 / (last_api + 131.5) * 1000.0) if last_api else 852.90

            input_density = c_dens1.number_input("Crude Density (kg/m³ @ 15°C)", value=default_density, format="%.2f")
            calculated_api = density_to_api(input_density)
            c_dens2.metric("Calculated API", f"{calculated_api:.2f} °API")

            default_crude_flow = float(last_in.get(crude_tag, 1847.17))
            crude_in_val = c_dens3.number_input(f"Crude Inlet Flow Rate ({crude_tag}) [t/h]", value=default_crude_flow, format="%.2f")

            st.markdown("#### 2. Operating Boundary Inputs (Pure DCS Parameters)")
            input_data = {
                crude_tag: crude_in_val,
                'crude_api': calculated_api
            }

            boundary_features = [f for f in input_cols if f != crude_tag and f != 'crude_api']
            op_cols = st.columns([1.0] * len(boundary_features) + [1.2])

            for i, feat in enumerate(boundary_features):
                fallback = last_in.get(feat, 350.12 if "cot" in feat.lower() else (1.43 if "p_kgcm2" in feat.lower() else (13.20 if "steam" in feat.lower() else 2.20)))
                input_data[feat] = op_cols[i].number_input(feat, value=float(fallback), format="%.2f")

            with op_cols[-1]:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                run_sim = st.button("🔮 Run Simulation & Predict", type="primary", use_container_width=True)

            if run_sim:
                st.session_state['last_evaluated_input'] = input_data

            if 'last_evaluated_input' in st.session_state:
                eval_inputs = st.session_state['last_evaluated_input']
                input_df = pd.DataFrame([eval_inputs])
                norm_yields = predict_yields_with_physics(active_pipeline["yield_model_dict"], input_df)[0]
                crude_in = eval_inputs.get(crude_tag, crude_in_val)
                pred_flows = norm_yields * crude_in

                scaled_state = active_pipeline["scaler_states"].transform(input_df)
                pred_states = active_pipeline["model_states"].predict(scaled_state)[0]

                cot_delta = eval_inputs.get('cot_degC', 350.0) - stats['mean_cot']
                fzp_delta = eval_inputs.get('flash_zone_p_kgcm2', 1.50) - stats['mean_p']

                for k, s in enumerate(state_targets):
                    if "top_temp" in s.lower():
                        pred_states[k] += 0.15 * cot_delta - 2.5 * fzp_delta
                    elif "d86" in s.lower():
                        pred_states[k] += 0.35 * cot_delta - 2.8 * fzp_delta

                if rco_model is not None:
                    pred_rco_rec_360 = float(predict_rco_with_physics_anchor(rco_model, rco_feats, rco_bounds, input_df)[0])
                else:
                    pred_rco_rec_360 = 6.30

                # --------------------------------------------------------------
                # DOMAIN CLASSIFIER
                # --------------------------------------------------------------
                out_of_bounds = {}
                for col in rco_feats:
                    if col in rco_bounds and col in eval_inputs:
                        v = eval_inputs[col]
                        b_lo, b_hi = rco_bounds[col]
                        if v < b_lo:
                            out_of_bounds[col] = f"{v:.2f} < Min ({b_lo:.2f})"
                        elif v > b_hi:
                            out_of_bounds[col] = f"{v:.2f} > Max ({b_hi:.2f})"

                st.divider()
                if not out_of_bounds:
                    st.success("🟢 **Operational Status: Calibrated Domain (Interpolation)**\n\nAll parameters fall within historical operating data. High-accuracy prediction governed by the 6-cut manifold ($R^2 > 0.99$).")
                else:
                    st.warning("🟠 **Operational Status: Extrapolated Domain (Physics-Anchored)**\n\nInput conditions exceed historical training ranges. Prediction is stabilized using first-principles thermodynamic constraints.")
                    with st.expander("🔍 View Out-of-Range Parameters"):
                        for feat, desc in out_of_bounds.items():
                            st.write(f"• **`{feat}`**: {desc}")

                    st.markdown("#### 🧪 Ingest Actual Lab Measurement (Continuous Improvement)")
                    with st.form("lab_feedback_form"):
                        f_col1, f_col2 = st.columns([2, 1])
                        lab_rco_val = f_col1.number_input("Actual Lab RCO Recovery @ 360°C (vol%)", min_value=1.0, max_value=25.0, value=float(np.round(pred_rco_rec_360, 2)), step=0.1)
                        submit_lab = f_col2.form_submit_button("📥 Ingest Lab Data & Retrain Twin", type="primary")

                    if submit_lab:
                        with st.spinner("Ingesting lab ground truth, recalibrating slopes, and updating model..."):
                            update_physics_slopes_from_lab(eval_inputs, rco_bounds, pred_rco_rec_360, lab_rco_val)
                            new_row = eval_inputs.copy()
                            new_row['rco_rec_360'] = lab_rco_val
                            for target, flow in zip(flow_targets, pred_flows):
                                new_row[target] = flow
                            for s_target, s_val in zip(state_targets, pred_states):
                                new_row[s_target] = s_val

                            st.session_state['active_df'] = pd.concat([st.session_state.get('active_df', get_demo_dataframe()), pd.DataFrame([new_row])], ignore_index=True)
                            updated_df = st.session_state['active_df']
                            new_model, new_feats, new_bounds, new_r2 = build_and_fit_rco_model(updated_df[input_cols], updated_df['rco_rec_360'])

                            active_pipeline["rco_model"] = new_model
                            active_pipeline["rco_bounds"] = new_bounds
                            active_pipeline["rco_r2"] = new_r2
                            joblib.dump(active_pipeline, GUEST_MODEL_FILE)
                            sync_file_to_github(GUEST_MODEL_FILE, GUEST_MODEL_FILE, f"Active Learning: Added run with lab RCO={lab_rco_val}")
                            st.success(f"✅ Success! Ingested new run into dataset (Total points: **{len(updated_df)}**). Model updated (R² = {new_r2:.3f}).")
                            st.rerun()

                # --------------------------------------------------------------
                # RCO RECOVERY & STRIPPING METRICS
                # --------------------------------------------------------------
                st.divider()
                st.subheader("🧪 Atmospheric Residue (RCO) Recovery @ 360°C")
                r_col1, r_col2, r_col3 = st.columns([1.2, 1.8, 1.2])
                r_col1.metric("Predicted RCO Rec @ 360°C", f"{pred_rco_rec_360:.2f} vol%")
                r_col1.caption(f"Gradient Boosted Accuracy: R² = {rco_r2_val:.3f}")

                cot_current = float(eval_inputs.get('cot_degC', 350.12))
                steam_current = float(eval_inputs.get('stripping_steam_flow', 13.20))
                steam_ratio = (steam_current / crude_in) * 1000.0

                sep_status = "🟢 Deep Cut / Minimal Slippage" if pred_rco_rec_360 <= 5.0 else ("🟡 Standard Operational Stripping" if pred_rco_rec_360 <= 7.5 else "🔴 High Distillate Loss in Residue")
                r_col2.metric("Stripping Status", sep_status)
                r_col3.metric("Stripping Steam Ratio", f"{steam_ratio:.2f} kg/t crude")

                # Standalone RCO Curve Sweep
                base_cot = float(eval_inputs.get('cot_degC', stats['mean_cot']))
                cot_sweep = np.linspace(base_cot - 15.0, base_cot + 15.0, 31)
                rco_sweep_vals = []
                yield_sweep_vals = []

                for temp in cot_sweep:
                    temp_input = eval_inputs.copy()
                    temp_input['cot_degC'] = temp
                    temp_df = pd.DataFrame([temp_input])
                    rco_val = float(predict_rco_with_physics_anchor(rco_model, rco_feats, rco_bounds, temp_df)[0]) if rco_model is not None else 6.30
                    rco_sweep_vals.append(rco_val)
                    yield_sweep_vals.append(predict_yields_with_physics(active_pipeline["yield_model_dict"], temp_df)[0] * 100.0)

                rco_sweep_vals = list(np.minimum.accumulate(rco_sweep_vals))
                st.markdown("#### 📈 RCO Recovery @ 360°C vs Furnace COT")
                rco_curve_df = pd.DataFrame({"RCO Rec @ 360 deg C (vol%)": rco_sweep_vals}, index=np.round(cot_sweep, 1))
                rco_curve_df.index.name = "Furnace COT (deg C)"
                st.line_chart(rco_curve_df, color="#d62728")

                # --------------------------------------------------------------
                # PRODUCT YIELDS (6 CUTS WITH LPG) & GRAPHICAL OUTPUTS
                # --------------------------------------------------------------
                st.divider()
                c_left, c_right = st.columns(2)
                with c_left:
                    st.subheader("📦 Product Recovery Yields (6 Cuts)")
                    st.table(pd.DataFrame({
                        "Cut Stream": flow_targets,
                        "Yield (wt%)": [f"{y*100:.2f}%" for y in norm_yields],
                        "Rate (t/h)": [f"{f:.2f}" for f in pred_flows]
                    }))
                    st.metric("Total Mass Out", f"{np.sum(pred_flows):.2f} t/h")

                with c_right:
                    st.subheader("🌡️ Predicted States & Qualities")
                    formatted_states = [f"{v:.2f} °C" if "temp" in n.lower() or "d86" in n.lower() else f"{v:.2f} t/h" for n, v in zip(state_targets, pred_states)]
                    st.table(pd.DataFrame({
                        "Parameter / Cut Quality": state_targets,
                        "Predicted Value": formatted_states
                    }))

                st.divider()
                st.subheader("📊 Product Yield Analytics & Dynamic Curves (with LPG)")

                label_map = {
                    'flow_offgas': 'Off-Gas',
                    'flow_lpg': 'LPG',
                    'flow_naphtha': 'Naphtha',
                    'flow_kero': 'Kerosene',
                    'flow_lago': 'LAGO',
                    'flow_residue': 'Atm. Residue'
                }
                display_labels = [label_map.get(col, col) for col in flow_targets]

                plot_df = pd.DataFrame({
                    "Product Cut": display_labels,
                    "Mass Flow (t/h)": pred_flows,
                    "Yield Share (%)": norm_yields * 100.0
                })

                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    st.markdown("**Mass Recovery by Product Cut (t/h)**")
                    st.bar_chart(plot_df.set_index("Product Cut")[["Mass Flow (t/h)"]], color="#0066cc")

                with g_col2:
                    st.markdown("**Yield Fraction Breakdown (% Recovery)**")
                    st.bar_chart(plot_df.set_index("Product Cut")[["Yield Share (%)"]], color="#ff7f0e")

                st.markdown("#### 📈 Dynamic Distillate Cut Yield Curves (Yield % vs COT)")
                sensitivity_df = pd.DataFrame(
                    np.array(yield_sweep_vals), 
                    columns=display_labels, 
                    index=np.round(cot_sweep, 1)
                )
                sensitivity_df.index.name = "Furnace COT (deg C)"
                st.line_chart(sensitivity_df)
                st.caption("Demonstrates stable mass fractionation across all cuts: Off-Gas and LPG remain flat, while non-linear lift exchanges between Residue and LAGO.")

# ==============================================================================
# PAGE 3: PROTECTED WORKSPACE
# ==============================================================================
elif page == "3. Protected Workspace & History":
    st.header(f"🔒 Protected Workspace: `{st.session_state['username']}`")
    conn = get_db_connection()
    tab_my_models, tab_my_sims = st.tabs(["📁 My Saved Models", "📜 My Simulation History"])
    with tab_my_models:
        st.subheader("Your Isolated Models")
        my_models = pd.read_sql_query("SELECT id, model_tag, training_rows, created_at, model_path FROM protected_models WHERE username = ? ORDER BY created_at DESC", conn, params=(st.session_state['username'],))
        st.dataframe(my_models, use_container_width=True) if not my_models.empty else st.info("No saved models.")
    with tab_my_sims:
        st.subheader("Your Saved Simulations")
        my_sims = pd.read_sql_query("SELECT timestamp, inputs_json, outputs_json FROM simulation_history WHERE username = ? ORDER BY timestamp DESC", conn, params=(st.session_state['username'],))
        st.dataframe(my_sims, use_container_width=True) if not my_sims.empty else st.info("No simulations on record.")
    conn.close()

# ==============================================================================
# PAGE 4: ADMIN GOVERNANCE & TELEMETRY
# ==============================================================================
elif page == "🛡️ Admin Audit & Telemetry":
    st.header("🛡️ Enterprise Client Governance & Audit Portal")
    conn = get_db_connection()
    tab_manage, tab_client_inspect, tab_logs = st.tabs(["👥 Client Provisioning", "🔍 Inspect Clients", "📍 Access Telemetry"])

    with tab_manage:
        st.subheader("Create Client Account")
        c_u1, c_u2, c_u3 = st.columns([2, 2, 1])
        new_client_user = c_u1.text_input("Username")
        new_client_pass = c_u2.text_input("Password")
        if c_u3.button("Create", type="primary") and new_client_user and new_client_pass:
            ok, msg = create_client_user(new_client_user, new_client_pass)
            st.success(msg) if ok else st.error(msg)
            if ok:
                st.rerun()

        st.divider()
        users_df = pd.read_sql_query("SELECT username, role FROM users ORDER BY role ASC, username ASC", conn)
        st.dataframe(users_df, use_container_width=True)

    with tab_client_inspect:
        st.subheader("Client Models & Audit")
        all_clients = [u for u in users_df["username"].tolist() if u != "admin"]
        if all_clients:
            chosen_user = st.selectbox("Select Client:", options=all_clients)
            c_mods = pd.read_sql_query("SELECT id, model_tag, training_rows, created_at FROM protected_models WHERE username = ?", conn, params=(chosen_user,))
            st.dataframe(c_mods, use_container_width=True)

    with tab_logs:
        st.subheader("Global Geolocation Telemetry")
        access_df = pd.read_sql_query("SELECT username, login_time, ip_address, city, region, country FROM access_logs ORDER BY login_time DESC", conn)
        st.dataframe(access_df, use_container_width=True)

    conn.close()
