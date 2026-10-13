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

# Physical Engineering Bounds matched to actual CDU operating cuts
YIELD_BOUNDS = {
    'flow_offgas': (0.0005, 0.050),
    'flow_naphtha': (0.020, 0.150),
    'flow_kero': (0.030, 0.120),
    'flow_lago': (0.150, 0.350),
    'flow_residue': (0.400, 0.650)
}

PHYSICS_SLOPES = {
    'flow_offgas': 0.00008,
    'flow_naphtha': 0.00045,
    'flow_kero': 0.00025,
    'flow_lago': 0.00100,
    'flow_residue': -0.00178
}

# ==============================================================================
# EMBEDDED PLANT RUNS WITH MEASURED RCO RECOVERY @ 360°C
# ==============================================================================
EXCEL_RECORDS = [
    {"rco_rec_360": 6.3, "crude_flow": 1847.17, "crude_density": 852.9, "crude_api": 34.3938, "cot_degC": 350.120, "flash_zone_p_kgcm2": 1.43, "stripping_steam_flow": 13.20, "top_pa": 380.11, "kero_pa_flow_tph": 394.33, "lago_pa_flow_tph": 151.520, "top_temp_degC": 120.26, "lago_d86_95_degC": 243.27, "flow_offgas": 6.85000, "flow_naphtha": 124.81, "flow_kero": 78.36, "flow_lago": 442.39, "flow_residue": 978.09},
    {"rco_rec_360": 6.25, "crude_flow": 1918.67, "crude_density": 859.4, "crude_api": 33.1256, "cot_degC": 354.775, "flash_zone_p_kgcm2": 1.45, "stripping_steam_flow": 13.12, "top_pa": 397.46, "kero_pa_flow_tph": 348.58, "lago_pa_flow_tph": 135.020, "top_temp_degC": 117.21, "lago_d86_95_degC": 250.47, "flow_offgas": 3.67757, "flow_naphtha": 102.53, "flow_kero": 106.20, "flow_lago": 434.00, "flow_residue": 1021.23},
    {"rco_rec_360": 14.5, "crude_flow": 1810.48, "crude_density": 849.9, "crude_api": 34.9755, "cot_degC": 354.825, "flash_zone_p_kgcm2": 1.50, "stripping_steam_flow": 13.50, "top_pa": 472.87, "kero_pa_flow_tph": 442.03, "lago_pa_flow_tph": 143.250, "top_temp_degC": 118.01, "lago_d86_95_degC": 248.21, "flow_offgas": 4.09316, "flow_naphtha": 145.09, "flow_kero": 104.56, "flow_lago": 454.65, "flow_residue": 897.09},
    {"rco_rec_360": 6.3, "crude_flow": 1677.31, "crude_density": 863.0, "crude_api": 32.4427, "cot_degC": 356.170, "flash_zone_p_kgcm2": 1.42, "stripping_steam_flow": 13.84, "top_pa": 332.02, "kero_pa_flow_tph": 335.93, "lago_pa_flow_tph": 114.155, "top_temp_degC": 113.60, "lago_d86_95_degC": 243.76, "flow_offgas": 2.47341, "flow_naphtha": 98.87, "flow_kero": 74.75, "flow_lago": 403.54, "flow_residue": 943.73},
    {"rco_rec_360": 2.0, "crude_flow": 1549.10, "crude_density": 867.9, "crude_api": 31.5190, "cot_degC": 357.890, "flash_zone_p_kgcm2": 1.36, "stripping_steam_flow": 14.74, "top_pa": 270.81, "kero_pa_flow_tph": 251.45, "lago_pa_flow_tph": 109.570, "top_temp_degC": 109.78, "lago_d86_95_degC": 243.61, "flow_offgas": 2.40643, "flow_naphtha": 46.99, "flow_kero": 82.75, "flow_lago": 355.85, "flow_residue": 870.43},
    {"rco_rec_360": 2.9, "crude_flow": 1366.14, "crude_density": 849.4, "crude_api": 35.0656, "cot_degC": 358.645, "flash_zone_p_kgcm2": 1.37, "stripping_steam_flow": 14.00, "top_pa": 271.14, "kero_pa_flow_tph": 238.39, "lago_pa_flow_tph": 98.240, "top_temp_degC": 114.70, "lago_d86_95_degC": 242.92, "flow_offgas": 2.00850, "flow_naphtha": 82.68, "flow_kero": 83.46, "flow_lago": 351.23, "flow_residue": 709.18},
    {"rco_rec_360": 6.8, "crude_flow": 1827.41, "crude_density": 866.6, "crude_api": 31.7730, "cot_degC": 351.285, "flash_zone_p_kgcm2": 1.48, "stripping_steam_flow": 13.90, "top_pa": 359.42, "kero_pa_flow_tph": 390.85, "lago_pa_flow_tph": 101.480, "top_temp_degC": 121.36, "lago_d86_95_degC": 244.17, "flow_offgas": 3.79992, "flow_naphtha": 77.62, "flow_kero": 64.03, "flow_lago": 466.04, "flow_residue": 1021.18},
    {"rco_rec_360": 5.9, "crude_flow": 1609.22, "crude_density": 854.9, "crude_api": 34.0018, "cot_degC": 354.230, "flash_zone_p_kgcm2": 1.43, "stripping_steam_flow": 14.00, "top_pa": 316.88, "kero_pa_flow_tph": 343.26, "lago_pa_flow_tph": 92.995, "top_temp_degC": 121.95, "lago_d86_95_degC": 242.87, "flow_offgas": 2.13730, "flow_naphtha": 57.84, "flow_kero": 57.21, "flow_lago": 419.08, "flow_residue": 891.33},
    {"rco_rec_360": 3.8, "crude_flow": 1491.72, "crude_density": 853.0, "crude_api": 34.3646, "cot_degC": 359.015, "flash_zone_p_kgcm2": 1.42, "stripping_steam_flow": 15.90, "top_pa": 353.21, "kero_pa_flow_tph": 334.69, "lago_pa_flow_tph": 97.005, "top_temp_degC": 117.91, "lago_d86_95_degC": 249.50, "flow_offgas": 5.76393, "flow_naphtha": 135.48, "flow_kero": 75.83, "flow_lago": 386.87, "flow_residue": 766.75},
    {"rco_rec_360": 5.7, "crude_flow": 1253.22, "crude_density": 859.6, "crude_api": 33.1007, "cot_degC": 358.550, "flash_zone_p_kgcm2": 1.35, "stripping_steam_flow": 13.99, "top_pa": 270.40, "kero_pa_flow_tph": 256.87, "lago_pa_flow_tph": 95.180, "top_temp_degC": 115.92, "lago_d86_95_degC": 248.91, "flow_offgas": 0.68082, "flow_naphtha": 48.56, "flow_kero": 68.45, "flow_lago": 354.82, "flow_residue": 631.37},
    {"rco_rec_360": 5.7, "crude_flow": 1926.91, "crude_density": 855.5, "crude_api": 33.8780, "cot_degC": 347.955, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 17.50, "top_pa": 472.44, "kero_pa_flow_tph": 454.19, "lago_pa_flow_tph": 117.785, "top_temp_degC": 116.12, "lago_d86_95_degC": 246.93, "flow_offgas": 4.46463, "flow_naphtha": 139.83, "flow_kero": 80.72, "flow_lago": 472.45, "flow_residue": 1009.91},
    {"rco_rec_360": 4.6, "crude_flow": 1954.94, "crude_density": 1005.9, "crude_api": 9.1616, "cot_degC": 350.210, "flash_zone_p_kgcm2": 1.49, "stripping_steam_flow": 17.98, "top_pa": 407.73, "kero_pa_flow_tph": 331.05, "lago_pa_flow_tph": 108.970, "top_temp_degC": 114.04, "lago_d86_95_degC": 250.31, "flow_offgas": 5.67824, "flow_naphtha": 131.59, "flow_kero": 126.69, "flow_lago": 411.36, "flow_residue": 1059.74},
    {"rco_rec_360": 6.0, "crude_flow": 1955.56, "crude_density": 845.3, "crude_api": 35.8735, "cot_degC": 345.885, "flash_zone_p_kgcm2": 1.60, "stripping_steam_flow": 14.24, "top_pa": 501.50, "kero_pa_flow_tph": 434.37, "lago_pa_flow_tph": 94.985, "top_temp_degC": 117.89, "lago_d86_95_degC": 249.05, "flow_offgas": 6.71467, "flow_naphtha": 104.10, "flow_kero": 128.90, "flow_lago": 455.70, "flow_residue": 1021.55},
    {"rco_rec_360": 9.7, "crude_flow": 1903.98, "crude_density": 837.5, "crude_api": 37.4303, "cot_degC": 339.700, "flash_zone_p_kgcm2": 1.72, "stripping_steam_flow": 11.98, "top_pa": 518.05, "kero_pa_flow_tph": 435.19, "lago_pa_flow_tph": 89.090, "top_temp_degC": 122.13, "lago_d86_95_degC": 246.84, "flow_offgas": 6.60169, "flow_naphtha": 100.21, "flow_kero": 128.61, "flow_lago": 435.34, "flow_residue": 939.22},
    {"rco_rec_360": 5.3, "crude_flow": 1852.52, "crude_density": 863.1, "crude_api": 32.4218, "cot_degC": 349.370, "flash_zone_p_kgcm2": 1.50, "stripping_steam_flow": 17.00, "top_pa": 405.79, "kero_pa_flow_tph": 379.08, "lago_pa_flow_tph": 79.870, "top_temp_degC": 118.36, "lago_d86_95_degC": 253.46, "flow_offgas": 3.47714, "flow_naphtha": 68.59, "flow_kero": 105.26, "flow_lago": 454.95, "flow_residue": 1020.43},
    {"rco_rec_360": 4.4, "crude_flow": 1903.44, "crude_density": 852.8, "crude_api": 34.3996, "cot_degC": 348.055, "flash_zone_p_kgcm2": 1.55, "stripping_steam_flow": 17.00, "top_pa": 500.05, "kero_pa_flow_tph": 466.85, "lago_pa_flow_tph": 96.675, "top_temp_degC": 121.27, "lago_d86_95_degC": 253.36, "flow_offgas": 4.85334, "flow_naphtha": 100.20, "flow_kero": 111.76, "flow_lago": 474.09, "flow_residue": 996.26},
    {"rco_rec_360": 2.5, "crude_flow": 1918.92, "crude_density": 855.2, "crude_api": 33.9380, "cot_degC": 350.020, "flash_zone_p_kgcm2": 1.59, "stripping_steam_flow": 17.70, "top_pa": 447.80, "kero_pa_flow_tph": 430.32, "lago_pa_flow_tph": 91.060, "top_temp_degC": 116.67, "lago_d86_95_degC": 249.48, "flow_offgas": 4.69515, "flow_naphtha": 94.98, "flow_kero": 97.03, "flow_lago": 448.48, "flow_residue": 1045.83},
    {"rco_rec_360": 2.0, "crude_flow": 1907.06, "crude_density": 851.9, "crude_api": 34.5807, "cot_degC": 349.615, "flash_zone_p_kgcm2": 1.61, "stripping_steam_flow": 15.15, "top_pa": 483.63, "kero_pa_flow_tph": 443.99, "lago_pa_flow_tph": 86.150, "top_temp_degC": 119.53, "lago_d86_95_degC": 250.35, "flow_offgas": 4.11358, "flow_naphtha": 96.79, "flow_kero": 112.57, "flow_lago": 451.35, "flow_residue": 1008.93},
    {"rco_rec_360": 5.8, "crude_flow": 1932.69, "crude_density": 846.2, "crude_api": 35.6955, "cot_degC": 350.255, "flash_zone_p_kgcm2": 1.68, "stripping_steam_flow": 17.89, "top_pa": 526.25, "kero_pa_flow_tph": 487.74, "lago_pa_flow_tph": 86.100, "top_temp_degC": 121.70, "lago_d86_95_degC": 253.09, "flow_offgas": 6.23427, "flow_naphtha": 104.22, "flow_kero": 114.39, "flow_lago": 476.78, "flow_residue": 996.47},
    {"rco_rec_360": 5.8, "crude_flow": 1918.36, "crude_density": 855.1, "crude_api": 33.9573, "cot_degC": 348.815, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 16.78, "top_pa": 494.40, "kero_pa_flow_tph": 456.44, "lago_pa_flow_tph": 83.730, "top_temp_degC": 121.17, "lago_d86_95_degC": 251.37, "flow_offgas": 4.87445, "flow_naphtha": 99.33, "flow_kero": 110.87, "flow_lago": 459.52, "flow_residue": 1011.99},
    {"rco_rec_360": 6.5, "crude_flow": 1936.50, "crude_density": 860.6, "crude_api": 32.9133, "cot_degC": 349.160, "flash_zone_p_kgcm2": 1.51, "stripping_steam_flow": 18.48, "top_pa": 431.57, "kero_pa_flow_tph": 401.09, "lago_pa_flow_tph": 75.025, "top_temp_degC": 117.64, "lago_d86_95_degC": 251.18, "flow_offgas": 3.58797, "flow_naphtha": 87.70, "flow_kero": 113.15, "flow_lago": 456.90, "flow_residue": 1070.99},
    {"rco_rec_360": 7.5, "crude_flow": 1930.41, "crude_density": 852.0, "crude_api": 34.5554, "cot_degC": 347.295, "flash_zone_p_kgcm2": 1.61, "stripping_steam_flow": 15.58, "top_pa": 493.84, "kero_pa_flow_tph": 447.63, "lago_pa_flow_tph": 104.425, "top_temp_degC": 124.66, "lago_d86_95_degC": 250.38, "flow_offgas": 4.49535, "flow_naphtha": 120.95, "flow_kero": 116.01, "flow_lago": 449.56, "flow_residue": 978.09},
    {"rco_rec_360": 17.5, "crude_flow": 1918.27, "crude_density": 851.3, "crude_api": 34.7017, "cot_degC": 345.850, "flash_zone_p_kgcm2": 1.61, "stripping_steam_flow": 16.82, "top_pa": 481.09, "kero_pa_flow_tph": 435.09, "lago_pa_flow_tph": 97.260, "top_temp_degC": 121.20, "lago_d86_95_degC": 248.86, "flow_offgas": 5.82920, "flow_naphtha": 99.46, "flow_kero": 115.22, "flow_lago": 455.99, "flow_residue": 980.88},
    {"rco_rec_360": 9.0, "crude_flow": 1927.75, "crude_density": 850.1, "crude_api": 34.9265, "cot_degC": 347.875, "flash_zone_p_kgcm2": 1.64, "stripping_steam_flow": 16.39, "top_pa": 509.63, "kero_pa_flow_tph": 436.59, "lago_pa_flow_tph": 99.840, "top_temp_degC": 123.48, "lago_d86_95_degC": 251.83, "flow_offgas": 6.69661, "flow_naphtha": 87.12, "flow_kero": 132.25, "flow_lago": 455.55, "flow_residue": 980.66},
    {"rco_rec_360": 9.5, "crude_flow": 1923.81, "crude_density": 848.7, "crude_api": 35.2089, "cot_degC": 347.885, "flash_zone_p_kgcm2": 1.67, "stripping_steam_flow": 15.40, "top_pa": 479.77, "kero_pa_flow_tph": 413.04, "lago_pa_flow_tph": 91.790, "top_temp_degC": 124.13, "lago_d86_95_degC": 251.27, "flow_offgas": 6.43440, "flow_naphtha": 98.38, "flow_kero": 122.13, "flow_lago": 435.13, "flow_residue": 978.07},
    {"rco_rec_360": 8.6, "crude_flow": 1912.23, "crude_density": 847.3, "crude_api": 35.4765, "cot_degC": 344.005, "flash_zone_p_kgcm2": 1.70, "stripping_steam_flow": 16.22, "top_pa": 457.27, "kero_pa_flow_tph": 369.21, "lago_pa_flow_tph": 91.005, "top_temp_degC": 123.75, "lago_d86_95_degC": 252.06, "flow_offgas": 6.38299, "flow_naphtha": 77.44, "flow_kero": 130.82, "flow_lago": 421.53, "flow_residue": 948.96},
    {"rco_rec_360": 7.5, "crude_flow": 1901.46, "crude_density": 854.6, "crude_api": 34.0502, "cot_degC": 342.505, "flash_zone_p_kgcm2": 1.64, "stripping_steam_flow": 16.46, "top_pa": 415.05, "kero_pa_flow_tph": 363.42, "lago_pa_flow_tph": 81.740, "top_temp_degC": 120.44, "lago_d86_95_degC": 247.37, "flow_offgas": 6.99080, "flow_naphtha": 53.29, "flow_kero": 109.58, "flow_lago": 416.74, "flow_residue": 995.98}
]

def get_demo_dataframe():
    return pd.DataFrame(EXCEL_RECORDS)

# ==============================================================================
# STEP 4: GITHUB AUTO-COMMIT SYNCHRONIZATION ENGINE
# ==============================================================================
def sync_file_to_github(local_file_path, repo_file_path, commit_message="Auto-sync from Streamlit App"):
    if "GITHUB_TOKEN" not in st.secrets or "GITHUB_REPO" not in st.secrets:
        return False

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
            repo.update_file(
                path=repo_file_path,
                message=commit_message,
                content=content,
                sha=existing_file.sha,
                branch=branch
            )
        except GithubException as e:
            if e.status == 404:
                repo.create_file(
                    path=repo_file_path,
                    message=commit_message,
                    content=content,
                    branch=branch
                )
            else:
                raise e

        return True
    except Exception as err:
        st.sidebar.warning(f"⚠️ GitHub Sync Alert: {err}")
        return False

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
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT,
            role TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            login_time TIMESTAMP,
            ip_address TEXT,
            city TEXT,
            region TEXT,
            country TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS protected_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            model_tag TEXT,
            model_path TEXT,
            training_rows INTEGER,
            created_at TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS simulation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp TIMESTAMP,
            inputs_json TEXT,
            outputs_json TEXT
        )
    ''')
    
    admin_pw = hashlib.sha256("Admin@123".encode()).hexdigest()
    user_pw = hashlib.sha256("User@123".encode()).hexdigest()
    c.execute("INSERT OR REPLACE INTO users VALUES (?, ?, ?)", ("admin", admin_pw, "admin"))
    c.execute("INSERT OR REPLACE INTO users VALUES (?, ?, ?)", ("engineer1", user_pw, "user"))
    conn.commit()
    conn.close()

init_db()

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
# PHYSICS-INFORMED ENGINE
# ==============================================================================
def density_to_api(density_val):
    sg = density_val / 1000.0 if density_val > 10.0 else density_val
    if sg <= 0:
        return 30.0
    return (141.5 / sg) - 131.5

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class PhysicsInformedYieldModel:
    def __init__(self):
        self.ml_model = MultiOutputRegressor(LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42))
        self.linear_reg = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.flow_targets = []
        self.baseline_stats = {}

    def fit(self, X, y_yields, flow_targets, baseline_stats):
        self.flow_targets = flow_targets
        self.baseline_stats = baseline_stats
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model.fit(X_scaled, y_yields)
        logits = np.log(np.clip(y_yields.values, 1e-4, 1.0))
        self.linear_reg.fit(X_scaled, logits)

    def predict(self, X_df):
        X_scaled = self.scaler.transform(X_df)
        ml_preds = self.ml_model.predict(X_scaled)
        linear_preds = softmax(self.linear_reg.predict(X_scaled))
        base_yields = 0.70 * ml_preds + 0.30 * linear_preds

        cot = X_df['cot_degC'].values if 'cot_degC' in X_df else self.baseline_stats['mean_cot']
        fzp = X_df['flash_zone_p_kgcm2'].values if 'flash_zone_p_kgcm2' in X_df else self.baseline_stats['mean_p']
        steam = X_df['stripping_steam_flow'].values if 'stripping_steam_flow' in X_df else self.baseline_stats['mean_steam']
        api = X_df['crude_api'].values if 'crude_api' in X_df else self.baseline_stats.get('mean_api', 34.0)

        delta_severity = (
            (cot - self.baseline_stats['mean_cot'])
            - 16.5 * (fzp - self.baseline_stats['mean_p'])
            + 0.55 * (steam - self.baseline_stats['mean_steam'])
            + 0.85 * (api - self.baseline_stats.get('mean_api', 34.0))
        )

        final_yields = np.zeros_like(base_yields)
        for i, col in enumerate(self.flow_targets):
            slope = PHYSICS_SLOPES.get(col, 0.0)
            adj = base_yields[:, i] + (slope * delta_severity)
            min_b, max_b = YIELD_BOUNDS.get(col, (0.0005, 0.90))
            final_yields[:, i] = np.clip(adj, min_b, max_b)

        return final_yields / np.sum(final_yields, axis=1, keepdims=True)

class RCOQualityModel:
    """Trained regression and thermodynamic sensitivity for RCO recovery @ 360°C"""
    def __init__(self):
        self.reg = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.baseline_stats = {}

    def fit(self, X, y, baseline_stats):
        self.baseline_stats = baseline_stats
        X_scaled = self.scaler.fit_transform(X)
        self.reg.fit(X_scaled, y)

    def predict(self, X_df):
        X_scaled = self.scaler.transform(X_df)
        preds = self.reg.predict(X_scaled)
        return np.clip(preds, 1.5, 20.0)

# ==============================================================================
# DEFAULT MODEL INITIALIZATION
# ==============================================================================
def train_default_guest_model_if_missing():
    if not os.path.exists(GUEST_MODEL_FILE):
        df = get_demo_dataframe()
        s_col = 'sulphur_wt_pct' if 'sulphur_wt_pct' in df.columns else ('sulfur_wt_pct' if 'sulfur_wt_pct' in df.columns else None)
        input_cols = ['crude_flow', 'crude_api', 'cot_degC', 'flash_zone_p_kgcm2', 'stripping_steam_flow', 'lago_d86_95_degC']
        if s_col:
            input_cols.insert(2, s_col)

        flow_cols = ['flow_offgas', 'flow_naphtha', 'flow_kero', 'flow_lago', 'flow_residue']
        state_cols = ['top_pa', 'kero_pa_flow_tph', 'lago_pa_flow_tph', 'top_temp_degC']
        crude_col = 'crude_flow'

        yield_targets = df[flow_cols].div(df[crude_col], axis=0)
        baseline_stats = {
            'mean_cot': float(df['cot_degC'].mean()),
            'mean_p': float(df['flash_zone_p_kgcm2'].mean()),
            'mean_steam': float(df['stripping_steam_flow'].mean()),
            'mean_api': float(df['crude_api'].mean()),
            'mean_rco_rec_360': float(df['rco_rec_360'].mean())
        }

        yield_model = PhysicsInformedYieldModel()
        yield_model.fit(df[input_cols], yield_targets, flow_cols, baseline_stats)

        scaler_states = StandardScaler()
        model_states = MultiOutputRegressor(LGBMRegressor(n_estimators=250, learning_rate=0.03, random_state=42))
        model_states.fit(scaler_states.fit_transform(df[input_cols]), df[state_cols])

        # RCO Recovery @ 360°C Model
        rco_model = RCOQualityModel()
        rco_model.fit(df[input_cols], df['rco_rec_360'], baseline_stats)

        pipeline = {
            "yield_model": yield_model,
            "rco_model": rco_model,
            "model_states": model_states,
            "scaler_states": scaler_states,
            "input_cols": input_cols,
            "flow_targets": flow_cols,
            "state_targets": state_cols,
            "crude_col": crude_col,
            "baseline_stats": baseline_stats,
            "training_rows": len(df),
            "last_known_inputs": df[input_cols].iloc[0].to_dict()
        }
        joblib.dump(pipeline, GUEST_MODEL_FILE)

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

page = st.sidebar.radio("Navigation", nav_options)

# ==============================================================================
# PAGE 1: MODEL TRAINING & INGESTION
# ==============================================================================
if page == "1. Model Training & DCS Upload":
    st.header("⚙️ Column Data Ingestion & Model Training")
    st.markdown("Upload historical plant logs or load the calibrated refinery dataset.")

    uploaded_file = st.file_uploader("Upload DCS Historical Data (CSV or Excel)", type=["csv", "xlsx"])
    col1, col2 = st.columns([1, 4])
    use_synthetic = col1.button("Load Demo DCS Dataset")

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        if any("unnamed" in str(col).lower() for col in raw_df.columns):
            raw_df.columns = raw_df.iloc[0].astype(str)
            raw_df = raw_df[1:].reset_index(drop=True)
        st.session_state['active_df'] = raw_df
        st.session_state['active_src_name'] = f"Uploaded File: `{uploaded_file.name}`"
    elif use_synthetic:
        st.session_state['active_df'] = get_demo_dataframe()
        st.session_state['active_src_name'] = "Loaded CDU Refinery Historical Dataset (27 runs with RCO@360)"

    if 'active_df' not in st.session_state:
        st.session_state['active_df'] = get_demo_dataframe()
        st.session_state['active_src_name'] = "Default Calibrated CDU Operating Dataset"

    df = st.session_state['active_df']
    st.info(f"📂 **Active Dataset:** {st.session_state.get('active_src_name', '')} | Rows: **{len(df)}** | Columns: **{len(df.columns)}**")

    st.subheader("Data Inspector")
    st.dataframe(df.head(5), use_container_width=True)
    with st.expander("🔍 View Complete Raw Dataset"):
        st.dataframe(df)

    has_density = any("dens" in str(c).lower() or "sg" in str(c).lower() for c in df.columns)
    if st.checkbox("Calculate crude_api automatically from Density/SG", value=has_density):
        dens_cols = list(df.columns)
        selected_dens = st.selectbox("Select Density Column", options=dens_cols, index=dens_cols.index('crude_density') if 'crude_density' in dens_cols else 0)
        df['crude_api'] = df[selected_dens].apply(density_to_api)
        st.success(f"Calculated `crude_api` from `{selected_dens}`")

    # Dynamic column identification
    rco_target_candidates = [c for c in df.columns if 'rco' in str(c).lower() or '360' in str(c).lower()]
    default_rco_col = rco_target_candidates[0] if rco_target_candidates else None

    sulfur_col_name = 'sulphur_wt_pct' if 'sulphur_wt_pct' in df.columns else ('sulfur_wt_pct' if 'sulfur_wt_pct' in df.columns else None)
    
    default_inputs = ['crude_flow', 'crude_api', 'cot_degC', 'flash_zone_p_kgcm2', 'stripping_steam_flow', 'lago_d86_95_degC']
    if sulfur_col_name:
        default_inputs.insert(2, sulfur_col_name)

    default_flows = ['flow_offgas', 'flow_naphtha', 'flow_kero', 'flow_lago', 'flow_residue']
    default_states = ['top_pa', 'kero_pa_flow_tph', 'lago_pa_flow_tph', 'top_temp_degC']

    c1, c2, c3 = st.columns(3)
    input_cols = c1.multiselect("Inputs (X)", list(df.columns), default=[c for c in default_inputs if c in df.columns])
    flow_cols = c2.multiselect("Product Flows (Y1)", list(df.columns), default=[c for c in default_flows if c in df.columns])
    state_cols = c3.multiselect("Internal States (Y2)", list(df.columns), default=[c for c in default_states if c in df.columns])
    crude_col = c1.selectbox("Crude Inlet Flow Tag", list(df.columns), index=list(df.columns).index('crude_flow') if 'crude_flow' in df.columns else 0)
    
    rco_col = None
    if default_rco_col:
        rco_col = c2.selectbox("RCO Recovery @ 360°C Target Column", options=rco_target_candidates, index=0)

    save_as_protected = False
    model_tag = "guest_model"
    if st.session_state["authenticated"]:
        st.divider()
        c_save1, c_save2 = st.columns([1, 2])
        save_as_protected = c_save1.checkbox("Save model into my protected private vault", value=True)
        if save_as_protected:
            model_tag = c_save2.text_input("Protected Model Tag", value=f"{st.session_state['username']}_v1")

    if st.button("🚀 Train Digital Twin", type="primary"):
        with st.spinner("Training hybrid yield model & RCO recovery engine..."):
            all_needed = list(set(input_cols + flow_cols + state_cols + [crude_col]))
            if rco_col:
                all_needed.append(rco_col)
            clean_df = df[all_needed].apply(pd.to_numeric, errors='coerce').dropna()

            total_out = clean_df[flow_cols].sum(axis=1)
            valid_df = clean_df[np.abs(total_out - clean_df[crude_col]) / clean_df[crude_col] < 0.25].copy()

            if valid_df.empty:
                st.error("❌ Mass balance error: Cleaned data does not close within tolerance.")
                st.stop()

            yield_targets = valid_df[flow_cols].div(valid_df[crude_col], axis=0)
            X_tr, X_te, yf_tr, yf_te, ys_tr, ys_te, c_tr, c_te = train_test_split(
                valid_df[input_cols], yield_targets, valid_df[state_cols], valid_df[crude_col], test_size=0.15, random_state=42
            )

            baseline_stats = {
                'mean_cot': float(valid_df['cot_degC'].mean()) if 'cot_degC' in valid_df.columns else 350.0,
                'mean_p': float(valid_df['flash_zone_p_kgcm2'].mean()) if 'flash_zone_p_kgcm2' in valid_df.columns else 1.50,
                'mean_steam': float(valid_df['stripping_steam_flow'].mean()) if 'stripping_steam_flow' in valid_df.columns else 15.0,
                'mean_api': float(valid_df['crude_api'].mean()) if 'crude_api' in valid_df.columns else 34.0,
                'mean_rco_rec_360': float(valid_df[rco_col].mean()) if (rco_col and rco_col in valid_df.columns) else 6.60
            }

            yield_model = PhysicsInformedYieldModel()
            yield_model.fit(X_tr, yf_tr, flow_cols, baseline_stats)

            scaler_states = StandardScaler()
            model_states = MultiOutputRegressor(LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42))
            model_states.fit(scaler_states.fit_transform(X_tr), ys_tr)

            # Fit RCO Recovery model
            rco_model = RCOQualityModel()
            if rco_col and rco_col in valid_df.columns:
                rco_model.fit(valid_df[input_cols], valid_df[rco_col], baseline_stats)
            else:
                rco_model.fit(valid_df[input_cols], pd.Series([6.6]*len(valid_df)), baseline_stats)

            pipeline = {
                "yield_model": yield_model,
                "rco_model": rco_model,
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

            # STEP 5 (Hook 4): Auto-sync trained .pkl models directly to GitHub
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

                sync_file_to_github(save_path, save_path, f"Auto-sync: New protected model {model_tag}")
                sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Registered protected model {model_tag}")
                st.success(f"🔒 Model saved to your vault and pushed to GitHub as `{model_tag}`!")
            else:
                joblib.dump(pipeline, GUEST_MODEL_FILE)
                sync_file_to_github(GUEST_MODEL_FILE, GUEST_MODEL_FILE, "Auto-sync: Updated guest model")
                st.success("🌐 Model trained and saved into public guest sandbox.")

# ==============================================================================
# PAGE 2: REAL-TIME YIELD & AUTONOMOUS RCO PREDICTION
# ==============================================================================
elif page == "2. Yield Prediction":
    st.header("🎯 Autonomous CDU Prediction & Dynamic Sensitivity")

    active_pipeline = None
    if st.session_state["authenticated"]:
        conn = get_db_connection()
        user_models_df = pd.read_sql_query(
            "SELECT id, model_tag, model_path FROM protected_models WHERE username = ?",
            conn, params=(st.session_state["username"],)
        )
        conn.close()

        source_choice = st.radio(
            "Prediction Model Source:", 
            ["Public Guest Sandbox Model", "My Private Vault Models"], 
            horizontal=True
        )

        if source_choice == "My Private Vault Models":
            if user_models_df.empty:
                st.warning("⚠️ You do not have any models in your private vault yet. You can predict with the 'Public Guest Sandbox Model' or train one on Page 1.")
                st.stop()
            else:
                tag_to_path = dict(zip(user_models_df["model_tag"], user_models_df["model_path"]))
                selected_tag = st.selectbox("Select Your Private Model:", options=list(tag_to_path.keys()))
                chosen_path = tag_to_path[selected_tag]
                
                if chosen_path and os.path.exists(chosen_path):
                    active_pipeline = joblib.load(chosen_path)
                else:
                    st.error("❌ Selected model artifact is missing from disk.")
                    st.stop()
        else:
            if os.path.exists(GUEST_MODEL_FILE):
                active_pipeline = joblib.load(GUEST_MODEL_FILE)
            else:
                st.warning("⚠️ Public guest model not found. Please train it on Page 1.")
                st.stop()
    else:
        if os.path.exists(GUEST_MODEL_FILE):
            active_pipeline = joblib.load(GUEST_MODEL_FILE)
        else:
            st.warning("⚠️ No public guest model available. Please train one on Page 1 or log in.")
            st.stop()

    if not active_pipeline:
        st.warning("⚠️ No trained model found. Please train a model on Page 1 first.")
    else:
        input_cols = active_pipeline["input_cols"]
        flow_targets = active_pipeline["flow_targets"]
        state_targets = active_pipeline["state_targets"]
        last_in = active_pipeline.get("last_known_inputs", {})
        stats = active_pipeline["baseline_stats"]

        st.subheader("1. Crude Assay Properties")
        c_dens1, c_dens2 = st.columns(2)
        last_api = float(last_in.get('crude_api', 34.0))
        default_density = float(141.5 / (last_api + 131.5) * 1000.0) if last_api else 855.0

        input_density = c_dens1.number_input("Crude Density (kg/m³ or SG @ 15°C)", value=default_density, format="%.2f")
        calculated_api = density_to_api(input_density)
        c_dens2.metric("Calculated Crude API", f"{calculated_api:.2f} °API")

        st.subheader("2. Operating Boundary Inputs (Initialized from Plant Runs)")
        input_data = {}
        cols = st.columns(3)
        for i, feat in enumerate(input_cols):
            if feat == 'crude_api':
                input_data[feat] = calculated_api
            else:
                fallback = last_in.get(feat, 350.0 if "cot" in feat.lower() else (1847.0 if "crude" in feat.lower() else (1.45 if "p_kgcm2" in feat.lower() else 13.5)))
                input_data[feat] = cols[i % 3].number_input(feat, value=float(fallback), format="%.2f")

        if st.button("🔮 Run Simulation & Predict", type="primary"):
            input_df = pd.DataFrame([input_data])
            norm_yields = active_pipeline["yield_model"].predict(input_df)[0]
            crude_in = input_data[active_pipeline["crude_col"]]
            pred_flows = norm_yields * crude_in

            scaled_state = active_pipeline["scaler_states"].transform(input_df)
            pred_states = active_pipeline["model_states"].predict(scaled_state)[0]

            # Dynamic thermal adjustments
            cot_delta = input_data.get('cot_degC', 350.0) - stats['mean_cot']
            fzp_delta = input_data.get('flash_zone_p_kgcm2', 1.50) - stats['mean_p']

            for k, s in enumerate(state_targets):
                if "top_temp" in s.lower():
                    pred_states[k] += 0.15 * cot_delta - 2.5 * fzp_delta
                elif "pa" in s.lower():
                    pred_states[k] += 0.80 * cot_delta

            # Autonomous RCO Recovery @ 360°C prediction
            pred_rco_rec_360 = float(active_pipeline["rco_model"].predict(input_df)[0])

            # STEP 5 (Hook 5): Auto-sync simulation runs to GitHub database
            if st.session_state["authenticated"]:
                conn = get_db_connection()
                outputs_to_save = dict(zip(flow_targets, pred_flows))
                outputs_to_save['rco_rec_360_vol_pct'] = pred_rco_rec_360
                conn.execute('''
                    INSERT INTO simulation_history (username, timestamp, inputs_json, outputs_json)
                    VALUES (?, ?, ?, ?)
                ''', (st.session_state["username"], datetime.now(), json.dumps(input_data), json.dumps(outputs_to_save)))
                conn.commit()
                conn.close()
                sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Logged run for {st.session_state['username']}")

            st.divider()
            c_left, c_right = st.columns(2)
            with c_left:
                st.subheader("📦 Product Recovery Yields")
                st.table(pd.DataFrame({
                    "Cut Stream": flow_targets,
                    "Yield (wt%)": [f"{y*100:.2f}%" for y in norm_yields],
                    "Rate (t/h)": [f"{f:.2f}" for f in pred_flows]
                }))
                st.metric("Total Mass Out", f"{np.sum(pred_flows):.2f} t/h")

            with c_right:
                st.subheader("🌡️ Predicted Column Profile")
                st.table(pd.DataFrame({
                    "Parameter": state_targets,
                    "Predicted Value": [f"{v:.2f} {'°C' if 'temp' in n.lower() else 't/h'}" for n, v in zip(state_targets, pred_states)]
                }))

            # ==============================================================
            # AUTONOMOUS RCO QUALITY @ 360°C PREDICTION
            # ==============================================================
            st.divider()
            st.subheader("🧪 Atmospheric Residue (RCO) Recovery @ 360°C")
            r_col1, r_col2, r_col3 = st.columns(3)
            
            r_col1.metric(
                label="Predicted RCO Recovery @ 360°C", 
                value=f"{pred_rco_rec_360:.2f} vol%"
            )

            if pred_rco_rec_360 <= 5.00:
                sep_status = "🟢 Deep Cut / Minimal Slippage"
            elif pred_rco_rec_360 <= 7.50:
                sep_status = "🟡 Typical Operational Stripping"
            else:
                sep_status = "🔴 High Distillate Loss in Residue"

            r_col2.metric(label="Residue Stripping Quality", value=sep_status)
            r_col3.metric(label="Stripping Steam Ratio", value=f"{(input_data.get('stripping_steam_flow', 14.0) / crude_in * 1000):.2f} kg/t crude")

            # ==============================================================
            # DEDICATED GRAPHS: FEED, RECOVERY & RCO SENSITIVITY
            # ==============================================================
            st.divider()
            st.subheader("📊 Product Yields & RCO Recovery Analytics")

            label_map = {
                'flow_offgas': 'Off-Gas & LPG',
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
                chart_data = plot_df.set_index("Product Cut")[["Mass Flow (t/h)"]]
                st.bar_chart(chart_data, color="#0066cc")

            with g_col2:
                st.markdown("**Yield Fraction Breakdown (% Recovery)**")
                yield_data = plot_df.set_index("Product Cut")[["Yield Share (%)"]]
                st.bar_chart(yield_data, color="#ff7f0e")

            # RCO Sensitivity Sweep Curve
            st.markdown("### 📈 RCO Recovery @ 360°C vs Furnace COT")
            base_cot = float(input_data.get('cot_degC', stats['mean_cot']))
            cot_sweep = np.linspace(base_cot - 15.0, base_cot + 15.0, 31)

            rco_sweep_vals = []
            yield_sweep_vals = []
            for temp in cot_sweep:
                temp_input = input_data.copy()
                temp_input['cot_degC'] = temp
                temp_df = pd.DataFrame([temp_input])
                rco_sweep_vals.append(float(active_pipeline["rco_model"].predict(temp_df)[0]))
                yield_sweep_vals.append(active_pipeline["yield_model"].predict(temp_df)[0] * 100.0)

            rco_curve_df = pd.DataFrame(
                {"RCO Rec @ 360°C (vol%)": rco_sweep_vals},
                index=np.round(cot_sweep, 1)
            )
            rco_curve_df.index.name = "Furnace COT (°C)"
            st.line_chart(rco_curve_df, color="#d62728")
            st.caption("Illustrates how increased thermal severity reduces distillate slippage in atmospheric bottoms.")

            # Dynamic Overall Distillate Yield Curves
            with st.expander("📈 Dynamic Distillate Cut Yield Curves (Yield % vs COT)", expanded=False):
                sensitivity_df = pd.DataFrame(
                    np.array(yield_sweep_vals), 
                    columns=display_labels, 
                    index=np.round(cot_sweep, 1)
                )
                sensitivity_df.index.name = "Furnace COT (°
