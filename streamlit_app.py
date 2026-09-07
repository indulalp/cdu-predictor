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
    'flow_offgas': (0.100, 0.250),    # ~15% to 20%
    'flow_naphtha': (0.020, 0.120),   # ~4% to 9%
    'flow_kero': (0.040, 0.100),      # ~5% to 8%
    'flow_lago': (0.150, 0.280),      # ~18% to 24%
    'flow_residue': (0.400, 0.650)    # ~48% to 58%
}

PHYSICS_SLOPES = {
    'flow_offgas': 0.00030,
    'flow_naphtha': 0.00045,
    'flow_kero': 0.00025,
    'flow_lago': 0.00100,
    'flow_residue': -0.00200
}

# ==============================================================================
# EMBEDDED EXCEL DATASET (cdu data.xlsx)
# ==============================================================================
EXCEL_RECORDS = [
    {"crude_flow": 1939.92, "crude_density": 873.8, "crude_api": 30.420176, "sulphur_wt_pct": 2.13, "cot_degC": 350.280, "flash_zone_p_kgcm2": 1.51, "stripping_steam_flow": 18.20, "top_pa": 436.93, "kero_pa_flow_tph": 406.19, "lago_pa_flow_tph": 1041.17, "top_temp_degC": 117.99, "lago_d86_95_degC": 251.44, "flow_offgas": 295.632125, "flow_naphtha": 86.17, "flow_kero": 113.28, "flow_lago": 391.03, "flow_residue": 1072.25},
    {"crude_flow": 1933.65, "crude_density": 875.2, "crude_api": 30.161163, "sulphur_wt_pct": 2.08, "cot_degC": 349.760, "flash_zone_p_kgcm2": 1.51, "stripping_steam_flow": 18.63, "top_pa": 424.83, "kero_pa_flow_tph": 394.05, "lago_pa_flow_tph": 1029.96, "top_temp_degC": 117.47, "lago_d86_95_degC": 251.60, "flow_offgas": 290.379661, "flow_naphtha": 84.27, "flow_kero": 113.09, "flow_lago": 390.26, "flow_residue": 1070.22},
    {"crude_flow": 1935.35, "crude_density": 875.8, "crude_api": 30.050411, "sulphur_wt_pct": 2.23, "cot_degC": 350.550, "flash_zone_p_kgcm2": 1.53, "stripping_steam_flow": 18.81, "top_pa": 444.88, "kero_pa_flow_tph": 412.42, "lago_pa_flow_tph": 1039.50, "top_temp_degC": 118.33, "lago_d86_95_degC": 252.17, "flow_offgas": 301.855768, "flow_naphtha": 95.36, "flow_kero": 113.84, "flow_lago": 394.83, "flow_residue": 1053.10},
    {"crude_flow": 1837.70, "crude_density": 872.0, "crude_api": 30.754415, "sulphur_wt_pct": 2.29, "cot_degC": 349.275, "flash_zone_p_kgcm2": 1.55, "stripping_steam_flow": 18.49, "top_pa": 443.77, "kero_pa_flow_tph": 403.30, "lago_pa_flow_tph": 1039.09, "top_temp_degC": 118.24, "lago_d86_95_degC": 250.06, "flow_offgas": 307.525607, "flow_naphtha": 96.48, "flow_kero": 111.62, "flow_lago": 375.93, "flow_residue": 974.26},
    {"crude_flow": 1918.65, "crude_density": 871.0, "crude_api": 30.940700, "sulphur_wt_pct": 2.16, "cot_degC": 349.340, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 16.64, "top_pa": 471.26, "kero_pa_flow_tph": 424.19, "lago_pa_flow_tph": 1043.68, "top_temp_degC": 120.68, "lago_d86_95_degC": 250.92, "flow_offgas": 328.723607, "flow_naphtha": 85.84, "flow_kero": 115.53, "flow_lago": 376.69, "flow_residue": 1015.18},
    {"crude_flow": 1940.04, "crude_density": 870.0, "crude_api": 31.127414, "sulphur_wt_pct": 2.18, "cot_degC": 350.100, "flash_zone_p_kgcm2": 1.59, "stripping_steam_flow": 17.27, "top_pa": 469.60, "kero_pa_flow_tph": 423.48, "lago_pa_flow_tph": 1044.65, "top_temp_degC": 119.95, "lago_d86_95_degC": 251.21, "flow_offgas": 323.873455, "flow_naphtha": 90.53, "flow_kero": 115.98, "flow_lago": 379.93, "flow_residue": 1013.62},
    {"crude_flow": 1936.03, "crude_density": 873.8, "crude_api": 30.420176, "sulphur_wt_pct": 2.28, "cot_degC": 350.660, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 18.22, "top_pa": 470.28, "kero_pa_flow_tph": 425.54, "lago_pa_flow_tph": 1044.58, "top_temp_degC": 120.09, "lago_d86_95_degC": 252.01, "flow_offgas": 324.193152, "flow_naphtha": 88.82, "flow_kero": 115.99, "flow_lago": 384.94, "flow_residue": 1018.69},
    {"crude_flow": 1940.45, "crude_density": 871.8, "crude_api": 30.791638, "sulphur_wt_pct": 2.22, "cot_degC": 349.280, "flash_zone_p_kgcm2": 1.61, "stripping_steam_flow": 17.60, "top_pa": 490.76, "kero_pa_flow_tph": 448.00, "lago_pa_flow_tph": 1044.44, "top_temp_degC": 122.43, "lago_d86_95_degC": 251.69, "flow_offgas": 341.000527, "flow_naphtha": 106.06, "flow_kero": 115.95, "flow_lago": 391.39, "flow_residue": 1000.19},
    {"crude_flow": 1927.76, "crude_density": 872.8, "crude_api": 30.605694, "sulphur_wt_pct": 2.35, "cot_degC": 346.635, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 15.52, "top_pa": 480.80, "kero_pa_flow_tph": 432.63, "lago_pa_flow_tph": 1044.65, "top_temp_degC": 123.96, "lago_d86_95_degC": 249.74, "flow_offgas": 347.207295, "flow_naphtha": 115.16, "flow_kero": 116.07, "flow_lago": 374.13, "flow_residue": 991.19},
    {"crude_flow": 1938.39, "crude_density": 866.0, "crude_api": 31.878580, "sulphur_wt_pct": 2.35, "cot_degC": 346.470, "flash_zone_p_kgcm2": 1.57, "stripping_steam_flow": 16.19, "top_pa": 468.26, "kero_pa_flow_tph": 419.14, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 122.32, "lago_d86_95_degC": 249.55, "flow_offgas": 338.476313, "flow_naphtha": 106.01, "flow_kero": 115.96, "flow_lago": 377.64, "flow_residue": 1007.09},
    {"crude_flow": 1937.52, "crude_density": 870.4, "crude_api": 31.052677, "sulphur_wt_pct": 2.35, "cot_degC": 345.925, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 17.06, "top_pa": 472.93, "kero_pa_flow_tph": 427.50, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 122.28, "lago_d86_95_degC": 249.51, "flow_offgas": 340.505672, "flow_naphtha": 105.00, "flow_kero": 116.03, "flow_lago": 378.89, "flow_residue": 1009.61},
    {"crude_flow": 1941.00, "crude_density": 869.6, "crude_api": 31.202334, "sulphur_wt_pct": 2.29, "cot_degC": 347.165, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 17.58, "top_pa": 482.02, "kero_pa_flow_tph": 435.53, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 123.08, "lago_d86_95_degC": 250.05, "flow_offgas": 346.529244, "flow_naphtha": 110.15, "flow_kero": 116.00, "flow_lago": 379.74, "flow_residue": 1004.89},
    {"crude_flow": 1937.93, "crude_density": 869.2, "crude_api": 31.277266, "sulphur_wt_pct": 2.19, "cot_degC": 347.935, "flash_zone_p_kgcm2": 1.54, "stripping_steam_flow": 16.74, "top_pa": 483.99, "kero_pa_flow_tph": 438.30, "lago_pa_flow_tph": 1044.97, "top_temp_degC": 122.95, "lago_d86_95_degC": 250.78, "flow_offgas": 348.064560, "flow_naphtha": 110.16, "flow_kero": 116.00, "flow_lago": 382.49, "flow_residue": 996.65},
    {"crude_flow": 1939.84, "crude_density": 866.6, "crude_api": 31.765636, "sulphur_wt_pct": 2.28, "cot_degC": 347.880, "flash_zone_p_kgcm2": 1.57, "stripping_steam_flow": 17.20, "top_pa": 492.20, "kero_pa_flow_tph": 444.60, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 123.49, "lago_d86_95_degC": 251.68, "flow_offgas": 352.333830, "flow_naphtha": 115.89, "flow_kero": 115.93, "flow_lago": 384.77, "flow_residue": 989.47},
    {"crude_flow": 1936.56, "crude_density": 870.4, "crude_api": 31.052677, "sulphur_wt_pct": 2.29, "cot_degC": 347.330, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 16.71, "top_pa": 484.54, "kero_pa_flow_tph": 435.59, "lago_pa_flow_tph": 1044.89, "top_temp_degC": 123.51, "lago_d86_95_degC": 250.91, "flow_offgas": 347.854086, "flow_naphtha": 112.56, "flow_kero": 115.98, "flow_lago": 380.20, "flow_residue": 996.86},
    {"crude_flow": 1941.52, "crude_density": 874.0, "crude_api": 30.383181, "sulphur_wt_pct": 2.27, "cot_degC": 347.455, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.21, "top_pa": 496.07, "kero_pa_flow_tph": 443.85, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 124.77, "lago_d86_95_degC": 251.05, "flow_offgas": 356.126569, "flow_naphtha": 123.00, "flow_kero": 115.98, "flow_lago": 381.71, "flow_residue": 980.99},
    {"crude_flow": 1939.92, "crude_density": 875.0, "crude_api": 30.198057, "sulphur_wt_pct": 2.30, "cot_degC": 347.600, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 17.37, "top_pa": 490.15, "kero_pa_flow_tph": 435.53, "lago_pa_flow_tph": 1044.97, "top_temp_degC": 124.74, "lago_d86_95_degC": 251.49, "flow_offgas": 353.473595, "flow_naphtha": 120.31, "flow_kero": 115.98, "flow_lago": 383.69, "flow_residue": 984.77},
    {"crude_flow": 1940.40, "crude_density": 873.8, "crude_api": 30.420176, "sulphur_wt_pct": 2.21, "cot_degC": 347.635, "flash_zone_p_kgcm2": 1.57, "stripping_steam_flow": 16.96, "top_pa": 497.02, "kero_pa_flow_tph": 445.69, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 125.10, "lago_d86_95_degC": 251.52, "flow_offgas": 356.162386, "flow_naphtha": 122.37, "flow_kero": 115.99, "flow_lago": 383.95, "flow_residue": 980.37},
    {"crude_flow": 1935.53, "crude_density": 874.0, "crude_api": 30.383181, "sulphur_wt_pct": 2.26, "cot_degC": 347.465, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.47, "top_pa": 497.23, "kero_pa_flow_tph": 443.08, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 125.79, "lago_d86_95_degC": 251.27, "flow_offgas": 357.513488, "flow_naphtha": 128.47, "flow_kero": 115.99, "flow_lago": 382.72, "flow_residue": 971.86},
    {"crude_flow": 1940.21, "crude_density": 876.0, "crude_api": 30.013584, "sulphur_wt_pct": 2.21, "cot_degC": 347.385, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 17.10, "top_pa": 500.41, "kero_pa_flow_tph": 449.19, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 125.79, "lago_d86_95_degC": 250.77, "flow_offgas": 357.771963, "flow_naphtha": 127.35, "flow_kero": 116.03, "flow_lago": 380.60, "flow_residue": 972.19},
    {"crude_flow": 1941.13, "crude_density": 875.0, "crude_api": 30.198057, "sulphur_wt_pct": 2.23, "cot_degC": 347.795, "flash_zone_p_kgcm2": 1.57, "stripping_steam_flow": 16.97, "top_pa": 503.20, "kero_pa_flow_tph": 449.98, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 125.99, "lago_d86_95_degC": 250.99, "flow_offgas": 359.851080, "flow_naphtha": 128.43, "flow_kero": 115.98, "flow_lago": 382.68, "flow_residue": 970.61},
    {"crude_flow": 1937.66, "crude_density": 873.4, "crude_api": 30.494275, "sulphur_wt_pct": 2.22, "cot_degC": 347.880, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.43, "top_pa": 501.99, "kero_pa_flow_tph": 451.98, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 126.11, "lago_d86_95_degC": 251.10, "flow_offgas": 359.971253, "flow_naphtha": 128.91, "flow_kero": 116.00, "flow_lago": 383.04, "flow_residue": 970.21},
    {"crude_flow": 1935.21, "crude_density": 875.4, "crude_api": 30.124286, "sulphur_wt_pct": 2.20, "cot_degC": 347.500, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.20, "top_pa": 500.41, "kero_pa_flow_tph": 449.61, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 126.10, "lago_d86_95_degC": 250.77, "flow_offgas": 360.596009, "flow_naphtha": 130.00, "flow_kero": 116.01, "flow_lago": 381.16, "flow_residue": 967.65},
    {"crude_flow": 1941.13, "crude_density": 876.4, "crude_api": 29.939970, "sulphur_wt_pct": 2.14, "cot_degC": 347.460, "flash_zone_p_kgcm2": 1.57, "stripping_steam_flow": 16.92, "top_pa": 500.56, "kero_pa_flow_tph": 448.91, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 126.10, "lago_d86_95_degC": 250.41, "flow_offgas": 361.353386, "flow_naphtha": 131.00, "flow_kero": 116.02, "flow_lago": 380.05, "flow_residue": 968.74},
    {"crude_flow": 1940.36, "crude_density": 878.0, "crude_api": 29.646355, "sulphur_wt_pct": 2.11, "cot_degC": 347.785, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.06, "top_pa": 503.73, "kero_pa_flow_tph": 450.00, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 126.23, "lago_d86_95_degC": 250.70, "flow_offgas": 362.464687, "flow_naphtha": 132.00, "flow_kero": 115.98, "flow_lago": 381.47, "flow_residue": 966.50},
    {"crude_flow": 1938.86, "crude_density": 876.0, "crude_api": 30.013584, "sulphur_wt_pct": 2.21, "cot_degC": 347.750, "flash_zone_p_kgcm2": 1.57, "stripping_steam_flow": 17.20, "top_pa": 503.11, "kero_pa_flow_tph": 450.00, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 126.30, "lago_d86_95_degC": 250.84, "flow_offgas": 363.090623, "flow_naphtha": 133.00, "flow_kero": 116.00, "flow_lago": 382.49, "flow_residue": 964.81},
    {"crude_flow": 1939.96, "crude_density": 874.0, "crude_api": 30.383181, "sulphur_wt_pct": 2.24, "cot_degC": 347.735, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.20, "top_pa": 503.00, "kero_pa_flow_tph": 449.62, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 126.31, "lago_d86_95_degC": 250.93, "flow_offgas": 364.088656, "flow_naphtha": 134.00, "flow_kero": 116.00, "flow_lago": 383.00, "flow_residue": 963.88},
    {"crude_flow": 1940.35, "crude_density": 874.4, "crude_api": 30.309355, "sulphur_wt_pct": 2.21, "cot_degC": 347.925, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.20, "top_pa": 505.28, "kero_pa_flow_tph": 450.00, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 126.47, "lago_d86_95_degC": 251.24, "flow_offgas": 366.195610, "flow_naphtha": 136.00, "flow_kero": 116.00, "flow_lago": 384.60, "flow_residue": 961.43},
    {"crude_flow": 1939.06, "crude_density": 877.0, "crude_api": 29.829989, "sulphur_wt_pct": 2.23, "cot_degC": 347.905, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.20, "top_pa": 507.03, "kero_pa_flow_tph": 450.00, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 126.68, "lago_d86_95_degC": 251.46, "flow_offgas": 368.513728, "flow_naphtha": 139.00, "flow_kero": 116.00, "flow_lago": 385.66, "flow_residue": 957.51},
    {"crude_flow": 1937.89, "crude_density": 877.0, "crude_api": 29.829989, "sulphur_wt_pct": 2.22, "cot_degC": 348.065, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.20, "top_pa": 512.44, "kero_pa_flow_tph": 450.00, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 127.34, "lago_d86_95_degC": 251.81, "flow_offgas": 372.457894, "flow_naphtha": 147.00, "flow_kero": 116.00, "flow_lago": 386.43, "flow_residue": 948.55},
    {"crude_flow": 1939.46, "crude_density": 874.0, "crude_api": 30.383181, "sulphur_wt_pct": 2.25, "cot_degC": 348.160, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.20, "top_pa": 515.65, "kero_pa_flow_tph": 450.00, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 127.81, "lago_d86_95_degC": 252.02, "flow_offgas": 375.986326, "flow_naphtha": 154.00, "flow_kero": 116.00, "flow_lago": 387.03, "flow_residue": 940.54},
    {"crude_flow": 1938.83, "crude_density": 874.0, "crude_api": 30.383181, "sulphur_wt_pct": 2.22, "cot_degC": 348.245, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.20, "top_pa": 518.73, "kero_pa_flow_tph": 450.00, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 128.26, "lago_d86_95_degC": 252.12, "flow_offgas": 381.925482, "flow_naphtha": 159.69, "flow_kero": 116.00, "flow_lago": 387.71, "flow_residue": 932.92}
]

def get_demo_dataframe():
    return pd.DataFrame(EXCEL_RECORDS)

# ==============================================================================
# STEP 4: GITHUB AUTO-COMMIT SYNCHRONIZATION ENGINE
# ==============================================================================
def sync_file_to_github(local_file_path, repo_file_path, commit_message="Auto-sync from Streamlit App"):
    """Pushes or updates a file directly in your GitHub repository via API."""
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
# DATABASE & ACCESS CONTROL LAYER (WITH STEP 5 AUTO-SYNC HOOKS)
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
# PHYSICS-INFORMED CONTINUOUS ENGINE
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
        api = X_df['crude_api'].values if 'crude_api' in X_df else self.baseline_stats.get('mean_api', 30.5)

        delta_severity = (
            (cot - self.baseline_stats['mean_cot'])
            - 16.5 * (fzp - self.baseline_stats['mean_p'])
            + 0.55 * (steam - self.baseline_stats['mean_steam'])
            + 0.85 * (api - self.baseline_stats.get('mean_api', 30.5))
        )

        final_yields = np.zeros_like(base_yields)
        for i, col in enumerate(self.flow_targets):
            slope = PHYSICS_SLOPES.get(col, 0.0)
            adj = base_yields[:, i] + (slope * delta_severity)
            min_b, max_b = YIELD_BOUNDS.get(col, (0.01, 0.90))
            final_yields[:, i] = np.clip(adj, min_b, max_b)

        return final_yields / np.sum(final_yields, axis=1, keepdims=True)

# ==============================================================================
# DEFAULT MODEL INITIALIZATION (ALWAYS TRAINED ON STARTUP)
# ==============================================================================
def train_default_guest_model_if_missing():
    if not os.path.exists(GUEST_MODEL_FILE):
        df = get_demo_dataframe()
        input_cols = ['crude_flow', 'crude_api', 'sulphur_wt_pct', 'cot_degC', 'flash_zone_p_kgcm2', 'stripping_steam_flow', 'lago_d86_95_degC']
        flow_cols = ['flow_offgas', 'flow_naphtha', 'flow_kero', 'flow_lago', 'flow_residue']
        state_cols = ['top_pa', 'kero_pa_flow_tph', 'lago_pa_flow_tph', 'top_temp_degC']
        crude_col = 'crude_flow'

        yield_targets = df[flow_cols].div(df[crude_col], axis=0)
        baseline_stats = {
            'mean_cot': float(df['cot_degC'].mean()),
            'mean_p': float(df['flash_zone_p_kgcm2'].mean()),
            'mean_steam': float(df['stripping_steam_flow'].mean()),
            'mean_api': float(df['crude_api'].mean())
        }

        yield_model = PhysicsInformedYieldModel()
        yield_model.fit(df[input_cols], yield_targets, flow_cols, baseline_stats)

        scaler_states = StandardScaler()
        model_states = MultiOutputRegressor(LGBMRegressor(n_estimators=250, learning_rate=0.03, random_state=42))
        model_states.fit(scaler_states.fit_transform(df[input_cols]), df[state_cols])

        pipeline = {
            "yield_model": yield_model,
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
# PAGE 1: GUEST & MEMBER TRAINING INTERFACE
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
        st.session_state['active_src_name'] = "Loaded CDU Refinery Historical Dataset (32 DCS runs)"

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

    default_inputs = ['crude_flow', 'crude_api', 'sulphur_wt_pct', 'cot_degC', 'flash_zone_p_kgcm2', 'stripping_steam_flow', 'lago_d86_95_degC']
    default_flows = ['flow_offgas', 'flow_naphtha', 'flow_kero', 'flow_lago', 'flow_residue']
    default_states = ['top_pa', 'kero_pa_flow_tph', 'lago_pa_flow_tph', 'top_temp_degC']

    c1, c2, c3 = st.columns(3)
    input_cols = c1.multiselect("Inputs (X)", list(df.columns), default=[c for c in default_inputs if c in df.columns])
    flow_cols = c2.multiselect("Product Flows (Y1)", list(df.columns), default=[c for c in default_flows if c in df.columns])
    state_cols = c3.multiselect("Internal States (Y2)", list(df.columns), default=[c for c in default_states if c in df.columns])
    crude_col = c1.selectbox("Crude Inlet Flow Tag", list(df.columns), index=list(df.columns).index('crude_flow') if 'crude_flow' in df.columns else 0)

    save_as_protected = False
    model_tag = "guest_model"
    if st.session_state["authenticated"]:
        st.divider()
        c_save1, c_save2 = st.columns([1, 2])
        save_as_protected = c_save1.checkbox("Save model into my protected private vault", value=True)
        if save_as_protected:
            model_tag = c_save2.text_input("Protected Model Tag", value=f"{st.session_state['username']}_v1")

    if st.button("🚀 Train Digital Twin", type="primary"):
        with st.spinner("Training model with continuous thermodynamic gradients..."):
            all_needed = list(set(input_cols + flow_cols + state_cols + [crude_col]))
            clean_df = df[all_needed].apply(pd.to_numeric, errors='coerce').dropna()

            total_out = clean_df[flow_cols].sum(axis=1)
            valid_df = clean_df[np.abs(total_out - clean_df[crude_col]) / clean_df[crude_col] < 0.05].copy()

            if valid_df.empty:
                st.error("❌ Mass balance error: Data does not close within 5%.")
                st.stop()

            yield_targets = valid_df[flow_cols].div(valid_df[crude_col], axis=0)
            X_tr, X_te, yf_tr, yf_te, ys_tr, ys_te, c_tr, c_te = train_test_split(
                valid_df[input_cols], yield_targets, valid_df[state_cols], valid_df[crude_col], test_size=0.2, random_state=42
            )

            baseline_stats = {
                'mean_cot': float(valid_df['cot_degC'].mean()) if 'cot_degC' in valid_df.columns else 348.0,
                'mean_p': float(valid_df['flash_zone_p_kgcm2'].mean()) if 'flash_zone_p_kgcm2' in valid_df.columns else 1.56,
                'mean_steam': float(valid_df['stripping_steam_flow'].mean()) if 'stripping_steam_flow' in valid_df.columns else 17.2,
                'mean_api': float(valid_df['crude_api'].mean()) if 'crude_api' in valid_df.columns else 30.5
            }

            yield_model = PhysicsInformedYieldModel()
            yield_model.fit(X_tr, yf_tr, flow_cols, baseline_stats)

            scaler_states = StandardScaler()
            model_states = MultiOutputRegressor(LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42))
            model_states.fit(scaler_states.fit_transform(X_tr), ys_tr)

            pipeline = {
                "yield_model": yield_model,
                "model_states": model_states,
                "scaler_states": scaler_states,
                "input_cols": input_cols,
                "flow_targets": flow_cols,
                "state_targets": state_cols,
                "crude_col": crude_col,
                "baseline_stats": baseline_stats,
                "training_rows": len(valid_df),
                "last_known_inputs": clean_df[input_cols].iloc[-1].to_dict()
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
# PAGE 2: REAL-TIME PREDICTION (USER ISOLATED)
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

        if user_models_df.empty:
            st.warning("⚠️ You do not have any models in your private vault yet. Please train and save one on Page 1 first.")
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
        # Unauthenticated guests default to the sandbox model
        if os.path.exists(GUEST_MODEL_FILE):
            active_pipeline = joblib.load(GUEST_MODEL_FILE)
        else:
            st.warning("⚠️ No default model available. Please train one on Page 1 or log in.")
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
        last_api = float(last_in.get('crude_api', 30.5))
        default_density = float(141.5 / (last_api + 131.5) * 1000.0) if last_api else 874.0

        input_density = c_dens1.number_input("Crude Density (kg/m³ or SG @ 15°C)", value=default_density, format="%.2f")
        calculated_api = density_to_api(input_density)
        c_dens2.metric("Calculated Crude API", f"{calculated_api:.2f} °API")

        st.subheader("2. Operating Boundary Inputs (Last Input Initialized)")
        input_data = {}
        cols = st.columns(3)
        for i, feat in enumerate(input_cols):
            if feat == 'crude_api':
                input_data[feat] = calculated_api
            else:
                fallback = last_in.get(feat, 348.0 if "cot" in feat.lower() else (1939.0 if "crude" in feat.lower() else (1.56 if "p_kgcm2" in feat.lower() else 17.2)))
                input_data[feat] = cols[i % 3].number_input(feat, value=float(fallback), format="%.2f")

        if st.button("🔮 Run Simulation & Predict", type="primary"):
            input_df = pd.DataFrame([input_data])
            norm_yields = active_pipeline["yield_model"].predict(input_df)[0]
            crude_in = input_data[active_pipeline["crude_col"]]
            pred_flows = norm_yields * crude_in

            scaled_state = active_pipeline["scaler_states"].transform(input_df)
            pred_states = active_pipeline["model_states"].predict(scaled_state)[0]

            # Dynamic thermal adjustments
            cot_delta = input_data.get('cot_degC', 348.0) - stats['mean_cot']
            fzp_delta = input_data.get('flash_zone_p_kgcm2', 1.56) - stats['mean_p']

            for k, s in enumerate(state_targets):
                if "top_temp" in s.lower():
                    pred_states[k] += 0.15 * cot_delta - 2.5 * fzp_delta
                elif "pa" in s.lower():
                    pred_states[k] += 0.80 * cot_delta

            # STEP 5 (Hook 5): Auto-sync simulation runs to GitHub database
            if st.session_state["authenticated"]:
                conn = get_db_connection()
                conn.execute('''
                    INSERT INTO simulation_history (username, timestamp, inputs_json, outputs_json)
                    VALUES (?, ?, ?, ?)
                ''', (st.session_state["username"], datetime.now(), json.dumps(input_data), json.dumps(dict(zip(flow_targets, pred_flows)))))
                conn.commit()
                conn.close()
                sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Logged run for {st.session_state['username']}")

            st.divider()
            c_left, c_right = st.columns(2)
            with c_left:
                st.subheader(" Product Recovery Yields")
                st.table(pd.DataFrame({
                    "Cut Stream": flow_targets,
                    "Yield (wt%)": [f"{y*100:.2f}%" for y in norm_yields],
                    "Rate (t/h)": [f"{f:.2f}" for f in pred_flows]
                }))
                st.metric("Total Mass Out", f"{np.sum(pred_flows):.2f} t/h (Closure: 0.00% error)")

            with c_right:
                st.subheader(" Predicted Column Profile")
                st.table(pd.DataFrame({
                    "Parameter": state_targets,
                    "Predicted Value": [f"{v:.2f} {'°C' if 'temp' in n.lower() else 't/h'}" for n, v in zip(state_targets, pred_states)]
                }))
            # ==============================================================
            # GRAPHICAL VISUALIZATIONS: FEED & PRODUCT RECOVERY
            # ==============================================================
            st.divider()
            st.subheader("📊 Feed & Recovery Analytics")

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

            with st.expander("📈 Dynamic Furnace Sensitivity Curve (Yield % vs COT)", expanded=True):
                base_cot = float(input_data.get('cot_degC', stats['mean_cot']))
                cot_sweep = np.linspace(base_cot - 15.0, base_cot + 15.0, 31)
                
                sweep_yields = []
                for temp in cot_sweep:
                    temp_input = input_data.copy()
                    temp_input['cot_degC'] = temp
                    temp_df = pd.DataFrame([temp_input])
                    y_pred = active_pipeline["yield_model"].predict(temp_df)[0]
                    sweep_yields.append(y_pred * 100.0)

                sweep_matrix = np.array(sweep_yields)
                sensitivity_df = pd.DataFrame(
                    sweep_matrix, 
                    columns=display_labels, 
                    index=np.round(cot_sweep, 1)
                )
                sensitivity_df.index.name = "Furnace COT (°C)"
                
                st.line_chart(sensitivity_df)
                st.caption("Shows physical yield shifts across a ±15°C COT range holding feed assay and pressure steady.")
# ==============================================================================
# PAGE 3: PROTECTED WORKSPACE (MEMBER EXCLUSIVE)
# ==============================================================================
elif page == "3. Protected Workspace & History":
    st.header(f"🔒 Protected Workspace: `{st.session_state['username']}`")
    conn = get_db_connection()

    tab_my_models, tab_my_sims = st.tabs(["📁 My Saved Models", "📜 My Simulation History"])

    with tab_my_models:
        st.subheader("Your Isolated Models")
        my_models = pd.read_sql_query(
            "SELECT id, model_tag, training_rows, created_at, model_path FROM protected_models WHERE username = ? ORDER BY created_at DESC",
            conn, params=(st.session_state['username'],)
        )
        if my_models.empty:
            st.info("No protected models saved yet. Train one on Page 1 while signed in.")
        else:
            st.dataframe(my_models, use_container_width=True)

    with tab_my_sims:
        st.subheader("Your Saved Simulations")
        my_sims = pd.read_sql_query(
            "SELECT timestamp, inputs_json, outputs_json FROM simulation_history WHERE username = ? ORDER BY timestamp DESC",
            conn, params=(st.session_state['username'],)
        )
        if my_sims.empty:
            st.info("No logged simulations on record.")
        else:
            st.dataframe(my_sims, use_container_width=True)

    conn.close()

# ==============================================================================
# PAGE 4: ADMIN GOVERNANCE & TELEMETRY (ADMIN EXCLUSIVE)
# ==============================================================================
elif page == "🛡️ Admin Audit & Telemetry":
    st.header("🛡️ Enterprise Client Governance & Audit Portal")
    conn = get_db_connection()

    tab_manage, tab_client_inspect, tab_logs = st.tabs([
        "👥 Client Account Provisioning", 
        "🔍 Inspect & Modify Client Spaces", 
        "📍 Access & Location Audit"
    ])

    with tab_manage:
        st.subheader("Create New Client Account")
        c_u1, c_u2, c_u3 = st.columns([2, 2, 1])
        new_client_user = c_u1.text_input("New Client Username", placeholder="e.g. refinery_client_a")
        new_client_pass = c_u2.text_input("Initial Password", placeholder="e.g. Pass@2026")
        
        if c_u3.button("Create Account", type="primary"):
            if new_client_user and new_client_pass:
                ok, msg = create_client_user(new_client_user, new_client_pass)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("Please provide both username and password.")

        st.divider()
        st.subheader("Existing Accounts")
        users_df = pd.read_sql_query("SELECT username, role FROM users ORDER BY role ASC, username ASC", conn)
        st.dataframe(users_df, use_container_width=True)

        st.markdown("#### Password Reset / Account Management")
        client_list = [u for u in users_df["username"].tolist() if u != "admin"]
        if client_list:
            c_sel, c_np, c_btn1, c_btn2 = st.columns([2, 2, 1, 1])
            selected_client = c_sel.selectbox("Select Client", options=client_list)
            reset_pw = c_np.text_input("New Password", placeholder="Enter new password")
            
            if c_btn1.button("Reset Password"):
                if reset_pw:
                    reset_client_password(selected_client, reset_pw)
                    st.success(f"Password updated for `{selected_client}`.")
                else:
                    st.error("Enter a valid password.")

            if c_btn2.button("Delete Client", type="secondary"):
                delete_client_user(selected_client)
                st.warning(f"Client `{selected_client}` deleted.")
                st.rerun()

    with tab_client_inspect:
        st.subheader("Client Data Inspector & Editor")
        all_clients = [u for u in users_df["username"].tolist() if u != "admin"]

        if not all_clients:
            st.info("No registered clients available to inspect.")
        else:
            chosen_user = st.selectbox("Select Client Profile to Inspect:", options=all_clients)
            
            col_m, col_s = st.columns(2)
            with col_m:
                st.markdown(f"**Models Saved by `{chosen_user}`:**")
                client_models = pd.read_sql_query(
                    "SELECT id, model_tag, training_rows, created_at, model_path FROM protected_models WHERE username = ?",
                    conn, params=(chosen_user,)
                )
                if client_models.empty:
                    st.caption("No models saved by this client.")
                else:
                    st.dataframe(client_models, use_container_width=True)
                    del_m_id = st.selectbox("Delete Model ID", options=client_models["id"].tolist(), key="del_m_key")
                    if st.button("Delete Selected Model", key="del_m_btn"):
                        m_row = client_models[client_models["id"] == del_m_id].iloc[0]
                        if os.path.exists(m_row["model_path"]):
                            os.remove(m_row["model_path"])
                        conn.execute("DELETE FROM protected_models WHERE id = ?", (del_m_id,))
                        conn.commit()
                        sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Admin removed model {del_m_id}")
                        st.success(f"Removed model `{m_row['model_tag']}`.")
                        st.rerun()

            with col_s:
                st.markdown(f"**Simulations Run by `{chosen_user}`:**")
                client_sims = pd.read_sql_query(
                    "SELECT timestamp, inputs_json, outputs_json FROM simulation_history WHERE username = ? ORDER BY timestamp DESC",
                    conn, params=(chosen_user,)
                )
                if client_sims.empty:
                    st.caption("No simulations logged for this client.")
                else:
                    st.dataframe(client_sims, use_container_width=True)

    with tab_logs:
        st.subheader("Global Sign-in Geolocation & Telemetry")
        access_df = pd.read_sql_query("SELECT username, login_time, ip_address, city, region, country FROM access_logs ORDER BY login_time DESC", conn)
        st.dataframe(access_df, use_container_width=True)

    conn.close()
