"""Convert machine sensor readings into symptoms."""

import numpy as np
import pandas as pd

THRESHOLDS = {
    "low_temp_diff_K": 8.6,          
    "low_speed_rpm": 1380,         
    "low_power_W": 3500,            
    "high_power_W": 9000,          
    "high_wear_min": 200,           
    "strain_limit": {"L": 11000, "M": 12000, "H": 13000},  
}

COLUMNS = ["udi", "pid", "type", "air", "proc", "rpm", "torque", "wear",
           "fail", "TWF", "HDF", "PWF", "OSF", "RNF"]

# load dataset and add derived columns
def load_data(path="data/ai4i2020.csv"):
    df = pd.read_csv(path)
    df.columns = COLUMNS
    return add_derived(df)


def add_derived(df):
    df = df.copy()
    df["dT"] = df["proc"] - df["air"]                        # K
    df["power"] = df["torque"] * df["rpm"] * 2 * np.pi / 60  # W
    df["strain"] = df["wear"] * df["torque"]                 # min*Nm
    return df

# convert machine readings into symptom certainty values
def row_to_symptoms(row):
    t = THRESHOLDS
    s = {}
    if pd.notna(row.get("dT")):
        if row["dT"] < t["low_temp_diff_K"]:
            s["low_temp_diff"] = 1.0
    if pd.notna(row.get("rpm")) and row["rpm"] < t["low_speed_rpm"]:
        s["low_speed"] = 1.0
    if pd.notna(row.get("power")):
        if row["power"] < t["low_power_W"]:
            s["low_power"] = 1.0
        if row["power"] > t["high_power_W"]:
            s["high_power"] = 1.0
    if pd.notna(row.get("strain")) and row.get("type") in t["strain_limit"]:
        if row["strain"] > t["strain_limit"][row["type"]]:
            s["high_strain"] = 1.0
    if pd.notna(row.get("wear")) and row["wear"] >= t["high_wear_min"]:
        s["high_tool_wear"] = 1.0
    return s