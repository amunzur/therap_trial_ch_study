import pandas as pd
import numpy as np
import os
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
import statsmodels.api as sm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu

mpl.rcParams['font.size'] = 10
mpl.rcParams['text.color'] = 'k'
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['legend.handletextpad'] = '0.8'
mpl.rcParams['legend.labelspacing'] = '0.4'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['axes.labelsize'] = 10

"""
Makes clinical characteristics supplementary table 
"""

# LOAD CHIP DATASETS
project_dir = os.environ.get("project_dir")
path_sample_information = f"{project_dir}/resources/sample_info.tsv"
path_clin=f"{project_dir}/clinical_data/TheraP supplementary tables - Clinical data (10-Dec-2023).tsv"
dir_figures=f"{project_dir}/figures"

sample_info=pd.read_csv(path_sample_information, sep="\t")
study_cohort=sample_info[sample_info["Timepoint"]=="Baseline"]["Patient_id"].unique()

clin_data=pd.read_csv(path_clin, skiprows=1, sep="\t")
clin_data=clin_data.rename(columns={"Patient": "Patient_id"})
clin_data=clin_data[clin_data["Patient_id"].isin(study_cohort)]

def return_iqr(values):
    q75 = round(np.percentile(values, 75), 2)
    q25 = round(np.percentile(values, 25), 2)
    iqr=f"{q25}-{q75}"
    return(iqr)

# Store table rows
table_rows = []
n_total_dict={"LuPSMA": 96, "Cabazitaxel": 82}

# Loop through arms
for arm in ['LuPSMA', 'Cabazitaxel']:
    clin_df_arm = clin_data[clin_data["Arm received"] == arm]
    n_total_arm = n_total_dict[arm]
    
    # Continuous variables
    for variable in ['Age', 'PSA', 'LDH', 'ALP', 'Haemoglobin', 'Neutrophil lymphocyte ratio']:
        median = round(clin_df_arm[variable].median(), 2)
        iqr = return_iqr(clin_df_arm[variable].dropna())
        row = next((r for r in table_rows if r["Characteristic"] == variable), None)
        if row:
            row[arm] = f"{median} ({iqr})"
        else:
            table_rows.append({"Characteristic": variable, arm: f"{median} ({iqr})"})
    
    # Categorical: Prior ARPI
    prior_arpi_categories = [
        "Abiraterone only",
        "Enzalutamide only",
        "Both Enzalutamide and Abiraterone",
        "Neither Enzalutamide nor Abiraterone"
    ]
    for category in prior_arpi_categories:
        label = f"Prior ARPI: {category}"
        n = clin_df_arm["Prior ARPI"].value_counts().get(category, 0)
        perc = round(n / n_total_arm * 100)
        row = next((r for r in table_rows if r["Characteristic"] == label), None)
        if row:
            row[arm] = f"{n} ({perc}%)"
        else:
            table_rows.append({"Characteristic": label, arm: f"{n} ({perc}%)"})
    
    # Gleason
    gleason_categories = {
        "≤7": "Gleason: ≤7",
        "≥8": "Gleason: ≥8",
        "Missing": "Gleason: Missing"
    }
    for value, label in gleason_categories.items():
        if value == "Missing":
            n = clin_df_arm["Gleason sum category"].isna().sum()
        else:
            n = clin_df_arm["Gleason sum category"].value_counts().get(value, 0)
        perc = round(n / n_total_arm * 100)
        row = next((r for r in table_rows if r["Characteristic"] == label), None)
        if row:
            row[arm] = f"{n} ({perc}%)"
        else:
            table_rows.append({"Characteristic": label, arm: f"{n} ({perc}%)"})
    
    # PSA50 response
    label = "PSA50 response"
    n = clin_df_arm["PSA50 response"].value_counts().get("Yes", 0)
    perc = round(n / n_total_arm * 100)
    row = next((r for r in table_rows if r["Characteristic"] == label), None)
    if row:
        row[arm] = f"{n} ({perc}%)"
    else:
        table_rows.append({"Characteristic": label, arm: f"{n} ({perc}%)"})
    
    # PSMA disease burden
    disease_burden_categories = {
        "Greater than 20": "PSMA disease burden: >20 lesions",
        "20 or less": "PSMA disease burden: ≤20 lesions"
    }
    for value, label in disease_burden_categories.items():
        n = clin_df_arm["PSMA disease burden stratification"].value_counts().get(value, 0)
        perc = round(n / n_total_arm * 100)
        row = next((r for r in table_rows if r["Characteristic"] == label), None)
        if row:
            row[arm] = f"{n} ({perc}%)"
        else:
            table_rows.append({"Characteristic": label, arm: f"{n} ({perc}%)"})
    
    # Disease metastases
    mets_categories=["Nodal metastases", 'Bone metastases', "Liver metastases", "Lung metastases", "Any visceral metastases"]
    for cat in mets_categories:
        n=clin_df_arm[cat].value_counts().get("Yes", 0)
        perc = round(n / n_total_arm * 100)
        row = next((r for r in table_rows if r["Characteristic"] == cat), None)
        if row:
            row[arm] = f"{n} ({perc}%)"
        else:
            table_rows.append({"Characteristic": cat, arm: f"{n} ({perc}%)"})

# Convert to DataFrame
table_df = pd.DataFrame(table_rows)
table_df = table_df[["Characteristic", "LuPSMA", "Cabazitaxel"]].fillna("–")

table_df.to_csv(f"{dir_figures}/supp/baseline_clin_characteristics.csv", index=False)


















# Variables of interest to include in table
variable_names=[]
n_total_dict={"LuPSMA": 96, "Cabazitaxel": 82}

for arm in ['LuPSMA', 'Cabazitaxel']:
    
    clin_df_arm=clin_data[clin_data["Arm received"]==arm]
    n_total_arm=n_total_dict[arm]
    
    # Numerical variables we can easily get a median and IQR for
    for variable in ['Age', 'PSA', 'LDH', 'ALP', 'Haemoglobin', 'Neutrophil lymphocyte ratio']:
        variable_median=clin_df_arm[variable].dropna()
        variable_iqr=return_iqr(clin_df_arm[variable].dropna())
    
    # Previous treatment, we print the n and percentage
    N_abi_only=clin_df_arm["Prior ARPI"].value_counts()["Abiraterone only"]
    N_enza_only=clin_df_arm["Prior ARPI"].value_counts()["Enzalutamide only"]
    N_both_abi_enza=clin_df_arm["Prior ARPI"].value_counts()["Both Enzalutamide and Abiraterone"]
    N_neither_abi_nor_enza=clin_df_arm["Prior ARPI"].value_counts()["Neither Enzalutamide nor Abiraterone"]
    
    PERC_abi_only=round(N_abi_only/n_total_arm*100)
    PERC_enza_only=round(N_enza_only/n_total_arm*100)
    PERC_both_abi_enza=round(N_both_abi_enza/n_total_arm*100)
    PERC_neither_abi_nor_enza=round(N_neither_abi_nor_enza/n_total_arm*100)
    
    # Gleason score
    N_gleason_low=clin_df_arm[clin_df_arm["Gleason sum category"]=="≤7"].shape[0]
    N_gleason_high=clin_df_arm[clin_df_arm["Gleason sum category"]=="≥8"].shape[0]
    N_gleason_missing=clin_df_arm[clin_df_arm["Gleason sum category"].isna()].shape[0]
    
    PERC_gleason_low=round(N_gleason_low/n_total_arm*100)
    PERC_gleason_high=round(N_gleason_high/n_total_arm*100)
    PERC_gleason_missing=round(N_gleason_missing/n_total_arm*100)
    
    # PSA50 response
    N_PSA50=clin_df_arm[clin_df_arm["PSA50 response"]=="Yes"].shape[0]
    PERC_PSA50=round(N_PSA50/n_total_arm*100)
    
    # Disease burden
    N_disease_burden_high=clin_df_arm["PSMA disease burden stratification"].value_counts()["Greater than 20"]
    N_disease_burden_lowh=clin_df_arm["PSMA disease burden stratification"].value_counts()["20 or less"]
    
    PERC_disease_burden_high=round(N_disease_burden_high/n_total_arm*100)
    PERC_disease_burden_lowh=round(N_disease_burden_lowh/n_total_arm*100)
    
    


    
    
    
    