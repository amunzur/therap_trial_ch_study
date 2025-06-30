"""
Makes clinical characteristics table.
"""

import pandas as pd
import numpy as np
import os

path_clinical_data="/groups/wyattgrp/users/amunzur/lu_chip/resources/clinical_tables/TheraP supplementary tables - Clinical data (10-Dec-2023).tsv"
path_sample_info="/groups/wyattgrp/users/amunzur/therap_trial_ch_study/resources/sample_info.tsv"

df_clin=pd.read_csv(path_clinical_data, sep="\t", skiprows=1)
sample_info=pd.read_csv(path_sample_info, sep="\t")
pts=sample_info["Patient_id"].unique()
df_clin=df_clin[df_clin["Patient"].isin(pts)]


# Sex
n_total = kidney.shape[0]
n_male = kidney[kidney["Sex"] == "Male"].shape[0]
n_female = kidney[kidney["Sex"] == "Female"].shape[0]
perc_male = round((n_male/n_total)*100, 1)
perc_female = round((n_female/n_total)*100, 1)

# Stage at initial diagnosis
stage = kidney["Disease stage at initial diagnosis"].value_counts(dropna = False).reset_index()
n_local = stage[stage["index"] == "Localized"]["Disease stage at initial diagnosis"].iloc[0]
n_metastatic = stage[stage["index"] == "Metastatic"]["Disease stage at initial diagnosis"].iloc[0]
perc_local = round(n_local/n_total*100, 1)
perc_metastatic = round(n_metastatic/n_total*100, 1)

# Variant histology at dx
kidney["subtype"] = kidney["subtype"].replace({"Papillary type 2": "Papillary", "Papillary, not specified": "Papillary", "Papillary type 1": "Papillary"})
hist = kidney["subtype"].value_counts(dropna = False).reset_index()
n_clear_cell = hist[hist["index"] == "Clear cell"]["subtype"].iloc[0]
n_pap = hist[hist["index"] == "Papillary"]["subtype"].iloc[0]
n_Chromophobe = hist[hist["index"] == "Chromophobe"]["subtype"].iloc[0]
n_undiff = hist[hist["index"] == "Undifferenciated or unclassifiable"]["subtype"].iloc[0]
n_mixed = hist[hist["index"] == "Mixed"]["subtype"].iloc[0]
n_unknown = hist[hist["index"].isna()]["subtype"].iloc[0]

perc_clear_cell = round(n_clear_cell/n_total*100, 1)
perc_pap = round(n_pap/n_total*100, 1)
perc_Chromophobe = round(n_Chromophobe/n_total*100, 1)
perc_undiff = round(n_undiff/n_total*100, 1)
perc_mixed = round(n_mixed/n_total*100, 1)
perc_unknown = round(n_unknown/n_total*100, 1)

# Age at initial diagnosis
ages = kidney[["Patient_id", "Date of birth", "DoDLD", "DODxM"]]
ages["Date of birth"] = pd.to_datetime(ages['Date of birth'])
ages["DoDLD"] = pd.to_datetime(ages['DoDLD'])
ages["DODxM"] = pd.to_datetime(ages['DODxM'])

ages["Age at Dx"] = round((ages["DoDLD"] - ages["Date of birth"]).dt.days/365.25)
median_age_at_dx = np.median((ages["Age at Dx"]).dropna())
min_age_dx = int(ages["Age at Dx"].dropna().min())
max_age_dx = int(ages["Age at Dx"].dropna().max())
range_age_at_dx = f"{min_age_dx}-{max_age_dx}"

# Age at metastatic diagnosis
ages["DODxM"] = round((ages["DODxM"] - ages["Date of birth"]).dt.days/365.25)
median_age_at_mrcc = np.median(ages["DODxM"].dropna())
min_age_mrcc = int(ages["DODxM"].dropna().min())
max_age_mrcc = int(ages["DODxM"].dropna().max())
range_age_at_mrcc = f"{min_age_mrcc}-{max_age_mrcc}"

# 1L treatment for metastatic disease
# kidney["Type of treatment"] = kidney["Type of treatment"].replace({
#                                                                     "Ipilimumab-nivolumab": "IO combination",
#                                                                     np.nan: "Pall",
#                                                                     "NKTR214+ Nivolumab + Axitinib": "IO-TKI combination",
#                                                                     "Pazopanib": "TKI",
#                                                                     "Pembrolizumab-axitinib": "IO-TKI combination",
#                                                                     "Pembrolizumab-lenvatinib": "IO-TKI combination",
#                                                                     "Quavonlimab-Pembrolizumab-Lenvatinib": "IO-TKI combination",
#                                                                     "Sunitinib": "TKI",
#                                                                     11: "Unknown",
#                                                                     12: "Unknown"
#                                                                     })

# tx = kidney["Type of treatment"].value_counts().reset_index()
# n_io_combo = tx[tx["index"] == "IO combination"]["Type of treatment"].iloc[0]
# n_io_tki_combo = tx[tx["index"] == "IO-TKI combination"]["Type of treatment"].iloc[0]
# n_combination = tx[tx["index"] == "TKI"]["Type of treatment"].iloc[0]
# n_none_or_pall = tx[tx["index"] == "Pall"]["Type of treatment"].iloc[0]

# perc_io_combo = round(n_io_combo/n_total*100, 1)
# perc_io_tki_combo = round(n_io_tki_combo/n_total*100, 1)
# perc_combination = round(n_combination/n_total*100, 1)
# perc_none_or_pall = round(n_none_or_pall/n_total*100, 1)

# What prior treatment they had before baseline blood collection / considering both nephrectomy and systemic treatment
kidney["Date of start of 1L systemic therapy"] = pd.to_datetime(kidney['Date of start of 1L systemic therapy'])
kidney["Date of GUBB draw"] = pd.to_datetime(kidney['Date of GUBB draw'])
kidney["Draw minus 1L treatment"] = (kidney["Date of GUBB draw"] - kidney["Date of start of 1L systemic therapy"]).dt.days
kidney["Therapy prior to draw"] = kidney["Draw minus 1L treatment"] >= 7 # We consider it exposed if the blood was drawn at least 1 days after initiation of treatment
kidney.loc[kidney["Did the patient receive systemic treatment?"] == False, "Therapy prior to draw"] = np.nan

treatment_counts = kidney[kidney["Therapy prior to draw"] == True]["Type of treatment"].replace({
    "Ipilimumab-nivolumab": "ICI doublet", 
    "Sunitinib": "VEGFR monotherapy", 
    "Pazopanib": "VEGFR monotherapy", 
    "Pembrolizumab-axitinib": "Combination ICI-VEGFR", 
    "Belzutifan + Lenvatinib + Pembrolizumab": "Other"}).value_counts().reset_index()

n_ici_doublet = treatment_counts[treatment_counts["index"] == "ICI doublet"]["Type of treatment"].iloc[0]
n_vegfr_mono = treatment_counts[treatment_counts["index"] == "VEGFR monotherapy"]["Type of treatment"].iloc[0]
n_ici_vegfr = treatment_counts[treatment_counts["index"] == "Combination ICI-VEGFR"]["Type of treatment"].iloc[0]
n_other = treatment_counts[treatment_counts["index"] == "Other"]["Type of treatment"].iloc[0]
n_none_or_pall = kidney[kidney["Therapy prior to draw"] !=  True].shape[0]
n_nephrectomy = kidney[kidney["nephrectomy"] == True].shape[0]

perc_ici_doublet = round(n_ici_doublet/n_total*100, 2)
perc_vegfr_mono = round(n_vegfr_mono/n_total*100, 2)
perc_ici_vegfr = round(n_ici_vegfr/n_total*100, 2)
perc_other = round(n_other/n_total*100, 2)
perc_none_or_pall = round(n_none_or_pall/n_total*100, 2)
perc_nephrectomy = round(n_nephrectomy/n_total*100, 2)

# Mets information
mets = kidney[["Patient_id", "Bone_Met", "LN_Met", "Lung_Met", "Liver_Met"]]
n_bone_mets = mets[mets["Bone_Met"] > 0].shape[0]
n_LN_mets = mets[mets["LN_Met"] > 0].shape[0]
n_lung_mets = mets[mets["Lung_Met"] > 0].shape[0]
n_liver_mets = mets[mets["Liver_Met"] > 0].shape[0]

perc_bone_mets = round(n_bone_mets/n_total*100, 1)
perc_LN_mets = round(n_LN_mets/n_total*100, 1)
perc_lung_mets = round(n_lung_mets/n_total*100, 1)
perc_liver_mets = round(n_liver_mets/n_total*100, 1)

# irAEs
ae = kidney["irAE"].value_counts().reset_index()
n_ae = ae[ae["index"] == 0]["irAE"].iloc[0]
n_no_ae = ae[ae["index"] == 1]["irAE"].iloc[0]
perc_ae = round(n_ae/(n_ae+n_no_ae)*100, 1)
perc_no_ae = round(n_no_ae/(n_ae+n_no_ae)*100, 1)

# Information about follow up samples
path_sample_information = "/groups/wyattgrp/users/amunzur/pipeline/resources/sample_lists/sample_information.tsv"
sample_info = pd.read_csv(path_sample_information, sep = "\t", names = ["Patient_id", "Date_collected", "Diagnosis", "Timepoint"])
sample_info = sample_info[(sample_info["Diagnosis"] == "Kidney") & (sample_info["Timepoint"] == "During treatment")]
OT_sample_counts = sample_info["Patient_id"].value_counts().reset_index(drop = True).value_counts().reset_index().rename(columns = {"index": "Number of OT samples", "Patient_id": "Number of patients"})

n_1_OT_sample = OT_sample_counts[OT_sample_counts["Number of OT samples"] == 1]["Number of patients"].iloc[0]
n_2_OT_sample = OT_sample_counts[OT_sample_counts["Number of OT samples"] == 2]["Number of patients"].iloc[0]

################################################################################

# Make the table
lines = [
    f"Clinical characteristics of {n_total} patients with mRCC\n", 
    f"Biological sex, n (%)\n", 
    f"Male\t{n_male} ({perc_male})\n",
    f"Female\t{n_female} ({perc_female})\n",
    "Stage at initial diagnosis, n (%)\n",
    f"Localized RCC\t{n_local} ({perc_local})\n",
    f"Metastatic RCC\t{n_metastatic} ({perc_metastatic})\n",
    f"Histology of the pathology specimen, n (%)\n",
    f"Clear cell\t{n_clear_cell} ({perc_clear_cell})\n",
    f"Papillary\t{n_pap} ({perc_pap})\n",
    f"Chromophobe\t{n_Chromophobe} ({perc_Chromophobe})\n",
    f"Undifferentiated\t{n_undiff} ({perc_undiff})\n",
    f"Mixed\t{n_mixed} ({perc_mixed})\n",
    f"Unknown\t{n_unknown} ({perc_unknown})\n",
    f"Age at initial diagnosis, median (range)\t{median_age_at_dx} ({range_age_at_dx})\n",
    f"Age at metastatic diagnosis, median (range)\t{median_age_at_mrcc} ({range_age_at_mrcc})\n",
    f"Treatment history prior to baseline blood collection\n"
    f"Nephrectomy\t{n_nephrectomy} ({perc_nephrectomy})\n",
    f"ICI doublet\t{n_ici_doublet} ({perc_ici_doublet})\n",
    f"VEGFR monotherapy\t{n_vegfr_mono} ({perc_vegfr_mono})\n",
    f"Combination ICI-VEGFR\t{n_ici_vegfr} ({perc_ici_vegfr})\n",
    f"Other\t{n_other} ({perc_other})\n",
    f"No systemic treatment or palliative\t{n_none_or_pall} ({perc_none_or_pall})\n",
    "Metastases n (%)\n",
    f"Bone\t{n_bone_mets} ({perc_bone_mets})\n",
    f"Lymph node\t{n_LN_mets} ({perc_LN_mets})\n",
    f"Lung\t{n_lung_mets} ({perc_lung_mets})\n",
    f"Liver\t{n_liver_mets} ({perc_liver_mets})\n",
    f"Unknown\t{n_unknown} ({perc_unknown})\n",
    "irAEs n (%)\n",
    f"Present\t{n_ae} ({perc_ae})\n",
    f"Absent\t{n_no_ae} ({perc_no_ae})\n"]

# "1L treatment for metastatic disease, n (%)\n",
# f"IO combination\t{n_io_combo} ({perc_io_combo})\n", 
# f"IO-TKI combination\t{n_io_tki_combo} ({perc_io_tki_combo})\n", 
# f"TKI\t{n_combination} ({perc_combination})\n", 
# f"None/palliative\t{n_none_or_pall} ({perc_none_or_pall})\n",

os.remove('/groups/wyattgrp/users/amunzur/pipeline/results/figures/pub_figures/kidney_clinical_table.tsv')
# Open a file in write mode
with open('/groups/wyattgrp/users/amunzur/pipeline/results/figures/pub_figures/kidney_clinical_table.tsv', 'w') as file:
    # Write each line to the file
    for line in lines:
        file.write(line)

