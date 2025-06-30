import numpy as np
import pandas as pd
import os
from scipy.stats import fisher_exact
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def make_barchart_for_aes(ae_name, results_df, ax, lighter_shade, darker_shade):
    """
    Shows prevalence of AEs and CH status for any CH, CH>2%, CH>10%.
    Runs fisher's exact.
    """
    ae_subset_df=results_df[results_df["Adverse_event"]==ae_name]
    offset=0.2
    annotation_dict={"CHIP status": "CH≥0.25%", "CH_2": "CH≥2%", "CH_10": "CH≥10%"}
    
    xtick_list=[]
    for xpos, ch_group in enumerate(["CHIP status", "CH_2", "CH_10"]):
        a=ae_subset_df[ae_subset_df["CH_Group"]==ch_group]["a (AE+, CH+)"].values[0]
        b=ae_subset_df[ae_subset_df["CH_Group"]==ch_group]["b (AE+, CH-)"].values[0]
        c=ae_subset_df[ae_subset_df["CH_Group"]==ch_group]["c (AE-, CH+)"].values[0]
        d=ae_subset_df[ae_subset_df["CH_Group"]==ch_group]["d (AE-, CH-)"].values[0]
        p=round(ae_subset_df[ae_subset_df["CH_Group"]==ch_group]["P_value"].values[0], 2)
        
        ax.bar(xpos-offset, a, color=darker_shade, width=0.35, edgecolor="none")
        ax.bar(xpos-offset, b, bottom=a, color=lighter_shade, width=0.35, edgecolor="none")
        ax.bar(xpos+offset, c, color=darker_shade, width=0.35, edgecolor="none")
        ax.bar(xpos+offset, d, bottom=c, color=lighter_shade, width=0.35, edgecolor="none")
        
        ax.text(xpos-offset, a-1, str(a), color="white", va="top", ha="center", fontsize=8)
        ax.text(xpos-offset, a+b-1, str(b), color="black", va="top", ha="center", fontsize=8)
        ax.text(xpos+offset, c-1, str(c), color="white", va="top", ha="center", fontsize=8)
        ax.text(xpos+offset, c+d-1, str(d), color="black", va="top", ha="center", fontsize=8)
        
        annot=annotation_dict[ch_group]
        ax.text(xpos, 95, f"{annot}\np={p}", va="bottom", ha="center", color="black", fontsize=8)
        
        xtick_list.append(xpos-offset)
        xtick_list.append(xpos+offset)
        
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_title(ae_name, y=1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(xtick_list)
    ax.set_xticklabels(np.tile(["present", "absent"], 3), rotation=90)
    
    return(ax)

project_dir = os.environ.get("project_dir")
path_utilities=f"{project_dir}/plotting_scripts/utilities.py"
with open(path_utilities, 'r') as file:
    script_code = file.read()
exec(script_code)

path_ae=f"{project_dir}/clinical_data/TheraP Adverse Events - Adverse events (safety cohort; n=183).csv"
path_ch_baseline=f"{project_dir}/CH_baseline.csv"
path_ch_prog=f"{project_dir}/CH_progression.csv"
path_sample_info = f"{project_dir}/resources/sample_info.tsv"
dir_figures=f"{project_dir}/figures/supp"

arm_color_dict={"LuPSMA": "#6f1fff", "Cabazitaxel": "#3e3939"}
arm_color_dict_lighter={"LuPSMA": "#d3bbfe", "Cabazitaxel": "#b2b0b0"}

sample_info=pd.read_csv(path_sample_info, sep="\t")

ae_df=pd.read_csv(path_ae)
ae_df.columns=["Patient_id", "Arm", "System_Organ_Class", "Adverse_event", "Highest_grade_event"]
ae_df["Patient_id"] = "TheraP-" + ae_df["Patient_id"].astype(str).str.zfill(3)
ae_df=ae_df[ae_df["Patient_id"].isin(sample_info["Patient_id"].unique())]

baseline_ch=pd.read_csv(path_ch_baseline)
prog_ch=pd.read_csv(path_ch_prog)

# Annotate CH status
baseline_ch=harmonize_vaf_columns(baseline_ch, timepoint="Baseline")
del baseline_ch["VAF_n"]
baseline_ch=baseline_ch.rename(columns={"VAF%": "VAF_n"})
mut_status = annotate_mutation_status_lu(baseline_ch, PATH_sample_information, annotate_what="CHIP", timepoint="Baseline", annotate_gene=False)
mut_status = mut_status.merge(sample_info[['Patient_id', 'Arm']].drop_duplicates())
mut_status["CHIP status"]=mut_status["CHIP status"].replace("Positive", True).replace("Negative", False)

max_ch_df = baseline_ch.groupby("Patient_id")["VAF_n"].max().reset_index()

muts_group2 = max_ch_df[max_ch_df["VAF_n"] >= 2].merge(sample_info[['Patient_id', 'Arm']].drop_duplicates()).assign(CH_2=True)
muts_group3 = max_ch_df[max_ch_df["VAF_n"] >= 10].merge(sample_info[['Patient_id', 'Arm']].drop_duplicates()).assign(CH_10=True)
dta_pts=baseline_ch[baseline_ch["Gene"].isin(["DNMT3A", "TET2", "ASXL1"])][["Patient_id", "Arm"]].drop_duplicates().assign(DTA_CH=True)
ddr_pts=baseline_ch[baseline_ch["Gene"].isin(["TP53", "PPM1D", "BRCA1", "BRCA2", "ATM", "CHEK2"])][["Patient_id", "Arm"]].drop_duplicates().assign(DDR_CH=True)

mut_status_baseline=mut_status.merge(muts_group2, how="outer").merge(muts_group3, how="outer").merge(dta_pts, how="outer").merge(ddr_pts, how="outer").fillna(False).drop("VAF_n", axis=1)

####### Intersect with AEs
ae_focus=["Thromboembolic event", "Anemia", "Neutropenia (+/- fever)", "Platelet count decreased", "White blood cell decreased"]

# Generates the contingency tables for running Fisher's exact test later on.
results = []

for ae_term in ae_focus:
    ae_df_subset=ae_df[ae_df["Adverse_event"]==ae_term].merge(mut_status_baseline)
    ae_df_subset=ae_df_subset.merge(mut_status_baseline, how="outer")
    
    ae_df_subset["System_Organ_Class"]=ae_df_subset["System_Organ_Class"].fillna(False)
    ae_df_subset["Adverse_event"]=ae_df_subset["Adverse_event"].fillna(False)
    
    for ch_group in ["CHIP status", "CH_2", "CH_10"]:
        # Only consider LuPSMA arm
        ch_subgroup = ae_df_subset[ae_df_subset["Arm"] == "LuPSMA"]
        
        # Create 2x2 contingency table for: AE presence vs CH status (True/False)
        # Counts:
        #   - AE present + CH present
        #   - AE present + CH absent
        #   - AE absent + CH present
        #   - AE absent + CH absent
        try:
            a = sum((ch_subgroup["Adverse_event"] == ae_term) & (ch_subgroup[ch_group] == True))
            b = sum((ch_subgroup["Adverse_event"] == ae_term) & (ch_subgroup[ch_group] == False))
            c = sum((ch_subgroup["Adverse_event"] != ae_term) & (ch_subgroup[ch_group] == True))
            d = sum((ch_subgroup["Adverse_event"] != ae_term) & (ch_subgroup[ch_group] == False))
            
            table = [[a, b], [c, d]]
            
            odds_ratio, p_value = fisher_exact(table)
            
            results.append({
                "Adverse_event": ae_term,
                "CH_Group": ch_group,
                "a (AE+, CH+)": a,
                "b (AE+, CH-)": b,
                "c (AE-, CH+)": c,
                "d (AE-, CH-)": d,
                "Odds_ratio": odds_ratio,
                "P_value": p_value
            })
        except Exception as e:
            print(f"Error in AE: {ae_term}, CH group: {ch_group} -> {e}")

results_df=pd.DataFrame(results)

# Plotting
fig = plt.figure(figsize=(7, 8))
gs_outer=gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.3)

for i, ae_name in enumerate(ae_focus):
    ae_ax=plt.subplot(gs_outer[i])
    ae_ax=make_barchart_for_aes(ae_name, results_df, ae_ax, lighter_shade=arm_color_dict_lighter["LuPSMA"], darker_shade=arm_color_dict["LuPSMA"])
    
    if i<3:
        ae_ax.set_xticklabels([])

fig.savefig("{dir_figures}/lupsma_aes_and_ch.png")
fig.savefig("{dir_figures}/lupsma_aes_and_ch.pdf")