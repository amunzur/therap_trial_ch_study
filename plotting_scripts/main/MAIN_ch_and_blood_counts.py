import pandas as pd
import numpy as np
import os
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu

project_dir = os.environ.get("project_dir")

baseline_ch_path=f"{project_dir}/CH_baseline.csv"
path_sample_info = f"{project_dir}/resources/sample_info.tsv"
figures = f"{project_dir}/figures/main"
path_clin_data=f"{project_dir}/clinical_data/TheraP supplementary tables - Clinical data (10-Dec-2023).tsv"

# Subset both dfs to patients with progression samples
sample_info = pd.read_csv(path_sample_info, sep= "\t")
arm_info=sample_info[sample_info["Timepoint"]=="Baseline"][["Patient_id", "Arm"]].drop_duplicates()
baseline_ch = pd.read_csv(path_baseline_mutations)
clin_df=pd.read_csv(path_clin_data, sep="\t", skiprows=1)[["Patient", "Neutrophil lymphocyte ratio"]].rename(columns={"Patient": "Patient_id"})

def return_blood_counts(pts, variable_name):
    """
    Given patient ids and a variable of interest, returns the values.
    """
    path_clin_data="/groups/wyattgrp/users/amunzur/lu_chip/resources/clinical_tables/TheraP supplementary tables - Clinical data (10-Dec-2023).tsv"
    clin_df=pd.read_csv(path_clin_data, sep="\t", skiprows=1).rename(columns={"Patient": "Patient_id"})
    clin_df_subset=clin_df[clin_df["Patient_id"].isin(pts)]
    
    values=clin_df_subset[variable_name].dropna().tolist()
    
    return(values)

def return_pts_with_ch(baseline_ch, min_vaf, max_vaf, arm=None):
    """
    Given a CH threshold returns the list of patients carrying CH with VAF >= to that.
    """
    if arm is None:
        baseline_ch_arm=baseline_ch.copy()
    else:
        baseline_ch_arm=baseline_ch[baseline_ch["Arm"]==arm]
    
    baseline_ch_max=baseline_ch_arm.groupby("Patient_id")["VAF_n"].max().reset_index()
    pts=baseline_ch_max[(baseline_ch_max["VAF_n"]>=min_vaf) & (baseline_ch_max["VAF_n"]<max_vaf)]["Patient_id"]
    return(pts)

def plot_boxes_blood_counts(bc_dict, ax):
    """
    Plots boxes from values in the dict.
    """
    flierprops_dict = dict(marker='o', markersize=5, markeredgecolor='black', linestyle='None')
    whiskerprops_dict =dict(color='black')
    medianprops_dict = dict(color='black')
    capprops_dict = dict(color='black')
    boxprops_dict = dict(facecolor="whitesmoke", edgecolor='black', linewidth = 0.7)  
    
    for i, (key, value) in enumerate(bc_dict.items()):
        boxplot = ax.boxplot(value, positions = [i], flierprops = flierprops_dict, boxprops = boxprops_dict, medianprops = medianprops_dict, capprops = capprops_dict, widths = 0.3, showfliers = False, patch_artist = True)
        ax.scatter(np.random.uniform(i-0.08, i+0.08, len(value)), value, s = 1, color = "black", alpha = 1, zorder = 100)
        
        # Calculate median and annotate above the median line
        median_value = round(np.median(value), 2)
        ax.text(i+0.35, median_value, f"{median_value:.2f}", ha='center', va='center', fontsize=7, color='red')
    
    return(ax)

def run_mwu(bc_dict):
    ch_neg=bc_dict["CH neg"]
    g1=bc_dict[0]
    g2=bc_dict[2]
    g3=bc_dict[10]
    
    mwu_chneg_vs_g1=mannwhitneyu(ch_neg, g1).pvalue
    mwu_chneg_vs_g2=mannwhitneyu(ch_neg, g2).pvalue
    mwu_chneg_vs_g3=mannwhitneyu(ch_neg, g3).pvalue
    
    mwu_dict={
        "CH neg vs 0.25-2": mwu_chneg_vs_g1, 
        "CH neg vs 2-10": mwu_chneg_vs_g2,
        "CH neg vs >10": mwu_chneg_vs_g3
        }
    
    return(mwu_dict)

def annotate_mwu_p(mwu_dict, ax):
    ymax=ax.get_ylim()[1]
    for i, (key, value) in enumerate(mwu_dict.items()):
        rounded_p=round(value, 2)
        if rounded_p<0.05:
            c="red"
        else:
            c="black"
        ax.text(i+1, ymax, f"p={rounded_p}", ha='center', va='top', fontsize=9, color=c)
    
    ax.text(0, ymax, "REF", ha='center', va='top', fontsize=9, color="black")
    
    return(ax)

def run_mwu_binary_ch(bc_dict):
    ch_neg=bc_dict["CH neg"]
    ch_pos=bc_dict["CH pos"]
    
    mwu_chneg_vs_g1=mannwhitneyu(ch_neg, ch_pos).pvalue
    
    mwu_dict={
        "CH neg vs CH pos": mwu_chneg_vs_g1, 
        }
    
    return(mwu_dict)
    


def main_boxp_plotting_function(blood_count, vaf_groups=None, is_binary=False, suffix="", arms=None, figures=figures):
    fig = plt.figure(figsize=(8, 4) if arms else (4, 3))
    arms = arms or []
    
    # Ensure there is at least one arm, so the len(arms) doesn't raise an error
    num_arms = len(arms) if arms else 1
    gs = gridspec.GridSpec(1, num_arms, width_ratios=[1]*num_arms, wspace=0.2)
    
    for i, arm in enumerate(arms or [None]):
        ax = plt.subplot(gs[i])
        bc_dict = {}
        
        ch_status=annotate_mutation_status_lu(baseline_ch, path_sample_info, "CHIP", "Baseline", annotate_gene=False).merge(arm_info)
        if arms:
            ch_neg_pts=ch_status[(ch_status["CHIP status"]=="Negative")&(ch_status["Arm"]==arm)]["Patient_id"]
        else:
            ch_neg_pts=ch_status[ch_status["CHIP status"]=="Negative"]["Patient_id"]
        
        ch_neg_values=return_blood_counts(ch_neg_pts, blood_count)
        bc_dict["CH neg"]=ch_neg_values
        
        if is_binary:
            if arms:
                ch_pos_pts=ch_status[(ch_status["CHIP status"]=="Positive")&(ch_status["Arm"]==arm)]["Patient_id"]
                bc_dict["CH pos"] = return_blood_counts(ch_pos_pts, blood_count)
            else:
                ch_pos_pts=ch_status[ch_status["CHIP status"]=="Positive"]["Patient_id"]
                bc_dict["CH pos"] = return_blood_counts(ch_pos_pts, blood_count)
        else:
            for min_vaf, max_vaf in vaf_groups:
                pts = return_pts_with_ch(baseline_ch, min_vaf=min_vaf, max_vaf=max_vaf, arm=arm)
                bc_dict[min_vaf] = return_blood_counts(pts, blood_count)
        
        ax = plot_boxes_blood_counts(bc_dict, ax)
        mwu_dict = run_mwu(bc_dict) if not is_binary else run_mwu_binary_ch(bc_dict)
        ax.set_xticks(range(len(bc_dict)))
        ax.set_xticklabels([str(k) for k in bc_dict.keys()])
        ax.set_xlabel("CH VAF% group" if not is_binary else "CH status")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(arm or f"Baseline {blood_count}")
        
        ax = annotate_mwu_p(mwu_dict, ax)
        
        if not is_binary:
            ax.set_xticklabels(["CH-", "0.25-2", "2-10", "≥10"])
        
    if arms:
        fig.suptitle(blood_count)
    gs.tight_layout(fig)
    fig.savefig(f"{figures}/{suffix}_{blood_count}.png")
    fig.savefig(f"{figures}/{suffix}_{blood_count}.pdf")

for blood_count in ["Haemoglobin", "Neutrophil lymphocyte ratio"]:
    main_boxp_plotting_function(blood_count, vaf_groups=[(0, 2), (2, 10), (10, 9999)], suffix="MAIN")
