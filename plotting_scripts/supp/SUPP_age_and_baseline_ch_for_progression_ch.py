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
Does age or baseline CH status impact treatment emergent CH? 
"""

# LOAD CHIP DATASETS
project_dir = os.environ.get("project_dir")

dir_figures=f"{project_dir}/figures"
baseline_ch_path=f"{project_dir}/CH_baseline.csv"
prog_ch_path=f"{project_dir}/CH_progression.csv"
path_sample_information = f"{project_dir}/resources/sample_info.tsv"
path_clin=f"{project_dir}/clinical_data/TheraP supplementary tables - Clinical data (10-Dec-2023).tsv"

sample_info=pd.read_csv(path_sample_information, sep="\t")
color_dict={"LuPSMA": "#8d75bd", "Cabazitaxel": "#7faf9d"}
boxp_color_dict={"LuPSMA": "#e3ddee", "Cabazitaxel": "#e0ebe7"}

utilities=f"{project_dir}/plotting_scripts/utilities.py"

for path in [utilities]:
    with open(path, 'r') as file:
        script_code = file.read()

exec(script_code)

pts_with_prog_samples=sample_info[sample_info["Timepoint"]=="FirstProgression"]["Patient_id"]

baseline_ch=pd.read_csv(baseline_ch_path)
progression_ch=pd.read_csv(prog_ch_path)
progression_ch=progression_ch[progression_ch["Independently detected at baseline"]==False]
baseline_ch_subset=baseline_ch[baseline_ch["Patient_id"].isin(pts_with_prog_samples)]

def make_baseline_ch_progression_plot(baseline_ch_subset, progression_ch, path_sample_information, ax_lu, ax_caba):
    """
    Barchart comparing baseline CH prevalence in patients with and without progression CH.
    """
    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.1], wspace=0.1)
    
    ax_lu= fig.add_subplot(gs[0])
    ax_caba = fig.add_subplot(gs[1], sharey=ax_lu)
    ax_legend=fig.add_subplot(gs[2], sharey=ax_lu)
    
    sample_info=pd.read_csv(path_sample_information, sep="\t")
    all_baseline_pts=sample_info[sample_info["Timepoint"]=="Baseline"]["Patient_id"].unique()
    
    arm_color_dict={"LuPSMA": "#6f1fff", "Cabazitaxel": "#3e3939"}
    arm_color_dict_lighter={"LuPSMA": "#d3bbfe", "Cabazitaxel": "#b2b0b0"}
    
    ax_dict={"LuPSMA": ax_lu, "Cabazitaxel": ax_caba}
    for arm in ["LuPSMA", "Cabazitaxel"]:
        all_progression_pts=sample_info[(sample_info["Timepoint"]=="FirstProgression") & (sample_info["Arm"]==arm)]["Patient_id"].unique()
        pts_prog_muts=progression_ch[progression_ch["Arm"]==arm]["Patient_id"].unique()
        pts_base_muts=baseline_ch_subset[baseline_ch_subset["Arm"]==arm]["Patient_id"].unique()
        
        a=len(list(set(pts_prog_muts) & set(pts_base_muts))) # Has baseline CH and progression CH
        b=len([x for x in pts_base_muts if x not in pts_prog_muts]) # Has baseline CH but no progression CH
        c=len([x for x in pts_prog_muts if x not in pts_base_muts]) # Has progression CH but no baseline CH
        d=len([x for x in all_progression_pts if x not in pts_prog_muts and x not in pts_base_muts]) # Has neither
        
        ax=ax_dict[arm]
        contingency = [[a, b], [c, d]]
        oddsratio, p_value = fisher_exact(contingency)
        p_value_rounded=round(p_value, 1)
        ax.text(0.5, 39, f"p={p_value_rounded}", ha="center", va="top")
        
        # Plot progression CH+ first
        ax.bar(0, a, color=arm_color_dict[arm], width=0.8)
        ax.bar(0, c, bottom=a, color=arm_color_dict_lighter[arm], width=0.8)
        
        # And now CH- at progression
        ax.bar(1, b, color=arm_color_dict[arm], width=0.8)
        ax.bar(1, d, bottom=b, color=arm_color_dict_lighter[arm], width=0.8)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Present", "Absent"], rotation=90)
        ax.set_xlabel("Treatment-emergent CH status")
        ax.set_title(arm)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("Number of patients with progression samples")
        ax.set_xlim((-0.5, 1.5))
        
        if arm=="Cabazitaxel":
            ax.set_ylabel("")
    
    # Add legend
    # LuPSMA legend block #############################
    lupsma_handles = [Line2D([0], [0], marker='s', color=c, label=l, markersize=5, linestyle='')
                      for c, l in zip([arm_color_dict["LuPSMA"], arm_color_dict_lighter["LuPSMA"]],
                                      ["CH+", "CH-"])]
    lupsma_legend = ax_legend.legend(handles=lupsma_handles, loc="upper left", bbox_to_anchor=(-0.5, 1), frameon=False, fontsize=6, handlelength=2, handletextpad=0.4, borderpad=0.3, title="LuPSMA", title_fontsize=7)
    ax_legend.add_artist(lupsma_legend)
    
    # Cabazitaxel legend block #############################
    caba_handles = [Line2D([0], [0], marker='s', color=c, label=l, markersize=5, linestyle='')
                    for c, l in zip([arm_color_dict["Cabazitaxel"], arm_color_dict_lighter["Cabazitaxel"]],
                                    ["CH+", "CH-"])]
    caba_legend = ax_legend.legend(handles=caba_handles, loc="upper left", bbox_to_anchor=(-0.5, 0.85), frameon=False, fontsize=6, handlelength=2, handletextpad=0.4, borderpad=0.3, title="Cabazitaxel", title_fontsize=7)
    ax_legend.add_artist(caba_legend)
    
    ax_legend.axis("off")
        
    gs.tight_layout(fig)
    fig.savefig(f"{dir_figures}/supp/base_prog_ch_fishers.png", bbox_inches="tight")
    fig.savefig(f"{dir_figures}/supp/base_prog_ch_fishers.pdf", transparent=True, bbox_inches="tight")

def compare_ages(progression_ch, path_clin, path_sample_information):
    """
    Compares the age at randomization in patients with and without progression CH
    """
    clin_df=pd.read_csv(path_clin, sep="\t", skiprows=1)[["Patient", "Arm received", "Age"]]
    clin_df.columns=["Patient_id", "Arm", "Age"]
    
    sample_info=pd.read_csv(path_sample_information, sep="\t")
    sample_info_prog=sample_info[sample_info["Timepoint"]=="FirstProgression"]
    
    arm_color_dict={"LuPSMA": "#6f1fff", "Cabazitaxel": "#3e3939"}
    
    progression_ch["CH status"]="Positive"
    merged=progression_ch[["Patient_id", "CH status"]].drop_duplicates().reset_index(drop=True).merge(sample_info_prog, how="outer")
    merged["CH status"]=merged["CH status"].fillna("Negative")
    merged=merged.merge(clin_df)
    
    # Plotting
    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.1], wspace=0.1)
    
    for i, arm in enumerate(["LuPSMA", "Cabazitaxel"]):
        ax=fig.add_subplot(gs[i])
        df=merged[merged["Arm"]==arm]
        
        age_data=[]
        for j, status in enumerate(["Positive", "Negative"]):
            df_subset=df[df["CH status"]==status]
            age_data.append(df_subset["Age"].tolist())
        
            xpos=[x+j for x in np.random.uniform(-0.2, 0.2, len(df_subset))]
            ax.scatter(xpos, df_subset["Age"], s=3, color=arm_color_dict[arm])
            
            # Plot boxp. no facecolor.
            box = ax.boxplot(df_subset["Age"], positions=[j], widths=0.12,
                             patch_artist=True, showcaps=True, boxprops=dict(facecolor='none', color=arm_color_dict[arm]),
                             whiskerprops=dict(color=arm_color_dict[arm]),
                             medianprops=dict(color=arm_color_dict[arm]),
                             flierprops=dict(marker='o', markersize=0, linestyle='none'), 
                             capprops=dict(color=arm_color_dict[arm]))
            # Annotate ns
            n_pts=df_subset.shape[0]
            ax.text(j, 46, f"n={n_pts}", fontsize=6, ha="center", va="bottom")
            
            # Annotate median age
            median=df_subset["Age"].median()
            ax.text(j+0.1, median, str(median), fontsize=6, va="center")
        
        # Run MWU
        p=mannwhitneyu(age_data[0], age_data[1]).pvalue
        p_rounded=round(p, 2)
        ax.text(0.5, 85, f"p={p_rounded}", fontsize=8, ha="center")
        
        # AES
        ax.set_xticks([0, 1])
        ax.set_xlim((-0.5, 1.5))
        ax.set_xticklabels(["Present", "Absent"], rotation=90)
        ax.set_xlabel("Treatment-emergent CH status")
        ax.set_title(arm)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("Age at randomization")
        ax.set_ylim((45, 86))
        
        if arm=="Cabazitaxel":
            ax.set_ylabel("")
    
    gs.tight_layout(fig)
    fig.savefig(f"{dir_figures}/supp/prog_ch_and_age.png", bbox_inches="tight")
    fig.savefig(f"{dir_figures}/supp/prog_ch_and_age.pdf", transparent=True, bbox_inches="tight")

# Generate the figures
compare_ages(progression_ch, path_clin, path_sample_information)
make_baseline_ch_progression_plot(baseline_ch_subset, progression_ch, path_sample_information, ax_lu, ax_caba)