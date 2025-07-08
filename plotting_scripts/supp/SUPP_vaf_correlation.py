import pandas as pd
import numpy as np
import os
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

def plot_cfdna_and_wbc_vaf_correlation(ch_df, gene_color_dict, ax):
    """
    Plots correlation between cfDNA and WBC DNA. Plots each mutation as a dot. Colored based on gene.
    """
    ch_df_subset=ch_df[(~pd.isnull(ch_df["VAF_n"])) & (~pd.isnull(ch_df["VAF_t"]))]
    
    wbc_vaf_list=[]
    cfdna_vaf_list=[]
    
    for i, row in ch_df_subset.iterrows():
        dot_color=gene_color_dict.get(row["Gene"], "Silver")
        wbc_vaf=row["VAF_n"]
        cfdna_vaf=row["VAF_t"]
        
        wbc_vaf_list.append(wbc_vaf)
        cfdna_vaf_list.append(cfdna_vaf)
        
        ax.scatter(wbc_vaf, cfdna_vaf, color=dot_color, s=6)
    
    # Spearman correlation
    rho, pval = spearmanr(wbc_vaf_list, cfdna_vaf_list)
    
    # Best fit line using numpy polyfit (1st degree)
    coeffs = np.polyfit(wbc_vaf_list, cfdna_vaf_list, 1)
    x_vals = np.linspace(min(wbc_vaf_list), max(wbc_vaf_list), 100)
    y_vals = np.polyval(coeffs, x_vals)
    ax.plot(x_vals, y_vals, color='gray', linestyle='--', linewidth=1)
    
    # Add annotation
    ax.text(0.05, 0.95, f"Spearman r = {rho:.2f}\np<0.001", transform=ax.transAxes,ha='left', va='top', fontsize=6)
    
    return(ax)

mpl.rcParams['hatch.linewidth'] = 0.3
mpl.rcParams['font.size'] = 8
mpl.rcParams['text.color'] = 'k'
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['legend.handletextpad'] = '0.8'
mpl.rcParams['legend.labelspacing'] = '0.4'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['font.family'] = 'Arial'

project_dir = os.environ.get("project_dir")
baseline_ch_path=f"{project_dir}/CH_baseline.csv"
progression_ch_path=f"{project_dir}/CH_progression.csv"
dir_figures=f"{project_dir}/figures/supp"

gene_color_dict = {
    "DNMT3A": "#004c4c",
    "TET2": "#008080",
    "ASXL1": "#5cadad",
    "": "white",
    "TP53": "#943434",
    "PPM1D":"#D46666",
    "CHEK2":"#E08B8B",
    "ATM":"#ECAFAF",
    " ": "white",
    "Other": "Silver"
}

# LOAD DATASETS
baseline_ch = pd.read_csv(baseline_ch_path)
progression_ch = pd.read_csv(progression_ch_path)
progression_ch=progression_ch[progression_ch["Independently detected at baseline"]==False]

# PLOTTING
fig = plt.figure(figsize=(6, 3))

gs_outer=gridspec.GridSpec(1, 3, width_ratios=[1,1,0.5], hspace=0.38, wspace=0.3)
baseline_ax=fig.add_subplot(gs_outer[0])  
prog_ax=fig.add_subplot(gs_outer[1])
legend_ax=fig.add_subplot(gs_outer[2])

baseline_ax.set_aspect('equal')
prog_ax.set_aspect('equal')

baseline_ax=plot_cfdna_and_wbc_vaf_correlation(baseline_ch, gene_color_dict, baseline_ax)
prog_ax=plot_cfdna_and_wbc_vaf_correlation(progression_ch, gene_color_dict, prog_ax)

baseline_ax.set_xticks([0, 20, 40, 60, 80])
baseline_ax.set_xticklabels(["0", "20", "40", "60", "80"], fontsize=8)
baseline_ax.set_yticks([0, 20, 40, 60, 80])
baseline_ax.set_yticklabels(["0", "20", "40", "60", "80"], fontsize=8)
baseline_ax.set_xlabel("WBC DNA VAF%")
baseline_ax.set_ylabel("cfDNA VAF%")
baseline_ax.spines[["top", "right"]].set_visible(False)
baseline_ax.set_title("Baseline CH", fontsize=10)

prog_ax.set_xticks([0, 2, 4, 6, 8, 10])
prog_ax.set_xticklabels(["0", "2", "4", "6", "8", "10"], fontsize=8)
prog_ax.set_yticks([0, 2, 4, 6, 8, 10])
prog_ax.set_yticklabels(["0", "2", "4", "6", "8", "10"], fontsize=8)
prog_ax.set_xlabel("WBC DNA VAF%")
prog_ax.set_ylabel("cfDNA VAF%")
prog_ax.spines[["top", "right"]].set_visible(False)
prog_ax.set_title("Treatment-emergent CH", fontsize=10)

legend_ax.axis("off")

legend_colors = gene_color_dict.values()
legend_labels = gene_color_dict.keys()
legend_handles = [plt.Line2D([0], [0], marker='s', color=color, label=label, markersize=5, linestyle='') for color, label in zip(legend_colors, legend_labels)]
legend=legend_ax.legend(handles=legend_handles, loc="upper left", frameon=False, handlelength=2, handletextpad = 0.1)
[txt.set_fontstyle('italic') for txt, label in zip(legend.get_texts(), gene_color_dict.keys()) if label != "Other"]

# Add legend
fig.savefig(f"{dir_figures}/SUPP_vaf_correlation.png")
fig.savefig(f"{dir_figures}/SUPP_vaf_correlation.pdf")