"""
Plots the mutation VAFs in select genes and compares between treatmenr groups.
"""

import pandas as pd
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

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

def make_boxp_for_vaf(muts_df, lighter_color_dict, darker_color_dict, ax, genes_list, add_legend=True):
    """
    Makes boxplots with swarm plot to compare the VAF between treatment groups
    """    
    xtick_list = []
    raw_pvals = []
    pval_positions = []
    
    for i, gene in enumerate(genes_list):
        vaf_data = {}
        for arm, offset in zip(["LuPSMA", "Cabazitaxel"], [-0.15, 0.15]):
            gene_vafs_list=np.log10(muts_df[(muts_df["Gene"]==gene)&(muts_df["Arm"]==arm)]["VAF%"])
            xpos=i+offset
            vaf_data[arm] = gene_vafs_list
            
            # Plot scatter
            for vaf in gene_vafs_list:
                jitter=np.random.uniform(-0.1, 0.1, 1)
                ax.scatter(xpos+jitter, vaf, color=lighter_color_dict[arm], s=2)
            
            # Plot boxp. no facecolor.
            box = ax.boxplot(gene_vafs_list, positions=[xpos], widths=0.12,
                             patch_artist=True, showcaps=False, boxprops=dict(facecolor='none', color=darker_color_dict[arm]),
                             whiskerprops=dict(color=darker_color_dict[arm]),
                             medianprops=dict(color=darker_color_dict[arm]),
                             flierprops=dict(marker='o', markersize=0, linestyle='none'))
            
        # Store raw p-value for later correction
        if all(len(vaf_data[arm]) > 0 for arm in ["LuPSMA", "Cabazitaxel"]):
            stat, p = mannwhitneyu(vaf_data["LuPSMA"], vaf_data["Cabazitaxel"], alternative='two-sided')
            raw_pvals.append(p)
            pval_positions.append(i)
        else:
            raw_pvals.append(np.nan)
            pval_positions.append(i)
                
        xtick_list.append(i)
    
    # Correct for multiple comparisons
    corrected = multipletests(raw_pvals, method='fdr_bh')
    corrected_pvals = corrected[1]
    
    # Annotate adjusted p-values
    for i, p in zip(pval_positions, corrected_pvals):
        if np.isnan(p):
            continue
        p_text = f"p={p:.1g}" if p <= 0.05 else "ns"
        ax.text(i, np.log10(105), p_text, ha="center", fontsize=8)
    
    ax.set_ylim((np.log10(0.2), np.log10(100)))
    ax.set_yticks([np.log10(0.25), np.log10(0.5), np.log10(1), np.log10(2), np.log10(10), np.log10(50), np.log10(100)])
    ax.set_yticklabels(["0.25", "0.5", "1", "2", "10", "50", "100"])
    ax.set_xticks(xtick_list)
    ax.set_xticklabels(genes_list, rotation=90, fontstyle="italic")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("VAF%")
    
    # Add legend
    if add_legend:
        legend_colors = darker_color_dict.values()
        legend_labels = darker_color_dict.keys()
        legend_handles = [plt.Line2D([0], [0], marker='s', color=color, label=label, markersize=5, linestyle='') for color, label in zip(legend_colors, legend_labels)]
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, handlelength=2, handletextpad = 0.1, ncol=1)
    
    return(ax)

project_dir = os.environ.get("project_dir")
baseline_ch_path=f"{project_dir}/CH_baseline.csv"
dir_figures=f"{project_dir}/figures/supp"

path_utilities=f"{project_dir}/plotting_scripts/utilities.py"
with open(path_utilities, 'r') as file:
    script_code = file.read()
exec(script_code)

arm_color_dict={"LuPSMA": "#6f1fff", "Cabazitaxel": "#3e3939"}
arm_color_dict_lighter={"LuPSMA": "#d3bbfe", "Cabazitaxel": "#b2b0b0"}
arm_color_dict_darker={"LuPSMA": "#e8e3f2", "Cabazitaxel": "#000000"}

baseline_ch = pd.read_csv(baseline_ch_path)
baseline_ch=harmonize_vaf_columns(baseline_ch, timepoint="Baseline")

# PLOTTING
# Baseline CH
fig, ax = plt.subplots(figsize=(7, 4))
genes_list=["DNMT3A", "TET2", "ASXL1", "TP53", "PPM1D", "ATM", "CHEK2", "KMT2D", "SF3B1"]

ax=make_boxp_for_vaf(muts_df=baseline_ch, lighter_color_dict=arm_color_dict, darker_color_dict=arm_color_dict, ax=ax, genes_list=genes_list)

ax.set_title("Baseline CH", loc='left', pad=10, x=0.0)
fig.tight_layout()
fig.savefig(f"{dir_figures}/SUPP_baseline_VAF_boxplot.png")
fig.savefig(f"{dir_figures}/SUPP_baseline_VAF_boxplot.pdf", transparent=True)