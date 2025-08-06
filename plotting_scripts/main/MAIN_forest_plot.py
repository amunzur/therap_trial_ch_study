import pandas as pd
import numpy as np
import os
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu
from scipy.stats import fisher_exact
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import ttest_ind
import math
from decimal import Decimal
from matplotlib.lines import Line2D

mpl.rcParams['font.size'] = 8
mpl.rcParams['text.color'] = 'k'
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['legend.handletextpad'] = '0.8'
mpl.rcParams['legend.labelspacing'] = '0.4'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['font.family'] = 'Arial'

project_dir = os.environ.get("project_dir")

path_sample_info = f"{project_dir}/resources/sample_info.tsv"
baseline_ch_path=f"{project_dir}/CH_baseline.csv"
progression_ch_path=f"{project_dir}/CH_progression.csv"
utilities=f"{project_dir}/plotting_scripts/utilities.py"
dir_figures=f"{project_dir}/figures/main"

for path in [utilities]:
    with open(path, 'r') as file:
        script_code = file.read()
    
    exec(script_code)

# LOAD DATASETS
sample_info = pd.read_csv(path_sample_info, sep= "\t")
pts_with_progression_samples = sample_info[sample_info["Timepoint"] == "FirstProgression"]["Patient_id"].drop_duplicates()
baseline_ch = pd.read_csv(baseline_ch_path)
baseline_ch_subset = baseline_ch[baseline_ch["Patient_id"].isin(pts_with_progression_samples)]
progression_ch = pd.read_csv(progression_ch_path)
progression_ch=progression_ch[progression_ch["Independently detected at baseline"]==False]

def format_p_value(p, sig_digits=1):
    if p == 0:
        return "0.00"  # Edge case
    exponent = math.floor(math.log10(abs(p)))  # Get exponent
    decimal_places = max(0, sig_digits - exponent - 1)  # Ensure at least 0 decimals
    return f"{p:.{decimal_places}f}"

# PLOTTING
fig = plt.figure(figsize=(5, 3))
gs_forest=gridspec.GridSpec(3, 3, width_ratios=[1,0.7,0.7], height_ratios=[0.35, 1, 0.18], wspace=0.1, hspace=0.15)

ax_forest = plt.subplot(gs_forest[1, 0])
ax_annots_or = plt.subplot(gs_forest[1, 1], sharey=ax_forest)
ax_annots_p = plt.subplot(gs_forest[1, 2], sharey=ax_forest)
ax_x_annots=plt.subplot(gs_forest[2, 0], sharex=ax_forest)

gene_dict={
    "gene_group_name": ["DTA", "DDR", 'all genes'],
    "gene_group": [["ASXL1", "TET2", "DNMT3A",], ["ATM", "CHEK2", "TP53", "PPM1D"], [0]],
    "min_y": [0, 3.5, 8]
}

y_tick_pos_list=[]
y_tick_label_list=[]
y_tick_label_color=[]

pvals_dict={}
for i in range(len(gene_dict["gene_group_name"])):
    gene_group_name = gene_dict["gene_group_name"][i]
    gene_group = gene_dict["gene_group"][i]
    min_y = gene_dict["min_y"][i]
    
    print(gene_group_name, gene_group, min_y)
    
    # STEP 1. PLOT THE FOREST
    for j, gene in enumerate(gene_group):
        ypos=j+min_y
        # gene_color = gene_color_dict.get(gene, "black")
        gene_color="black"
                
        if gene_group_name=="all genes":
            progression_ch_subset=progression_ch[progression_ch["VAF_n"]>=gene]
        else:
            progression_ch_subset=progression_ch.copy()
        
        y_tick_pos_list.append(ypos)
        y_tick_label_color.append(gene_color)
        if gene==0:
            y_tick_label_list.append("All genes")
        else:
            y_tick_label_list.append(gene)
        
        or_dict=calculate_OR_and_p(progression_ch_subset, gene, ntotal_pt_lu=60, n_total_pt_caba=47)
        OR=or_dict["OR"]
        ci_lower=or_dict["CI_lower"]
        ci_higher=or_dict["CI_upper"]
        pval=or_dict["p Fisher"]
        
        pvals_dict[gene]=pval
        
        if OR!=float("inf"):
            ax_forest.errorbar(OR, ypos, xerr=[[OR-ci_lower], [ci_higher-OR]], fmt='o', color=gene_color, capsize=3, label="OR with 95% CI", linewidth=0.5, ms=3, elinewidth=0.5)
            
            OR_rounded=round(OR, 1)
            ci_lower_rounded=round(ci_lower, 1)
            ci_higher_rounded=round(or_dict["CI_upper"], 1)
            pval_reformatted=format_p_value(or_dict["p Fisher"])
            ax_annots_or.text(0, ypos, f"{OR_rounded} ({ci_lower_rounded}-{ci_higher_rounded})", ha="left", fontsize=8)
            ax_annots_p.text(0, ypos, f"{pval_reformatted}", ha="left", fontsize=8)
        
        else:
            ax_forest.text(2, ypos, "infinite", ha="left")

# Printing corrected P values, noncorrected are annotated on plot
pvals_dict["All genes"]=pvals_dict.pop(0)

rejected, pvals_corrected, _, _ = multipletests(list(pvals_dict.values()), alpha=0.05, method='fdr_bh')
corrected_pvals_dict = dict(zip(pvals_dict.keys(), pvals_corrected))


ax_forest.axvline(1, color='gray', linestyle='--', label='Neutral (OR=1)', linewidth=0.5)
ax_forest.set_ylim((-0.5, 9))
ax_forest.set_yticks(y_tick_pos_list)
ax_forest.set_yticklabels(y_tick_label_list, fontstyle="italic")
for label, color in zip(ax_forest.get_yticklabels(), y_tick_label_color):
    label.set_color(color)

ax_forest.set_xlim((-3, 30))
ax_forest.set_xticks((1, 5, 10, 15, 20, 25, 30))
ax_forest.set_xticklabels(('1', '5', '10', '15', '20', '25', '30'), fontsize=8)
ax_forest.spines[["left", "top", "right"]].set_visible(False)

ax_forest.tick_params("y", left=False, pad=-1)
# ax_forest.set_yticklabels(["All genes", "DTA", "DDR", "PPM1D"])
# ax_forest.scatter(ax_forest.get_xlim()[1]-0.3, 3, marker=">", color="black", s=9, edgecolor="None")
# ax_forest.scatter(ax_forest.get_xlim()[1]-0.3, 2, marker=">", color="black", s=9, edgecolor="None")

ax_annots_or.set_title("OR (95% CI)", loc="left")
ax_annots_p.set_title("p value", loc="left")
ax_forest.set_title("Newly acquired\nprogression mutations")
ax_forest.set_xlabel("")

ax_annots_or.axis('off')
ax_annots_p.axis('off')
ax_x_annots.axis('off')

ax_x_annots.text(1, 0.7, f"Favours Caba. ← OR → Favours Lu.", ha="center", va="top", fontsize=8)

gs_outer.tight_layout(fig)
fig.savefig(f"{dir_figures}/MAIN_forest.png")
fig.savefig(f"{dir_figures}/MAIN_forest.pdf", facecolor='none', transparent=True)