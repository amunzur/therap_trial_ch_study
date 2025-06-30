import pandas as pd
import numpy as np
import os
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.stats import ttest_ind
import seaborn as sns
from lifelines import KaplanMeierFitter
from adjustText import adjust_text
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
from datetime import datetime
import matplotlib.cm as cm
from lifelines.statistics import logrank_test
from scipy.stats import fisher_exact
from lifelines import KaplanMeierFitter
import matplotlib.ticker as ticker
import upsetplot
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import fisher_exact
import statsmodels.api as sm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.rcParams['font.size'] = 10
mpl.rcParams['text.color'] = 'k'
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['legend.handletextpad'] = '0.8'
mpl.rcParams['legend.labelspacing'] = '0.4'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['axes.labelsize'] = 10

"""
From baseline to progression plots VAF change in DDR and DTA genes in LuPSMA vs Cabazitaxel arms.
"""

def plot_pie(df, ax, baseline_colname="Baseline_VAF", prog_colname="Progression_VAF", gene_subset=None, arm=None):
    """
    """
    if gene_subset is not None:
        df=df[df["Gene"].isin(gene_subset)]
    
    if arm is not None:
        df=df[df["Arm"]==arm]
    
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    
    incr_muts=0
    decr_muts=0
    for i, row in df.iterrows():
        baseline_ch_vaf=row[baseline_colname]
        prog_ch_vaf=row[prog_colname]
        if prog_ch_vaf>baseline_ch_vaf: 
            incr_muts+=1
        else:
            decr_muts+=1
    
    ax.pie([incr_muts, decr_muts], labels=["VAF↑", "VAF↓"], autopct='%d%%', startangle=140, colors=["tomato", "deepskyblue"])
    return(ax)

utilities="/groups/wyattgrp/users/amunzur/therap_trial_ch_study/plotting_scripts/utilities.py"

for path in [utilities]:
    with open(path, 'r') as file:
        script_code = file.read()
    
    exec(script_code)


# LOAD CHIP DATASETS
project_dir = os.environ.get("project_dir")

dir_figures=f"{project_dir}/figures/main"
baseline_ch_path=f"{project_dir}/CH_baseline.csv"
prog_ch_path="{project_dir}/CH_progression.csv"
path_sample_information = f"{project_dir}/resources/sample_info.tsv"

sample_info=pd.read_csv(path_sample_information, sep="\t")
color_dict={"LuPSMA": "#8d75bd", "Cabazitaxel": "#7faf9d"}
boxp_color_dict={"LuPSMA": "#e3ddee", "Cabazitaxel": "#e0ebe7"}

pts_with_prog_samples=sample_info[sample_info["Timepoint"]=="FirstProgression"]["Patient_id"]

baseline_ch=pd.read_csv(baseline_ch_path)
progression_ch=pd.read_csv(prog_ch_path)
baseline_ch_subset=baseline_ch[baseline_ch["Patient_id"].isin(pts_with_prog_samples)]

baseline_ch_subset=harmonize_vaf_columns(baseline_ch_subset, timepoint="Baseline")
progression_ch=harmonize_vaf_columns(progression_ch, timepoint="FirstProgression")

baseline_ch_subset["Baseline alt"]=baseline_ch_subset["Alt_forward"]+baseline_ch_subset["Alt_reverse"]
baseline_ch_subset=baseline_ch_subset[['Patient_id', 'Arm', 'Gene', 'VAF%','Baseline alt', 'Depth', 'Progression vaf%', 'Progression alt count', 'Progression depth']]
baseline_ch_subset.columns=['Patient_id', 'Arm', 'Gene', 'Baseline VAF', 'Baseline alt', 'Baseline depth', 'Progression VAF', 'Progression alt', 'Progression depth']

progression_ch["Progression alt"]=progression_ch["Alt_forward"]+progression_ch["Alt_reverse"]
progression_ch=progression_ch[progression_ch["Independently detected at baseline"]==False]
progression_ch=progression_ch[['Patient_id', 'Arm', 'Gene', 'VAF%', 'Progression alt', 'Depth', 'Baseline vaf%', 'Baseline alt count', 'Baseline depth']]
progression_ch.columns=['Patient_id', 'Arm', 'Gene', 'Progression VAF', 'Progression alt', 'Progression depth', 'Baseline VAF', 'Baseline alt', 'Baseline depth']

########################
combined_muts=pd.concat([baseline_ch_subset, progression_ch], ignore_index=True)

combined_muts.loc[combined_muts["Baseline VAF"]==0, "Baseline VAF"]=0.25
combined_muts.loc[combined_muts["Progression VAF"]==0, "Progression VAF"]=0.25

fig = plt.figure(figsize=(6, 5))
outer_gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.2], width_ratios=[1, 1], hspace = 0.15, wspace = 0.15)

DTA_genes=['DNMT3A', 'TET2', 'ASXL1']
DDR_genes=['PPM1D', 'TP53', 'ATM', 'CHEK2']

first_ax = None
for i, (genes_list, genes_list_name) in enumerate(zip([DTA_genes, DDR_genes], ["DTA", "DDR"])):
    for j, arm in enumerate(["LuPSMA", "Cabazitaxel"]):
        if first_ax is None:
            ax = plt.subplot(outer_gs[i, j])  # First subplot
            first_ax = ax  # Store reference
        else:
            ax = plt.subplot(outer_gs[i, j], sharey=first_ax)  # Share y-axis
        
        subset_muts=combined_muts[(combined_muts["Gene"].isin(genes_list)) & (combined_muts["Arm"]==arm)]
        
        ax_pie=inset_axes(ax, width="35%", height="35%", loc='upper left')
        ax_pie=plot_pie(subset_muts, gene=None, ax_pie=ax_pie, label_var_incr=True)
        
        ax, nmuts=plot_vaf_change(subset_muts, gene=None, ax=ax, plot_delta=True, plot_days=False)
        ax.set_title(f"{genes_list_name} {arm}")
        ax.set_xticklabels(["Baseline", "Progression"])
        ax.text(0.6, -20, str(f"n={nmuts}"), ha='left', va='center', fontsize=8, color='black')
        
        ax.set_yticks([-20, -10, 0, 10, 20, 30])
        ax.set_yticklabels([str(val) for val in ["-20", "-10", "0", "10", "20", "30"]])
        
        if j>0:
            ax.set_ylabel("")

# Add legend
ax_legend=plt.subplot(outer_gs[2, 0])
ax_legend.axis("off")
legend_colors = ["orangered", "royalblue", "mediumseagreen"]
legend_labels = ["VAF↑", "Stable", "VAF↓"]
legend_handles = [plt.Line2D([0], [0], marker='s', color=color, label=label, markersize=5, linestyle='') for color, label in zip(legend_colors, legend_labels)]
ax_legend.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize = 8, handlelength=2, handletextpad = 0.1, ncol=3)

outer_gs.tight_layout(fig)
fig.savefig(f"{dir_figures}/MAIN_DDR_vs_DTA_genes_vaf_change.png")
fig.savefig(f"{dir_figures}/MAIN_DDR_vs_DTA_genes_vaf_change.pdf", transparent=True)