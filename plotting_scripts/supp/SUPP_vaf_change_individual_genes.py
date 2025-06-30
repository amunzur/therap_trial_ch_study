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
import matplotlib.patches as patches
from matplotlib.path import Path

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
path_sample_info = f"{project_dir}/resources/sample_info.tsv"
path_weeks_to_progression=f"{project_dir}/resources/weeks_to_progression.csv"
baseline_ch_path=f"{project_dir}/CH_baseline.csv"
progression_ch_path=f"{project_dir}/CH_progression.csv"
dir_figures=f"{project_dir}/figures/supp"

path_utilities=f"{project_dir}/plotting_scripts/utilities.py"
with open(path_utilities, 'r') as file:
    script_code = file.read()
exec(script_code)

# LOAD DATASETS
sample_info = pd.read_csv(path_sample_info, sep= "\t")
pts_with_progression_samples = sample_info[sample_info["Timepoint"] == "FirstProgression"]["Patient_id"].drop_duplicates()
baseline_ch = pd.read_csv(baseline_ch_path)
baseline_ch_subset = baseline_ch[baseline_ch["Patient_id"].isin(pts_with_progression_samples)]
progression_ch = pd.read_csv(progression_ch_path)

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
baseline_ch_subset["Timepoint"]="Baseline"
progression_ch["Timepoint"]="FirstProgression"

combined_muts=pd.concat([baseline_ch_subset, progression_ch], ignore_index=True)
combined_muts.loc[combined_muts["Baseline VAF"]==0, "Baseline VAF"]=0.25
combined_muts.loc[combined_muts["Progression VAF"]==0, "Progression VAF"]=0.25

weeks_to_progression=pd.read_csv(path_weeks_to_progression)
combined_muts=combined_muts.merge(weeks_to_progression)
combined_muts["Date diff in months"]=combined_muts["Date diff in weeks"]*7/30

# PLOTTING
fig = plt.figure(figsize=(8, 4))
gs=gridspec.GridSpec(2, 7, height_ratios=[1, 1], width_ratios=[1,1,1,1,1,1,1], wspace=0.25, hspace=0.4)

plot_days=True
gene_list = ['DNMT3A', 'TET2', 'ASXL1', 'TP53', 'PPM1D', 'CHEK2', 'ATM']

first_ax = None
for i, arm in enumerate(["LuPSMA", "Cabazitaxel"]):
    for j, gene in enumerate(gene_list):
        if first_ax is None:
            ax=plt.subplot(gs[i, j])  # Assign subplot correctly
            first_ax = ax  # Store reference
        else:
            ax=plt.subplot(gs[i, j], sharex=first_ax)  # Share y-axis
        
        if plot_days and gene in ["PPM1D", "DNMT3A", "TET2"] and arm=="LuPSMA":
            ax_pie=inset_axes(ax, width="35%", height="35%", loc='upper right')
        else:
            ax_pie=inset_axes(ax, width="35%", height="35%", loc='upper left')
        
        combined_muts_subset=combined_muts[(combined_muts["Gene"]==gene) & (combined_muts["Arm"]==arm)]
        ax, nmuts=plot_vaf_change(combined_muts_subset, gene, ax, plot_delta=True, plot_days=plot_days)
        ax_pie=plot_pie(combined_muts_subset, gene, ax_pie)
        
        if gene=="DNMT3A":
            ax.set_title(r"$\bf{" + arm + r"}$" + f"\n{gene}", fontstyle="italic", loc="left", fontsize=6)
        else:
            ax.set_title(f"{gene}", fontstyle="italic", loc="left", fontsize=6)
        
        if i==0:
            ax.set_xlabel("")
            ax.tick_params("x", labelbottom=False)
            if not plot_days:
                ax.set_xticklabels("")
        if j>0:
            ax.set_yticklabels("")
            ax.set_ylabel("")
        if arm=="Cabazitaxel" and gene=="TP53":
            ax.set_xlabel("Months to progression")
        
        ax.set_ylim((-28, 35))
        ax.set_yticks([-20, -10, 0, 10, 20, 30])
        if j==0:
            ax.set_yticklabels(["-20", "-10", "0", "10", "20", "30"])
        else:
            ax.tick_params(labelleft=False)
        
        ax.set_xticks([0, 24, 48])
        ax.set_xticklabels(["0", "24", "48"])
        
        ax.text(3, -25, f"n={nmuts}", ha="left", va="bottom", fontsize=6)

gs_outer.tight_layout(fig)
fig.savefig(f"{dir_figures}/SUPP_vaf_change_individual_genes.png")
fig.savefig(f"{dir_figures}/SUPP_vaf_change_individual_genes.pdf", facecolor='none', transparent=True)