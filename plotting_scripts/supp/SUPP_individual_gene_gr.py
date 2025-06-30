import pandas as pd
import numpy as np
import os
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import ttest_ind

color_dict={"LuPSMA": "#8d75bd", "Cabazitaxel": "#7faf9d"}
boxp_color_dict={"LuPSMA": "#e3ddee", "Cabazitaxel": "#e0ebe7"}

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
combined_muts["Growth rate"]=np.log(combined_muts["Progression VAF"]/combined_muts["Baseline VAF"])/(combined_muts["Date diff in weeks"]*7/30)

# Comparing growth rate in specific genes, Lu vs Caba
fig = plt.figure(figsize=(7, 5))
gene_list_dta=['DNMT3A', 'TET2', 'ASXL1']
gene_list_ddr=['TP53', 'PPM1D', 'CHEK2', 'ATM']

gs_outer = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.5)

for i, (group_name, gene_group) in enumerate(zip(["DTA genes", "DDR genes"], [gene_list_dta, gene_list_ddr])):
    gs_gene_group=gridspec.GridSpecFromSubplotSpec(1, max(len(gene_list_ddr), len(gene_list_dta)), width_ratios = np.repeat(1, max(len(gene_list_ddr), len(gene_list_dta))), wspace=0.2, subplot_spec=gs_outer[i])
    for j, gene in enumerate(gene_group):
        ax=plt.subplot(gs_gene_group[j])
        ax, nmuts=compare_growth_rate_per_gene_BOX(combined_muts, ax, gene=gene)
        ax.set_title(f"{gene}", fontstyle="italic", loc="left", fontsize=10)
        if j>0:
            ax.set_yticklabels("")
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Growth rate (month)")
        if group_name=="DTA genes":
            ax_legend=plt.subplot(gs_gene_group[-1])
            ax_legend.spines[["top", "right", "left", "bottom"]].set_visible(False)
            ax_legend.set_xticks([])
            ax_legend.set_xticklabels([])
            ax_legend.set_yticks([])
            ax_legend.set_yticklabels([])
            
            timepoint_color_dict = {"First detected at baseline": "blue", "First detected at progression": "darkorange"}
            legend_colors = timepoint_color_dict.values()
            legend_labels = timepoint_color_dict.keys()
            legend_handles = [plt.Line2D([0], [0], marker='o', color=color, label=label, markeredgecolor ="black", markersize=5, linestyle='') for color, label in zip(legend_colors, legend_labels)]
            ax_legend.legend(handles=legend_handles, loc="upper right", frameon=False, handlelength=2, handletextpad = 0.1)
        
        # Plot two additional scatter that is cut off by the y limits
        if gene=="TET2":
            ax.scatter(-0.7, -0.6, marker="v", edgecolor='black', linewidth = 0.7, s=8, color="blue")
            ax.scatter(-0.1, -0.6, marker="v", edgecolor='black', linewidth = 0.7, s=8, color="blue")
            ax.text(-0.7, -0.58, "-1.3", va="bottom", ha="center", fontsize=6)
            ax.text(-0.1, -0.58, "-1.2", va="bottom", ha="center", fontsize=6)
            
        ax.set_ylim((-0.8, 0.8))
        ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])

fig.text(0.06, 0.45, "B", fontsize=20, fontweight="bold", ha="center")
fig.text(0.06, 0.90, "A", fontsize=20, fontweight="bold", ha="center")

gs.tight_layout(fig)
fig.savefig(os.path.join(dir_figures, "SUPP_individual_genes_gr_ln.pdf"), transparent=True)
fig.savefig(os.path.join(dir_figures, "SUPP_individual_genes_gr_ln.png"))