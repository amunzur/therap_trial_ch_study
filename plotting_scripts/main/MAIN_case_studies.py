import pandas as pd
import numpy as np
import os
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import mannwhitneyu, ttest_ind
from matplotlib.patches import Patch

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

def plot_treatment_dates(pt, sample_info, df_cycles, ax):
    """
    Plots vertical lines indicative of treatment cycles
    """
    # Annotate when baseline and progression samples were collected
    baseline_date=sample_info[(sample_info["Patient_id"]==pt)&(sample_info["Timepoint"]=="Baseline")]["Date_collected"].values[0]
    progression_date=sample_info[(sample_info["Patient_id"]==pt)&(sample_info["Timepoint"]=="FirstProgression")]["Date_collected"].values[0]
    
    pt_df_cycles=df_cycles[df_cycles["Patient"]==pt]
    
    xtick_list=[]
    
    # Plot vertical tick for progression
    # ax.text(progression_date, ax.get_ylim()[1], "Progression", color="black", fontsize=8, rotation=90, ha='right', va='top')
    ax.axvline(progression_date, color="black", linestyle="--", linewidth=1)
        
    # Plot vertical lines for treatment cycles
    pt_df_cycles = df_cycles[df_cycles["Patient"] == pt]
    for _, row in pt_df_cycles.iterrows():
        cycle_date = pd.to_datetime(row["Treatment date"])
        ax.axvline(cycle_date, color="gray", linestyle="-", linewidth=0.5)
    
    # AES    
    ax.set_xlim(baseline_date - pd.Timedelta(days=60), progression_date + pd.Timedelta(days=60))
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.spines[["top", "right"]].set_visible(False)
    
    return(ax)

def plot_mutations(pt, progression_ch, sample_info, gene_color_dict, ax):
    """
    Plots VAF changes from baseline to progression
    """
    pt_muts=progression_ch[progression_ch["Patient_id"]==pt]
    pt_muts.loc[pt_muts["Baseline vaf%"]<0.25, "Baseline vaf%"]=0.25
    pt_muts["log Baseline vaf%"]=np.log10(pt_muts["Baseline vaf%"])
    pt_muts["log Progression vaf%"]=np.log10(pt_muts["Progression vaf%"])
    
    baseline_date=sample_info[(sample_info["Patient_id"]==pt)&(sample_info["Timepoint"]=="Baseline")]["Date_collected"].values[0]
    progression_date=sample_info[(sample_info["Patient_id"]==pt)&(sample_info["Timepoint"]=="FirstProgression")]["Date_collected"].values[0]
    
    for i, row in pt_muts.iterrows():
        edge_color="black" if row["Independently detected at baseline"] else "none"
        gene=row["Gene"]
        baseline_vaf=row["log Baseline vaf%"]
        progression_vaf=row["log Progression vaf%"]
        color=gene_color_dict[gene]
        
        ax.scatter(baseline_date, baseline_vaf, color=color, edgecolor=edge_color, s=20, zorder=3)
        ax.scatter(progression_date, progression_vaf, color=color, edgecolor="black", s=20, zorder=3)
        ax.plot([baseline_date, progression_date], [baseline_vaf, progression_vaf], color=color, linewidth=2, alpha=0.8, zorder=2)
    
    ax.set_ylim(np.log10(0.22), np.log10(100))
    ax.set_yticks([np.log10(0.25), np.log10(1), np.log10(2), np.log10(10), np.log10(50), np.log10(100)])
    ax.set_yticklabels(["0.25", "1", "2", "10", "50", "100"])
    ax.set_title(pt, loc="left", fontsize=10)
    return(ax)

gene_color_dict = {
    "ATM": "#ECAFAF",          # pale red-pink
    "CHEK2": "#E08B8B",        # light muted red
    "PPM1D": "#D46666",        # medium red
    "TP53": "#943434",         # deep red
    "ASXL1": "#5cadad",        # soft teal
    "TET2": "#008080",         # classic teal
    "DNMT3A": "#004c4c",       # dark teal
    "SF3B1": "forestgreen",    # rich green
    "SRSF2": "#77dd77",        # light green, softer than forestgreen
    "KMT2D": "#2c4cd9",        
    "IDH1": "#9966cc",         
    "BRCC3": "#ff6e60",        
    "NF1": "#65ffbc",          
    "RUNX1": "#b3b3b3"         
}

project_dir = os.environ.get("project_dir")

path_sample_info_w_dates = "/groups/wyattgrp/users/amunzur/lu_chip/resources/sample_lists/sample_info.tsv"
path_ncycles="/groups/wyattgrp/users/amunzur/lu_chip/resources/clinical_tables/TheraP supplementary tables - Treatment dates (22-Feb-2025).csv"
# baseline_ch_path=f"{project_dir}/CH_baseline.csv"
progression_ch_path=f"{project_dir}/CH_progression.csv"
dir_figures=f"{project_dir}/figures/main"

utilities=f"{project_dir}/plotting_scripts/utilities.py"
with open(utilities, 'r') as file:
    script_code = file.read()

progression_ch=pd.read_csv(progression_ch_path)
progression_ch=harmonize_vaf_columns(progression_ch, "FirstProgression")
sample_info=pd.read_csv(path_sample_info_w_dates, sep="\t")
sample_info["Date_collected"] = pd.to_datetime(sample_info["Date_collected"], format="%Y%b%d")
df_cycles=pd.read_csv(path_ncycles)

select_pts=["TheraP-107", 'TheraP-182', "TheraP-003"]
df=progression_ch[["Patient_id", "Gene", "VAF%", "Baseline vaf%", 'Protein_annotation', "Independently detected at baseline"]]
df=df[df["Patient_id"].isin(select_pts)]
df=df.rename(columns={"VAF%": "Progression vaf%"})

fig = plt.figure(figsize=(9, 3))
gs = GridSpec(1, 4, figure=fig)

for i, pt in enumerate(select_pts):
    ax=fig.add_subplot(gs[i])
    ax=plot_treatment_dates(pt, sample_info, df_cycles, ax=ax)
    ax=plot_mutations(pt, df, sample_info, gene_color_dict, ax)
    
    if i>0:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("VAF%")

# Add legend
ax_legend=ax=fig.add_subplot(gs[3])
handles = [Patch(facecolor=color, edgecolor='none', label=gene) for gene, color in gene_color_dict.items()]
ax_legend.axis("off")
ax_legend.legend(handles=handles, loc="center", frameon=False, ncol=3, fontsize=7)

fig.savefig(f"{dir_figures}/case_studies.png")
fig.savefig(f"{dir_figures}/case_studies.pdf")
