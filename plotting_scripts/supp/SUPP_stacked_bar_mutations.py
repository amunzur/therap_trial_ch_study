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

mut_dict = {
    "Missense": '#79B443',
    "Stop gain": '#BD4398',
    "Frameshift InDel": '#FFC907',
    "Nonframeshift InDel": '#a9a9a9',
    "Splicing": "darkorange"}

renaming_dict={
    "missense": 'Missense',
    "stopgain": 'Stop gain',
    "frameshift_deletion": 'Frameshift InDel',
    "frameshift_insertion": "Frameshift InDel",
    "nonframeshift_deletion": 'Nonframeshift InDel',
    "nonframeshift_insertion": 'Nonframeshift InDel',
    "promoter": "Missense",
    "splicing": "Splicing"   
}

project_dir = os.environ.get("project_dir")

path_sample_info = f"{project_dir}/resources/sample_info.tsv"
baseline_ch_path=f"{project_dir}/CH_baseline.csv"
dir_figures=f"{project_dir}/figures/supp"

path_utilities=f"{project_dir}/plotting_scripts/utilities.py"
with open(path_utilities, 'r') as file:
    script_code = file.read()

exec(script_code)

# LOAD DATASETS
sample_info = pd.read_csv(path_sample_info, sep= "\t")
baseline_ch_main = pd.read_csv('https://docs.google.com/spreadsheets/d/1goOtwFlxgQDitW30VGVEz8hkT47OHqhDtjJJLpmb5I0/export?gid=0&format=csv')
# baseline_ch_main = pd.read_csv(baseline_ch_path)
df=baseline_ch_main[["Gene", "Effect"]]

gene_order=baseline_ch["Gene"].value_counts().reset_index()["index"].reset_index()
gene_order.columns=["xpos", "Gene"]

df=df.value_counts().reset_index().merge(gene_order)
df.columns=['Gene', 'Effect', 'n', 'xpos']
df["Effect"]=df["Effect"].map(renaming_dict)
df["color"]=df["Effect"].map(mut_dict)

fig, ax = plt.subplots(figsize=(4.5, 2.5))

effect_order = ["Missense", "Splicing", "Stop gain", "Frameshift InDel", "Nonframeshift InDel"]
for i, group in df.groupby("Gene"):
    group["Effect"]=group["Effect"].astype(pd.CategoricalDtype(categories=effect_order, ordered=True))
    group = group.sort_values("Effect")
    bottom=0
    for _, row in group.iterrows():
        ax.bar(row["xpos"], row["n"], bottom=bottom, color=row["color"])
        bottom+=row["n"]

n_genes=df["Gene"].unique().shape[0]
ax.spines[["top", "right"]].set_visible(False)
ax.set_xticks(range(0, n_genes))
ax.set_xticklabels(df["Gene"].unique(), rotation=90, fontsize=6)
ax.set_xlim(-1, n_genes+1)
ax.set_ylabel("Number of mutations")

# Add legend
legend_colors = mut_dict.values()
legend_labels = mut_dict.keys()
legend_handles = [plt.Line2D([0], [0], marker='s', color=color, label=label, markersize=5, linestyle='') for color, label in zip(legend_colors, legend_labels)]
ax.legend(handles=legend_handles, loc="upper right", frameon=False, handlelength=2, handletextpad = 0.1)

fig.tight_layout()
fig.savefig(f"{dir_figures}/SUPP_mut_distritbution.png")
fig.savefig(f"{dir_figures}/SUPP_mut_distritbution.pdf")