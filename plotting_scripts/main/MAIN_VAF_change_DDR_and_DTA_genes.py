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
from scipy import stats

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
From baseline to progression plots VAF change in DDR and DTA genes in LuPSMA vs Cabazitaxel arms.
"""

def round_sig(x, sig=2):
    """
    Rounds to n significant figures.
    """
    if x == 0:
        return 0
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

def plot_boxp_vaf_changes(mutations_df, arm_color_dict, ax_boxp):
    """
    Plots a boxplot comparing the median VAF change in two treatment arms.
    """
    raw_vaf_data_dict={}
    for xpos, arm in enumerate(["LuPSMA", "Cabazitaxel"]):
        arm_subset_df=mutations_df[mutations_df["Arm"]==arm]
        arm_subset_df["VAF change"]=arm_subset_df["Progression VAF"]-arm_subset_df["Baseline VAF"]
                
        # Plot swarm plot
        vaf_data=arm_subset_df["VAF change"]
        for vaf in vaf_data:
                jitter=np.random.uniform(-0.3, 0.3, 1)
                ax_boxp.scatter(xpos+jitter, vaf, color=arm_color_dict[arm], s=0.5, alpha=0.5)
                
        # Append raw VAF change values for significance test
        raw_vaf_data_dict[arm]=arm_subset_df["VAF change"]
    
        # Annotate median and run
        median = round_sig(np.median(vaf_data))
        iqr = [round_sig(np.percentile(vaf_data, 25)), round_sig(np.percentile(vaf_data, 75))]
        ax_boxp.text(xpos, 30, median, fontsize=6, ha="center")
        ax_boxp.text(xpos, 25, iqr, fontsize=6, ha="center")
        
    # Run significance test
    mwu_result=mannwhitneyu(raw_vaf_data_dict["LuPSMA"], raw_vaf_data_dict["Cabazitaxel"])
    pval=mwu_result.pvalue
    pval_rounded=round_sig(pval, 4)
    ax_boxp.text(0.5, 20, f"MWU p={pval_rounded}", fontsize=6)
    
    # AES
    # ax_boxp.set_ylim((-np.log10(26), np.log10(34)))
    # ax_boxp.set_yticks([-np.log10(20), -np.log10(10), np.log10(0), np.log10(10), np.log10(20), np.log10(30)])
    # ax_boxp.set_yticklabels(["-20", "-10", "0", "10", "20", "30"])
    ax_boxp.set_xticks([0, 1])
    ax_boxp.set_xticklabels(["LuPSMA", "Cabazitaxel"])
    ax_boxp.set_xlim((-0.7, 1.3))
    ax_boxp.spines[["top", "right"]].set_visible(False)
    ax_boxp.set_ylabel("VAF%")
    
    return(ax_boxp)




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






# LOAD CHIP DATASETS
project_dir = os.environ.get("project_dir")

dir_figures=f"{project_dir}/figures/main"
baseline_ch_path=f"{project_dir}/CH_baseline.csv"
prog_ch_path=f"{project_dir}/CH_progression.csv"
path_sample_information = f"{project_dir}/resources/sample_info.tsv"

utilities=f"{project_dir}/plotting_scripts/utilities.py"

for path in [utilities]:
    with open(path, 'r') as file:
        script_code = file.read()
    
    exec(script_code)

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

fig = plt.figure(figsize=(5, 5))
outer_gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 0.2], width_ratios=[1, 1, 0.6], hspace = 0.15, wspace = 0.1)

all_genes=combined_muts["Gene"].unique()
DTA_genes=['DNMT3A', 'TET2', 'ASXL1']
DDR_genes=['PPM1D', 'TP53', 'ATM', 'CHEK2']

arm_color_dict={"LuPSMA": "#6f1fff", "Cabazitaxel": "#3e3939"}

first_ax = None
for i, (genes_list, genes_list_name) in enumerate(zip([all_genes, DTA_genes, DDR_genes], ["All genes", "DTA", "DDR"])):
        
    for j, arm in enumerate(["LuPSMA", "Cabazitaxel"]):
        if first_ax is None:
            ax = plt.subplot(outer_gs[i, j])  # First subplot
            first_ax = ax  # Store reference
        else:
            ax = plt.subplot(outer_gs[i, j], sharey=first_ax)  # Share y-axis
        
        subset_muts=combined_muts[(combined_muts["Gene"].isin(genes_list)) & (combined_muts["Arm"]==arm)]
        
        ax_pie=inset_axes(ax, width="35%", height="35%", loc='upper left')
        ax_pie=plot_pie(subset_muts, gene=None, ax_pie=ax_pie)
        
        ax, nmuts=plot_vaf_change(subset_muts, gene=None, ax=ax, plot_delta=True, plot_days=False)
        ax.set_title(f"{genes_list_name} {arm}")
        ax.set_xticklabels(["Baseline", "Progression"])
        ax.text(0.6, -20, str(f"n={nmuts}"), ha='left', va='center', fontsize=8, color='black')
        
        ax.set_yticks([-20, -10, 0, 10, 20, 30])
        ax.set_yticklabels([str(val) for val in ["-20", "-10", "0", "10", "20", "30"]])
        
        if j>0:
            ax.set_ylabel("")
        
    # Plot the boxplot of VAF changes
    ax_boxp=plt.subplot(outer_gs[i, 2], sharey=ax)
    mutations_df=combined_muts[combined_muts["Gene"].isin(genes_list)]
    ax_boxp=plot_boxp_vaf_changes(mutations_df, arm_color_dict, ax_boxp=ax_boxp)

# Add legend
ax_legend=plt.subplot(outer_gs[3, 0])
ax_legend.axis("off")
legend_colors = ["orangered", "royalblue", "mediumseagreen"]
legend_labels = ["VAF↑", "Stable", "VAF↓"]
legend_handles = [plt.Line2D([0], [0], marker='s', color=color, label=label, markersize=5, linestyle='') for color, label in zip(legend_colors, legend_labels)]
ax_legend.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize = 8, handlelength=2, handletextpad = 0.1, ncol=3)

outer_gs.tight_layout(fig)
fig.savefig(f"{dir_figures}/MAIN_DDR_vs_DTA_genes_vaf_change.png")
fig.savefig(f"{dir_figures}/MAIN_DDR_vs_DTA_genes_vaf_change.pdf", transparent=True)