import pandas as pd
import numpy as np
import os
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import mannwhitneyu, ttest_ind

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


def compare_growth_rate_baseline_and_prog_BOX(df, ax, boxp_color_dict, scatter_color_dict):
    """
    Compares the growth rate of baseline and progression mutations. doesn't separate into baseline and progression.
    """
    col_to_use="Growth rate"
    
    # Boxp aes
    flierprops_dict = dict(marker='o', markersize=5, markeredgecolor='black', linestyle='None')
    whiskerprops_dict =dict(color='black')
    medianprops_dict = dict(color='black')
    capprops_dict = dict(color='black')
    
    baseline_muts=df[df["Timepoint"]=="Baseline"]
    progression_muts=df[df["Timepoint"]=="FirstProgression"]
        
    for i, mutation_list in enumerate([baseline_muts, progression_muts]):
        growth_rate_list=[]
        
        for x_offset, arm in zip([-0.22, 0.22], ["LuPSMA", "Cabazitaxel"]):
            gr=mutation_list[mutation_list["Arm"]==arm][col_to_use]
            boxprops_dict = dict(facecolor=boxp_color_dict[arm], edgecolor='black', linewidth = 0.7)  
            boxplot = ax.boxplot(gr, positions = [i+x_offset], flierprops = flierprops_dict, boxprops = boxprops_dict, medianprops = medianprops_dict, capprops = capprops_dict, widths = 0.4, showfliers = False, patch_artist = True)
            
            for gr_value in gr:
                ax.scatter(np.random.uniform(i + x_offset - 0.08, i + x_offset + 0.08), gr_value,
                    s=4, edgecolor="black", linewidths=0.2, color=scatter_color_dict[arm], 
                    marker="o", alpha=1, zorder=100)            
            
            gr_raw=mutation_list[mutation_list["Arm"]==arm]["Growth rate"].tolist()
            growth_rate_list.append(gr_raw)
        
        # Run MWU on VAFs
        p = float(f"{ttest_ind(growth_rate_list[0], growth_rate_list[1], equal_var=False).pvalue:.1g}")
        ax.text(i, 0.25, str(f"t test\np={p}"), ha='center', va='top', fontsize=7, color='black')
    
    ax.set_ylabel("Log growth rate", labelpad=-3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Progression"])
    ax.set_ylim((-0.2, 0.3))
    # ax.tick_params("x", bottom=False)
    ax.spines[["top", "right"]].set_visible(False)
    return(ax)

project_dir = os.environ.get("project_dir")

path_sample_info = f"{project_dir}/resources/sample_info.tsv"
path_weeks_to_progression=f"{project_dir}/resources/weeks_to_progression.csv"
baseline_ch_path=f"{project_dir}/CH_baseline.csv"
progression_ch_path=f"{project_dir}/CH_progression.csv"
dir_figures=f"{project_dir}/figures/main"

utilities=f"{project_dir}/plotting_scripts/utilities.py"
with open(utilities, 'r') as file:
    script_code = file.read()

exec(script_code)

arm_color_dict={"LuPSMA": "#6f1fff", "Cabazitaxel": "#3e3939"}
arm_color_dict_lighter={"LuPSMA": "#d3bbfe", "Cabazitaxel": "#b2b0b0"}
arm_color_dict_darker={"LuPSMA": "#e8e3f2", "Cabazitaxel": "#000000"}

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
combined_muts["Growth rate"]=np.log(combined_muts["Progression VAF"]/combined_muts["Baseline VAF"])/combined_muts["Date diff in weeks"]

fig = plt.figure(figsize=(2.5, 8))
gs = GridSpec(2, 1, figure=fig)
ax_ddr = fig.add_subplot(gs[0])
ax_dta = fig.add_subplot(gs[1])

dta_genes=["DNMT3A", "TET2", "ASXL1"]
ddr_genes=["PPM1D", "TP53", "ATM", "CHEK2"]

combined_muts_ddr=combined_muts[combined_muts["Gene"].isin(ddr_genes)]
combined_muts_dta=combined_muts[combined_muts["Gene"].isin(dta_genes)]

ax_ddr=compare_growth_rate_baseline_and_prog_BOX(combined_muts_ddr, ax_ddr, boxp_color_dict=arm_color_dict_lighter, scatter_color_dict=arm_color_dict)
ax_dta=compare_growth_rate_baseline_and_prog_BOX(combined_muts_dta, ax_dta, boxp_color_dict=arm_color_dict_lighter, scatter_color_dict=arm_color_dict)

ax_ddr.set_title("DDR genes")
ax_dta.set_title("DTA genes")

ax_ddr.set_ylim((-0.22, 0.25))
ax_dta.set_ylim((-0.22, 0.25))

fig.savefig(os.path.join(dir_figures, "MAIN_growth_rate.png"))
fig.savefig(os.path.join(dir_figures, "MAIN_growth_rate.pdf"), transparent=True)