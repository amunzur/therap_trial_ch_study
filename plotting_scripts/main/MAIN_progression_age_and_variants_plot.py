import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportions_ztest
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

mpl.rcParams['hatch.linewidth'] = 0.3
mpl.rcParams['font.size'] = 8
mpl.rcParams['text.color'] = 'k'
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['legend.handletextpad'] = '0.8'
mpl.rcParams['legend.labelspacing'] = '0.2'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['font.family'] = 'Arial'

project_dir = os.environ.get("project_dir")

path_sample_info = f"{project_dir}/resources/sample_info.tsv"
path_ncycles=f"{project_dir}/resources/TheraP supplementary tables - Treatment dates (22-Feb-2025).csv"
baseline_ch_path=f"{project_dir}/CH_baseline.csv"
progression_ch_path=f"{project_dir}/CH_progression.csv"
path_age=f"{project_dir}/resources/age.csv"
path_clin_data=f"{project_dir}/clinical_data/TheraP supplementary tables - Clinical data (10-Dec-2023).tsv"
dir_figures=f"{project_dir}/figures/main"
utilities=f"{project_dir}/plotting_scripts/utilities.py"

for path in [utilities]:
    with open(path, 'r') as file:
        script_code = file.read()

exec(script_code)


arm_color_dict={"LuPSMA": "#6f1fff", "Cabazitaxel": "#3e3939"}
arm_color_dict_lighter={"LuPSMA": "#d3bbfe", "Cabazitaxel": "#b2b0b0"}

gene_color_dict = {
    "ATM":"#ECAFAF",
    "CHEK2":"#E08B8B",
    "PPM1D":"#D46666",
    "TP53": "#943434",

    "ASXL1": "#5cadad",
    "TET2": "#008080",
    "DNMT3A": "#004c4c",

    "SF3B1": "forestgreen",

    "Other": "Silver"
}

age_color_dict={
    "40-60": "gainsboro",
    "60-70": "silver",
    "70-80": "grey",
    "80-90": "dimgray"
}

# LOAD CLINICAL DATASETS
sample_info = pd.read_csv(path_sample_info, sep= "\t")
pts_with_progression_samples = sample_info[sample_info["Timepoint"] == "FirstProgression"]["Patient_id"].drop_duplicates()
age_df=pd.read_csv(path_age)

ncycles_df=pd.read_csv(path_ncycles)[["Patient", "Treatment arm", "Cycle"]]
ncycles_df=ncycles_df.groupby(["Patient", "Treatment arm"])["Cycle"].max().reset_index()
ncycles_df.columns=["Patient_id", "Arm", "Cycle"]

# LOAD CH DATASETS
progression_ch_main = pd.read_csv(progression_ch_path)
progression_ch_main=harmonize_vaf_columns(progression_ch_main, "FirstProgression")
baseline_ch_main = pd.read_csv(baseline_ch_path)
baseline_ch_main=harmonize_vaf_columns(baseline_ch_main, "Baseline")

# Determine CH status at baseline and progression for patients that have both baseline and progression samples available
baseline_ch_main_subset=baseline_ch_main[baseline_ch_main["Patient_id"].isin(pts_with_progression_samples)]
mut_status_baseline = annotate_mutation_status_lu(baseline_ch_main_subset, path_sample_info, annotate_what="CHIP", timepoint="Baseline", annotate_gene=False)
mut_status_prog = annotate_mutation_status_lu(progression_ch_main.rename(columns={"Timepoint": "Timepoint_t"}), path_sample_info, annotate_what="CHIP", timepoint="FirstProgression", annotate_gene=False)

# Generate nmuts df
progression_ch = progression_ch_main[["Patient_id", "Gene", "VAF%", "Independently detected at baseline"]]

# Count mutations grouped by Patient_id and whether detected at baseline
nmuts_prog_df = (progression_ch.groupby(["Patient_id", "Independently detected at baseline"]).size().unstack(fill_value=0).reset_index())
nmuts_prog_df.columns.name = None
nmuts_prog_df = nmuts_prog_df.rename(columns={
    True: "nmuts_also_at_baseline",
    False: "nmuts_new_at_progression"
})
nmuts_prog_df["total n muts"]=nmuts_prog_df["nmuts_also_at_baseline"]+nmuts_prog_df["nmuts_new_at_progression"]

# Merge with patients who have progression samples (if needed)
nmuts_prog_df = nmuts_prog_df.merge(pts_with_progression_samples, how="outer").fillna(0)

# Determine patient order based on age
pt_info=age_df.merge(ncycles_df)
pt_info=pt_info[pt_info["Patient_id"].isin(pts_with_progression_samples)]
bins = [40, 60, 70, 80, 90]
labels = ["40-60", "60-70", "70-80", "80-90"]
pt_info["Age_bin"] = pd.cut(pt_info["Age"], bins=bins, labels=labels, right=False)
pt_info=pt_info.merge(nmuts_prog_df)
pt_info=pt_info.sort_values(["Age_bin", "Arm", "total n muts"]).reset_index(drop=True)

# Organize the gene muts df
gene_df=progression_ch_main[["Patient_id", "Gene", "VAF%"]]
gene_df["log vaf"]=np.log10(gene_df["VAF%"])
gene_df["color"]=gene_df["Gene"].map(gene_color_dict).fillna("silver")

# PLOTTING
fig = plt.figure(figsize=(8, 4.5))
gs_outer=gridspec.GridSpec(5, 1, height_ratios=[1,0.05,0.1,0.1,1], hspace=0.05)
ax0=plt.subplot(gs_outer[0])
ax1=plt.subplot(gs_outer[1], sharex=ax0)
ax2=plt.subplot(gs_outer[2], sharex=ax0)
ax3=plt.subplot(gs_outer[3], sharex=ax0)
ax4=plt.subplot(gs_outer[4], sharex=ax0)

# Plot the age distribution barchart
n_age_groups=pt_info["Age_bin"].value_counts().reset_index()
n_age_groups.columns=["Age bin", "n"]
n_age_groups=n_age_groups.sort_values("Age bin").reset_index(drop=True)

n_age_groups=pt_info[["Age_bin", "Arm"]].value_counts().reset_index()
n_age_groups.columns=["Age bin", "Arm", "n"]
n_age_groups=n_age_groups.sort_values(["Age bin", "Arm"]).reset_index(drop=True)

# Plot number of cycles of treatment
norm = mcolors.Normalize(vmin=pt_info["Cycle"].min(), vmax=pt_info["Cycle"].max())
cmap = cm.get_cmap("Blues")  # Choose your colormap

# Plot age distr
left_age = 0
left_treatment = 0

for age_bin, group in n_age_groups.groupby("Age bin"):
    # Plot age bin total on ax1
    age_n = group["n"].sum()
    age_color = age_color_dict[age_bin]
    ax2.barh(0, age_n, left=left_age, color=age_color)

    # Annotate age group on the bars
    x_midpoint=age_n/2+left_age
    left_age += age_n

    if age_bin=="80-90":
        text_color="white"
    else:
        text_color="black"
    ax2.text(x_midpoint, 0, age_bin, fontsize=8, ha="center", va="center", color=text_color)

    # Plot treatments within that age bin on ax2
    for _, row in group.iterrows():
        arm_color = arm_color_dict[row["Arm"]]
        ax1.barh(0, row["n"], left=left_treatment, color=arm_color)
        left_treatment += row["n"]

    ax1.axis("off")


# Plot n muts
for xpos, row in pt_info.iterrows():
    pt_id=row["Patient_id"]

    ax0.bar(xpos, row["nmuts_new_at_progression"], color=arm_color_dict[row["Arm"]], zorder=2)
    ax0.bar(xpos, row["nmuts_also_at_baseline"], bottom=row["nmuts_new_at_progression"], color=arm_color_dict_lighter[row["Arm"]], zorder=2)

    # Plot number of cycles of treatment received
    ax3.bar(xpos, 1,color=cmap(norm(row["Cycle"])))

    # Now plot the gene VAFs
    gene_vafs_pt=gene_df[gene_df["Patient_id"]==pt_id]
    ax4.scatter(np.repeat(xpos, gene_vafs_pt.shape[0]), -gene_vafs_pt["log vaf"], color=gene_vafs_pt["color"], s=2, zorder=1)

    # vertical line up to the highest VAF (which is the lowest numerical)
    y_min=(-gene_vafs_pt["log vaf"]).min()
    ax4.plot([xpos, xpos], [y_min, -np.log10(0.22)], color="gray", linewidth=0.2, zorder=0)

ax4.set_ylim(-np.log10(100), -np.log10(0.22))
ax4.set_yticks([-np.log10(0.25), -np.log10(1), -np.log10(2), -np.log10(10), -np.log10(50), -np.log10(100)])
ax4.set_yticklabels(["0.25", "1", "2", "10", "50", "100"])
ax4.spines[["bottom", "right"]].set_visible(False)
ax4.set_ylabel("VAF%")

ax0.set_xlim((-1, xpos+1))
ax0.spines[["top", "right"]].set_visible(False)
ax0.set_yticks([0, 5, 10, 15, 20])
ax0.set_yticklabels(['0', "5", '10', '15', "20"])
ax0.set_xticks([])
ax0.set_ylabel("Number of mutations")

# Plot tick lines
# for y_tick in ax0.get_yticks():
#     ax0.axhline(y_tick, linewidth=0.2, linestyle="--", color="black", zorder=0)

for ax in [ax2, ax3]:
    ax.tick_params("both", bottom=False, labelbottom=False, left=False, labelleft=False)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

ax3.set_ylabel("Number of cycles", rotation=0, ha="right", va="center")
ax2.set_ylabel("Age", rotation=0, ha="right", va="center")

# LEGENDS
# LuPSMA legend block #############################
lupsma_handles = [Line2D([0], [0], marker='s', color=c, label=l, markersize=5, linestyle='')
                  for c, l in zip([arm_color_dict["LuPSMA"], arm_color_dict_lighter["LuPSMA"]],
                                  ["Tx-emergent", "Pre-existing"])]
lupsma_legend = ax0.legend(handles=lupsma_handles, loc="upper left", bbox_to_anchor=(0, 1), frameon=False, fontsize=6, handlelength=2, handletextpad=0.4, borderpad=0.3, title="LuPSMA", title_fontsize=7)
ax0.add_artist(lupsma_legend)

# Cabazitaxel legend block #############################
caba_handles = [Line2D([0], [0], marker='s', color=c, label=l, markersize=5, linestyle='')
                for c, l in zip([arm_color_dict["Cabazitaxel"], arm_color_dict_lighter["Cabazitaxel"]],
                                ["Tx-emergent", "Pre-existing"])]
caba_legend = ax0.legend(handles=caba_handles, loc="upper left", bbox_to_anchor=(0.12, 1), frameon=False, fontsize=6, handlelength=2, handletextpad=0.4, borderpad=0.3, title="Cabazitaxel", title_fontsize=7)
ax0.add_artist(caba_legend)

# Gene color legend block #############################
# Split genes into categories
ddr_genes = ["TP53", "PPM1D", "CHEK2", "ATM"]  # Example DDR genes
dta_genes = ["DNMT3A", "TET2", "ASXL1"]
other_genes = ["SF3B1", "Other"]  # Add other relevant genes if needed

# Helper to create legend handles
def make_handles(genes):
    return [
        Line2D([0], [0], marker='s', color=gene_color_dict[g],
               label=(g if g == "Other" else f"${g}$"), markersize=5, linestyle='')
        for g in genes if g in gene_color_dict
    ]

# Create handles per category
ddr_handles = make_handles(ddr_genes)
dta_handles = make_handles(dta_genes)
other_handles = make_handles(other_genes)

# Add legends side-by-side
ddr_legend = ax0.legend(handles=ddr_handles, loc="upper left", bbox_to_anchor=(0.27, 1), frameon=False, fontsize=6, title_fontsize=7, handlelength=2, handletextpad=0.4, borderpad=0.3)
ax0.add_artist(ddr_legend)

dta_legend = ax0.legend(handles=dta_handles, loc="upper left", bbox_to_anchor=(0.37, 1), frameon=False, fontsize=6, title_fontsize=7, handlelength=2, handletextpad=0.4, borderpad=0.3)
ax0.add_artist(dta_legend)

other_legend = ax0.legend(handles=other_handles, loc="upper left", bbox_to_anchor=(0.47, 1), frameon=False, fontsize=6, title_fontsize=7, handlelength=2, handletextpad=0.4, borderpad=0.3)
ax0.add_artist(other_legend)

# Cycles colorbar #############################
norm = Normalize(vmin=pt_info["Cycle"].min(), vmax=pt_info["Cycle"].max())
cmap = cm.Blues
cax = fig.add_axes([0.6, 0.75, 0.015, 0.12])
cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
cb.set_label("Number of cycles", fontsize=6)
cb.set_ticks([1, 5, 10])
cb.ax.tick_params(labelsize=6)
cb.ax.yaxis.set_label_position('left')
for spine in cb.ax.spines.values():
    spine.set_linewidth(0.5)  # or any smaller linewidth you want

# Make ticks thinner
cb.ax.tick_params(width=0.5)

fig.savefig(f"{dir_figures}/MAIN_treatment_muts.png")
fig.savefig(f"{dir_figures}/MAIN_treatment_muts.pdf")