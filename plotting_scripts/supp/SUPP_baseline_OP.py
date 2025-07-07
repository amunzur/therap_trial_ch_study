import pandas as pd
import numpy as np
import os
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

project_dir = os.environ.get("project_dir")
baseline_ch_path=f"{project_dir}/CH_baseline.csv"
dir_figures=f"{project_dir}/figures/supp"
path_sample_info = f"{project_dir}/resources/sample_info.tsv"
path_age_df=f"{project_dir}/resources/age.csv"

color_dict={1: "#FFA0A0", 2: "#FF6565", 3: "#FF0000"}

# Load CH dataset
ch=pd.read_csv(baseline_ch_path)

# Load sample info and get arm-specific subsets
sample_info = pd.read_csv(path_sample_info, sep="\t")
baseline_info = sample_info[sample_info["Timepoint"] == "Baseline"]
baseline_info_lu = baseline_info[baseline_info["Arm"] == "LuPSMA"]
baseline_info_caba = baseline_info[baseline_info["Arm"] == "Cabazitaxel"]
n_total_dict = {"LuPSMA": len(baseline_info_lu), "Cabazitaxel": len(baseline_info_caba)}

# Count mutations per patient and gene
gene_order = ["DNMT3A", "TET2", "ASXL1", "PPM1D", "TP53", "CHEK2", "ATM", "SF3B1", "U2AF1", "SH2B3"]
gene_counts=ch[["Patient_id", "Arm", "Gene"]].value_counts().reset_index(name="Count")

gene_counts["Count"] = gene_counts["Count"].clip(upper=3)
gene_counts["Color"] = gene_counts["Count"].map(color_dict)
gene_counts = gene_counts[gene_counts["Gene"].isin(gene_order)]
gene_counts["Gene"] = pd.Categorical(gene_counts["Gene"], categories=gene_order, ordered=True)
gene_counts = gene_counts.sort_values(by=["Arm", "Gene", "Count"], ascending=[False, True, False])

# Enumerate samples for plotting
def enumerate_samples(df, arm_name):
    patients = df["Patient_id"].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={"index": "Samples_enumerated"})
    no_ch = [pid for pid in (baseline_info[baseline_info["Arm"] == arm_name]["Patient_id"]) if pid not in gene_counts["Patient_id"].unique()]
    no_ch_df = pd.DataFrame(no_ch, columns=["Patient_id"]).reset_index().rename(columns={"index": "Samples_enumerated"})
    no_ch_df["Samples_enumerated"] += patients["Samples_enumerated"].max() + 1
    return pd.concat([patients, no_ch_df], ignore_index=True)

samples_lu = enumerate_samples(gene_counts[gene_counts["Arm"] == "LuPSMA"], "LuPSMA")
samples_caba = enumerate_samples(gene_counts[gene_counts["Arm"] == "Cabazitaxel"], "Cabazitaxel")
samples_enumerated = pd.concat([samples_lu, samples_caba], ignore_index=True)
gene_counts = gene_counts.merge(samples_enumerated, on="Patient_id")

# Load and merge age info
clin_data = pd.read_csv(path_age_df)
norm = Normalize(vmin=clin_data["Age"].min(), vmax=clin_data["Age"].max())
cmap = cm.get_cmap("Blues")
clin_data["Age_color"] = clin_data["Age"].apply(lambda age: cmap(norm(age)))
clin_data = clin_data.merge(samples_enumerated, on="Patient_id").merge(sample_info[["Patient_id", "Arm"]])

# Compute gene frequency for horizontal bar plot
gene_freq = gene_counts[["Arm", "Gene", "Count", "Color"]].value_counts().reset_index(name="npts")
n_total_df=pd.DataFrame.from_dict(n_total_dict, orient='index', columns=["Total"]).reset_index().rename(columns={"index": "Arm"})
gene_freq=gene_freq.groupby(["Arm", "Gene"])["npts"].sum().reset_index().merge(n_total_df)
gene_freq["freq"]=gene_freq["npts"]/gene_freq["Total"]

# VAF info in log scale
ch_subset=ch[ch["Gene"].isin(gene_order)][["Arm", "Gene", "VAF_n", "VAF_t"]]
ch_subset["VAF"] = ch_subset["VAF_n"].combine_first(ch_subset["VAF_t"])
ch_subset["VAF log"]=np.log10(ch_subset["VAF"])

######################################
# Plotting
######################################
fig = plt.figure(figsize=(9, 3.5))
gs_outer=gridspec.GridSpec(1, 3, width_ratios=[1,1,0.35], wspace=0.1)

gs_lu=gridspec.GridSpecFromSubplotSpec(4, 2, width_ratios=[1, 0.3], height_ratios=[3, 4, 3, 0.4], wspace=0, subplot_spec=gs_outer[0])
gs_caba=gridspec.GridSpecFromSubplotSpec(4, 2, width_ratios=[1, 0.3], height_ratios=[3, 4, 3, 0.4], wspace=0, subplot_spec=gs_outer[1])

bar_height=0.8
bar_width=0.9

gene_dict={"dta": ["ASXL1", "TET2", "DNMT3A"], "ddr": ["ATM", "CHEK2", "TP53", "PPM1D"], "splicing": ["SF3B1", "U2AF1", "SH2B3"]}

for i, arm, arm_gs in zip([0, 2], ['LuPSMA', 'Cabazitaxel'], [gs_lu, gs_caba]):
    ax_age=plt.subplot(arm_gs[3, 0])
    arm_ntotal=n_total_dict[arm]
    arm_df=gene_counts[(gene_counts["Arm"]==arm)].drop_duplicates()
    ax_age.spines[["left", "right", "top", "bottom"]].set_visible(False)
    ax_age.set_yticks([0.5])
    ax_age.set_yticklabels(["Age"], fontsize=6)
    ax_age.tick_params(labelbottom=False, bottom=False, left=False, pad=-2)
    
    if arm=="Cabazitaxel":
        ax_age.tick_params(labelleft=False)
    
    clin_data_arm=clin_data[clin_data["Arm"]==arm]
    for a, row in clin_data_arm.iterrows():
        ax_age.bar(row["Samples_enumerated"], height=bar_height, width=bar_width, color=row["Age_color"])
    
    for j, gene_group in enumerate(["dta", "ddr", "splicing"]):
        gene_list=gene_dict[gene_group]
        ax=plt.subplot(arm_gs[j,0])
        ax_barh=plt.subplot(arm_gs[j,1])
        ax_swarm=ax_barh.twiny()
        
        if gene_group=="dta":
            ax.set_title(arm, fontsize=8)
        
        for k, gene in enumerate(gene_list):
            arm_gene_df=gene_counts[(gene_counts["Arm"]==arm) & (gene_counts["Gene"]==gene)]
            
            # Plot the gray squares
            for x in range(0, arm_ntotal):
                ax.bar(x, height=bar_height, width=bar_width, bottom=k, color="whitesmoke")
            
            # Plot the gene colors
            for l, row in arm_gene_df.iterrows():
                ax.bar(row["Samples_enumerated"], height=bar_height, width=bar_width, bottom=k, color=row["Color"])
                        
            # Plot the barh gene frequency
            gene_freq_arm_gene=gene_freq[(gene_freq["Arm"]==arm) & (gene_freq["Gene"]==gene)]
            ax_barh.barh(k, gene_freq_arm_gene["freq"].values[0], color="darkgray", height=bar_height)
            
            # Plot VAF swarm plot
            gene_vaf_df=ch_subset[(ch_subset["Gene"]==gene) & (ch_subset["Arm"]==arm)][["Gene", "VAF log"]]
            for _, row in gene_vaf_df.iterrows():
                jitter=np.random.uniform(-0.3, 0.3, 1)
                ax_swarm.scatter(row["VAF log"], k+jitter, color="black", s=0.5)
            
            # ax_barh.set_ylim((-0.5, len(gene_list)+0.5))
            ax_barh.set_xlim((0, 0.5))
            ax_barh.set_xticks([0, 0.25, 0.5])
            ax_barh.set_xticklabels(["0", "25", "50"], fontsize=6)
            ax_barh.spines[["top", "right"]].set_visible(False)
            ax_barh.tick_params(labelleft=False, left=False)
            
            ax_swarm.spines[["bottom", "right", "left"]].set_visible(False)
            ax_swarm.set_xlim(np.log10(0.22), np.log10(40))
            ax_swarm.set_xticks([np.log10(0.25), np.log10(2), np.log10(10), np.log(40)])
            ax_swarm.set_xticklabels(["0.25", "2", "10", "40"], fontsize=6)
            
            if gene_group!="dta":
                ax_swarm.tick_params(labeltop=False)
            else:
                ax_swarm.set_xlabel("VAF%", fontsize=6)
            
            if gene_group!="splicing":
                ax_barh.tick_params(labelbottom=False)
            else:
                ax_barh.set_xlabel("% with mutation", fontsize=6)
        
        # AES
        # ax.axis("off")
        ax.set_yticks([y + 0.5 for y in range(0, k+1)])
        ax.set_yticklabels(gene_list, fontstyle="italic", fontsize=8)
        ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelbottom=False)
        ax.tick_params(axis='y', pad=-2)
        if arm=="Cabazitaxel":
            ax.tick_params(labelleft=False)

# Add legends
ax_legend=plt.subplot(gs_outer[2])

legend_colors = color_dict.values()
legend_labels = ["≥3" if label == 3 else label for label in color_dict.keys()]
legend_handles = [plt.Line2D([0], [0], marker='s', color=color, label=label, markersize=5, linestyle='') for color, label in zip(legend_colors, legend_labels)]
leg1 = ax_legend.legend(handles=legend_handles, loc="upper right", title="n mutations", title_fontsize=6, frameon=False, handlelength=2, fontsize=6, labelspacing=0.2, handletextpad=0.05, ncol=1)
ax_legend.add_artist(leg1)

sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = plt.colorbar(sm, ax=ax_legend, orientation="vertical", fraction=0.15, pad=0.04)
cbar.set_label("Age", fontsize=6)
cbar.outline.set_visible(False)  # Remove frame
cbar.ax.tick_params(labelsize=6, width=0.5)  # Set tick label font size
box = ax_legend.get_position()
cbar.ax.set_position([box.x1 - 0.05, 0.3, 0.05, 0.25])  # [x0, y0, width, height] — tweak as needed

ax_legend.axis("off")

fig.savefig(os.path.join(dir_figures, "SUPP_Baseline_OP.png"))        
fig.savefig(os.path.join(dir_figures, "SUPP_Baseline_OP.pdf"))    