import pandas as pd
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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

def plot_per_patient_counts_grouped_bar_chart(muts_df, grouping_col_name, color_dict, group1_ntotal, group2_ntotal, ax, fontsize=5):
    """
    In a given ax plots the fraction of the bladder and kidney cohorts separately that have CH mutations.
    """
    # Extract names of groups
    group1_name = "LuPSMA"
    group2_name = "Cabazitaxel"
    
    # Determine colors for each group
    group1_color = color_dict[group1_name]
    group2_color = color_dict[group2_name]
    
    # get mut counts
    mutation_counts = muts_df.groupby([grouping_col_name, 'Patient_id']).size().reset_index(name='Mutation count')
    count_table = mutation_counts.groupby([grouping_col_name, 'Mutation count']).size().reset_index(name='Number of patients')
    pivot_table = count_table.pivot(index='Mutation count', columns=grouping_col_name, values='Number of patients').fillna(0).astype(int)
    pivot_table.columns = [f'Number of patients {col.lower()}' for col in pivot_table.columns]
    pivot_table.reset_index(inplace=True)
    pivot_table[f"{group1_name}_fraction"] = pivot_table[f"Number of patients {group1_name.lower()}"] / group1_ntotal
    pivot_table[f"{group2_name}_fraction"] = pivot_table[f"Number of patients {group2_name.lower()}"] / group2_ntotal
    
    # Get the number of patients with 0 mutations
    n_ch_positive_group1 = muts_df[muts_df[grouping_col_name] == group1_name]["Patient_id"].unique().shape[0]
    n_ch_positive_group2 = muts_df[muts_df[grouping_col_name] == group2_name]["Patient_id"].unique().shape[0]
    n_0_group1 = group1_ntotal - n_ch_positive_group1
    n_0_group2 = group2_ntotal - n_ch_positive_group2
    
    pts_with_0_muts_dict = {
        "Mutation count": 0, 
        f"Number of patients {group1_name.lower()}": n_0_group1, 
        f"Number of patients {group2_name.lower()}": n_0_group2,
        f"{group1_name}_fraction": n_0_group1/group1_ntotal,
        f"{group2_name}_fraction": n_0_group2/group2_ntotal}
    
    pts_with_0_muts_df = pd.DataFrame.from_dict(pts_with_0_muts_dict, orient='index').T
    pivot_table = pd.concat([pivot_table, pts_with_0_muts_df]).sort_values(by = "Mutation count").reset_index(drop = True)
    pivot_table["xpos"] = pivot_table.index
    # pivot_table.loc[pivot_table["xpos"] == 12, "xpos"] = 13
    
    # For the swarm plot get the vafs of all muts
    muts_df = muts_df.merge(mutation_counts, how = "left")
    group1_muts = muts_df[muts_df[grouping_col_name] == group1_name].reset_index(drop = True)
    group2_muts = muts_df[muts_df[grouping_col_name] == group2_name].reset_index(drop = True)
    
    # Plot them
    ax2 = ax.twinx()
    for i, row in pivot_table.iterrows():
        ax.bar(row["Mutation count"]-0.2, row[f"{group1_name}_fraction"], color=group1_color, width = 0.4, edgecolor = "None")
        ax.text(row["Mutation count"] - 0.2, 0.35, int(row[f"Number of patients {group1_name.lower()}"]), ha='center', va='center', fontsize=fontsize, color='black')
        
        ax.bar(row["Mutation count"]+0.2, row[f"{group2_name}_fraction"], color=group2_color, width = 0.4, edgecolor = "None")
        ax.text(row["Mutation count"] + 0.2, 0.35, int(row[f"Number of patients {group2_name.lower()}"]), ha='center', va='center', fontsize=fontsize, color='black')
    
    # plot the vafs in logscale
    muts_df["VAF_n_log"] = np.log10(muts_df["VAF_n"].replace(0, np.nan))
    for i, row in muts_df.iterrows():
        if row[grouping_col_name] == group1_name: 
            offset = -0.2
        else:
            offset = 0.2
        jitter = np.random.uniform(-0.08, 0.08, 1)
        ax2.scatter(row["Mutation count"]+offset+jitter[0], row["VAF_n_log"], color="black", s = 0.15, alpha = 0.7)
    
    # plot the vafs for bladder
    # for i, row in bladder_muts.iterrows():
    #     jitter = np.random.normal(-0.05, 0.05, 1)
    #     print(row["Mutation count"]-0.2+jitter, row["VAF_n"])
    #     ax2.scatter(row["Mutation count"]-0.2+jitter, row["VAF_n"], color="black", s = 2)
    
    # Aes
    for a in [ax, ax2]:
        a.spines["top"].set_visible(False)
    # x ticks
    pivot_table["Mutation count"]=pivot_table["Mutation count"].astype(int)
    ax.set_xticks(pivot_table["Mutation count"])
    ax.set_xticklabels(pivot_table["Mutation count"])
    ax.set_xlabel("Number of CH mutations")
    ax.set_ylabel("% of patients in arm")
    ax.tick_params(axis='x', bottom=False)
    ax2.tick_params(axis='x', pad=-5)
    # ax.tick_params(axis="both", direction="out", which="both", left=True, bottom=True , colors='k')    
    ax.set_ylim((0, 0.35))
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_yticklabels(["0", "10", "20", "30"])
    ax2.set_ylim((np.log10(0.25), np.log10(100)))
    ax2.set_yticks([np.log10(0.25), np.log10(1), np.log10(2), np.log10(10), np.log10(50), np.log10(100)])
    ax2.set_yticklabels([".25", "1", "2", "10", "50", "100"])
    ax2.set_ylabel("WBC VAF %")
    
    # Add legend
    legend_colors = color_dict.values()
    legend_labels = color_dict.keys()
    legend_handles = [plt.Line2D([0], [0], marker='s', color=color, label=label, markersize=5, linestyle='') for color, label in zip(legend_colors, legend_labels)]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize = 8, handlelength=2, handletextpad = 0.5)
    return([ax, ax2])

project_dir = os.environ.get("project_dir")
dir_figures=f"{project_dir}/figures/supp"

path_utilities=f"{project_dir}/plotting_scripts/utilities.py"
with open(path_utilities, 'r') as file:
    script_code = file.read()

exec(script_code)

baseline_ch_path=f"{project_dir}/CH_baseline.csv"
baseline_ch=pd.read_csv(baseline_ch_path)
baseline_ch=harmonize_vaf_columns(baseline_ch, "Baseline")
del baseline_ch["VAF_n"]
baseline_ch=baseline_ch.rename(columns={"VAF%": "VAF_n"})

lupsma_baseline_npts=96
caba_baseline_npts=82

fig, ax1 = plt.subplots(figsize=(3, 2.2))

ax1, ax1_twin = plot_per_patient_counts_grouped_bar_chart(muts_df = baseline_ch,
                                                          grouping_col_name = "Arm",
                                                          color_dict = {"LuPSMA": "#6f1fff", "Cabazitaxel": "#3e3939"},
                                                          group1_ntotal = lupsma_baseline_npts,
                                                          group2_ntotal = caba_baseline_npts,
                                                          ax = ax1, 
                                                          fontsize=6)
ax1.tick_params("x", bottom=True)
# ax1.set_title("Baseline CH")

fig.tight_layout()
fig.savefig(os.path.join(dir_figures, "n_muts_at_baseline.png"))
fig.savefig(os.path.join(dir_figures, "n_muts_at_baseline.pdf"))