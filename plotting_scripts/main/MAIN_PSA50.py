import pandas as pd
import numpy as np
import os
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats

project_dir = os.environ.get("project_dir")

path_muts=f"{project_dir}/CH_baseline.csv"
path_clin=f"{project_dir}/clinical_data/TheraP supplementary tables - Clinical data (10-Dec-2023).tsv"
PATH_sample_information=f"{project_dir}/resources/sample_info.tsv"
dir_figures=f"{project_dir}/figures/main"

utilities=f"{project_dir}/plotting_scripts/utilities.py"
with open(utilities, 'r') as file:
    script_code = file.read()

exec(script_code)

sample_info=pd.read_csv(PATH_sample_information, sep="\t")
clin=pd.read_csv(path_clin, sep="\t", skiprows=1)
psa_clin_main=clin[['Patient', 'Best PSA response', 'PSA50 response']].rename(columns={'Patient': 'Patient_id'})

muts=pd.read_csv(path_muts)
mut_status=annotate_mutation_status_lu(muts, PATH_sample_information, annotate_what="CHIP", timepoint="Baseline", annotate_gene = False).merge(sample_info[['Patient_id', 'Arm']].drop_duplicates())

# Group patients based on max CH VAF
max_ch_df=muts.groupby("Patient_id")["VAF_n"].max().reset_index()
ch_neg=mut_status[mut_status["CHIP status"]=="Negative"][["Patient_id", "Arm"]].merge(sample_info[['Patient_id', 'Arm']].drop_duplicates())
ch_pos=muts[["Patient_id", "Arm"]].drop_duplicates()

muts_group1=max_ch_df[(max_ch_df["VAF_n"]>=0.25) & (max_ch_df["VAF_n"]<2)].merge(sample_info[['Patient_id', 'Arm']].drop_duplicates())
muts_group2=max_ch_df[(max_ch_df["VAF_n"]>=2) & (max_ch_df["VAF_n"]<10)].merge(sample_info[['Patient_id', 'Arm']].drop_duplicates())
muts_group3=max_ch_df[max_ch_df["VAF_n"]>=10].merge(sample_info[['Patient_id', 'Arm']].drop_duplicates())
muts_group4=pd.concat([ch_neg, muts_group1[["Patient_id", "Arm"]]])
                                
color_dict_psa50_response={"Yes": "#c0cced", "No": "#ffbba2"}
color_dict_arm={"LuPSMA": "#6f1fff", "Cabazitaxel": "#3e3939"}
ntotal_dict={"LuPSMA": 94, "Cabazitaxel": 80}

def plot_psa_waterfall(psa_clin_subset, stratify_by, color_dict, ax, add_legend):
    """
    Plots a waterfall bar chart of the best PSA response for each patient.
    """
    psa_clin_subset["bar_color"]=psa_clin_subset[stratify_by].map(color_dict)
    ax.bar(psa_clin_subset["Patient_id"], psa_clin_subset["Best PSA response"], color=psa_clin_subset["bar_color"])
    if add_legend:
        legend_colors = color_dict.values()
        legend_labels = color_dict.keys()
        legend_handles = [plt.Line2D([0], [0], marker='s', color=color, label=label, markersize=5, linestyle='') for color, label in zip(legend_colors, legend_labels)]
        ax.legend(handles=legend_handles, loc="best", frameon=False, fontsize = 6, title_fontsize=6, handlelength=2, handletextpad = 0.1, title=stratify_by)
    #
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([])
    ax.set_xticklabels([])   
    return(ax)

def plot_psa_waterfall_grid(fig, gs, groups, y_labels, arms=["LuPSMA", "Cabazitaxel"], add_legend_waterfall=True):
    """
    Creates a grid of PSA waterfall plots for different CH groups and treatment arms.
    
    Parameters:
        fig (matplotlib.figure.Figure): Figure object.
        gs (matplotlib.gridspec.GridSpec): GridSpec object.
        groups (list): List of DataFrames for CH groups.
        y_labels (list): List of y-axis labels corresponding to groups.
        arms (list, optional): Treatment arms to plot.
        add_legend_waterfall (bool, optional): Whether to add a legend for the waterfall plots.
    """
    for i, arm in enumerate(arms):
        for j, (y_label, ch_df) in enumerate(zip(y_labels, groups)):
            psa_df = ch_df[ch_df["Arm"] == arm].merge(psa_clin_main).sort_values(by="Best PSA response", ascending=False)
            n_pts=psa_df.shape[0]
            ax = fig.add_subplot(gs[i, j])
            ax.set_ylabel(arm if j == 0 else "", labelpad=-3)
            ax.set_title(f"{y_label}\nn={n_pts}", fontsize=8)
            ax.set_ylim((-100, 50))
            ax.set_yticks([-100, -50, 0, 50])
            ax.set_yticklabels(["-100", "-50", "0", "50"], fontsize=8)
            ax = plot_psa_waterfall(psa_df, stratify_by="PSA50 response", color_dict=color_dict_psa50_response, ax=ax, 
                                    add_legend=(add_legend_waterfall and j == len(groups) - 1 and i == 1))

def plot_psa_response_bar_grid(fig, gs, groups, y_labels, add_legend_bar=True, arms=["LuPSMA", "Cabazitaxel"]):
    """
    Creates a grid of PSA response proportion bar plots for different CH groups.
    
    Parameters:
        fig (matplotlib.figure.Figure): Figure object.
        gs (matplotlib.gridspec.GridSpec): GridSpec object.
        groups (list): List of DataFrames for CH groups.
        y_labels (list): List of y-axis labels corresponding to groups.
        add_legend_bar (bool, optional): Whether to add a legend for the bar plots.
    """
    x_offset=0.4
    for i, arm in enumerate(arms):
        ax=fig.add_subplot(gs[i, gs.ncols - 1])
        ticklabel_list=[]
        values_list_contingency_table=[]
        
        for j, (group, ylabel) in enumerate(zip(groups, y_labels)):
            group_arm=group[group["Arm"]==arm].merge(psa_clin_main).sort_values(by="Best PSA response", ascending=False)
            group_arm_psa_pos=group_arm[group_arm["PSA50 response"]=="Yes"].shape[0]
            group_arm_psa_neg=group_arm[group_arm["PSA50 response"]=="No"].shape[0]
            
            ax.bar(j, group_arm_psa_pos, color=color_dict_psa50_response["Yes"])
            ax.bar(j, group_arm_psa_neg, bottom=group_arm_psa_pos, color=color_dict_psa50_response["No"])
            
            ticklabel_list.append(ylabel)
            values_list_contingency_table.append([group_arm_psa_pos, group_arm_psa_neg])
                
        ax.set_xticks(range(0, len(groups)))
        if len(groups)>2:
            ax.set_xticklabels(ticklabel_list, fontsize=8, rotation=90)
        else:
            ax.set_xticklabels(ticklabel_list, fontsize=8)
        
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.set_yticklabels(["0", "20", "40", "60", "80"], fontsize=8)
        ax.set_ylabel("n patients")
        
        # Run significance test and print on plot
        contingency_table = values_list_contingency_table
        if len(groups)>2:
            # Run chi square test
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            p_formatted=f"{p:.1g}"
            print("Chi-square p-value:", p_formatted)
        else:
            # Run Fisher's
            oddsratio, p = stats.fisher_exact(contingency_table)
            p_formatted=f"{p:.1g}"
            print("Fisher's exact p-value:", p)
        
        xpos=j/2
        ax.text(xpos, 78, f"p={p_formatted}", va="top", ha="center", fontsize=6)


# Plot: CH presence vs absence
fig = plt.figure(figsize=(5.5, 4))
gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.5], hspace=0.3, wspace=0.4)
plot_psa_waterfall_grid(fig, gs, [ch_neg, ch_pos], ["CH-", "CH+"])
plot_psa_response_bar_grid(fig, gs, [ch_neg, ch_pos], ["CH-", "CH+"])
fig.savefig(os.path.join(dir_figures, "main", "MAIN_PSA_response_CH_positive_vs_negative.png"))
fig.savefig(os.path.join(dir_figures, "main", "MAIN_PSA_response_CH_positive_vs_negative.pdf"), transparent=True)

# Plot 1: Stratification based on CH VAF
fig = plt.figure(figsize=(8, 5))
gs = gridspec.GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.5], hspace=0.5)
plot_psa_waterfall_grid(fig, gs, [ch_neg, muts_group1, muts_group2, muts_group3], ["CH-", "0.25%-2%", "2-10%", "≥10%"])
plot_psa_response_bar_grid(fig, gs, [ch_neg, muts_group1, muts_group2, muts_group3], ["CH-", "0.25%-2%", "2-10%", "≥10%"])
fig.savefig(os.path.join(dir_figures, "supp", "SUPP_PSA_response_vs_CH_4bins.png"))
fig.savefig(os.path.join(dir_figures, "supp", "SUPP_PSA_response_vs_CH_4bins.pdf"), transparent=True)


