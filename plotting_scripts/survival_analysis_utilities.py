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
from lifelines import CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import median_survival_times
from lifelines.statistics import multivariate_logrank_test

project_dir = os.environ.get("project_dir")
baseline_ch = pd.read_csv(f"{project_dir}/CH_baseline.csv")
clinical_data = pd.read_csv(f"{project_dir}/clinical_data/TheraP supplementary tables - Clinical data (10-Dec-2023).tsv", sep = "\t", skiprows = 1)
sample_info = pd.read_csv(f"{project_dir}/resources/sample_info.tsv", sep = "\t")
baseline_therap = sample_info[(sample_info["Cohort"] == "TheraP") & (sample_info["Timepoint"] == "Baseline")]
PATH_sample_information=f"{project_dir}/resources/sample_info.tsv"

color_dict={"LuPSMA": "#8d75bd", "Cabazitaxel": "#7faf9d"}

utilities=f"{project_dir}/plotting_scripts/utilities.py"
with open(utilities, 'r') as file:
    script_code = file.read()

exec(script_code)

# === SET UP OS DATA FRAME ===
survival_data = clinical_data[["Patient", "Overall survival status", "Overall survival months", "First progression status", "First progression months"]]
survival_data.columns = ["Patient_id", "Death", "OS from cfDNA collection (mo)", "Progression", "PFS (mo)"]
all_pts=survival_data.merge(sample_info[['Patient_id', 'Arm']].drop_duplicates())

# === ANNOTATE CH STATUS ===
baseline_ch=harmonize_vaf_columns(baseline_ch, timepoint="Baseline")
del baseline_ch["VAF_n"]
baseline_ch=baseline_ch.rename(columns={"VAF%": "VAF_n"})
mut_status = annotate_mutation_status_lu(baseline_ch, PATH_sample_information, annotate_what="CHIP", timepoint="Baseline", annotate_gene=False)
mut_status = mut_status.merge(sample_info[['Patient_id', 'Arm']].drop_duplicates())

# === GROUP PATIENTS BY MAX CH VAF ===
max_ch_df = baseline_ch.groupby("Patient_id")["VAF_n"].max().reset_index()

# Define mutation-based groups
ch_neg = mut_status[mut_status["CHIP status"] == "Negative"][["Patient_id", "Arm"]].merge(survival_data).assign(group="CH-")
ch_pos = mut_status[mut_status["CHIP status"] == "Positive"][["Patient_id", "Arm"]].merge(survival_data).assign(group="CH+")

muts_group1 = max_ch_df[(max_ch_df["VAF_n"] >= 0.25) & (max_ch_df["VAF_n"] < 2)]
muts_group2 = max_ch_df[(max_ch_df["VAF_n"] >= 2) & (max_ch_df["VAF_n"] < 10)]
muts_group3 = max_ch_df[max_ch_df["VAF_n"] >= 10]

# Merge groups with sample information
muts_group1 = muts_group1.merge(sample_info[['Patient_id', 'Arm']].drop_duplicates()).merge(survival_data).assign(group="0.25-2")
muts_group2 = muts_group2.merge(sample_info[['Patient_id', 'Arm']].drop_duplicates()).merge(survival_data).assign(group="2-10")
muts_group3 = muts_group3.merge(sample_info[['Patient_id', 'Arm']].drop_duplicates()).merge(survival_data).assign(group=">10")

# Group 4: CH-negative patients + low-VAF patients
muts_group4 = pd.concat([ch_neg, muts_group1]).merge(survival_data).assign(group="CH < 2%")

# And these are the gene groups
dta_pts=baseline_ch[baseline_ch["Gene"].isin(["DNMT3A", "TET2", "ASXL1"])][["Patient_id", "Arm"]].drop_duplicates().merge(survival_data).assign(group="DTA CH")
ddr_pts=baseline_ch[baseline_ch["Gene"].isin(["TP53", "PPM1D", "BRCA1", "BRCA2", "ATM", "CHEK2"])][["Patient_id", "Arm"]].drop_duplicates().merge(survival_data).assign(group="")
ddr_pts["group"]="DDR CH"

no_ddr_patients=baseline_ch[~baseline_ch["Gene"].isin(["TP53", "PPM1D", "BRCA1", "BRCA2", "ATM", "CHEK2"])][["Patient_id", "Arm"]].drop_duplicates().merge(survival_data).assign(group="")
no_ddr_patients=pd.concat([no_ddr_patients, ch_neg]).assign(group="No DDR CH")

# === PREPARE DATA FOR KAPLAN-MEIER PLOTS ===
groups = {
    "All pts": all_pts,
    "CH-": ch_neg,
    "CH+": ch_pos,
    "CH 0.25-2%": muts_group1,
    "CH 2-10%": muts_group2,
    "CH > 10%": muts_group3,
    "CH < 2%": muts_group4,
    "DDR CH": ddr_pts,
    "No DDR CH": no_ddr_patients,
}

curve_colors = {
    "All pts": "black",
    "CH-": "royalblue",  # Soft sky blue
    "CH+": "r",  # Medium blue
    "CH neg-2%": "#32cd32",
    "CH < 2%": "#3bce3b",  # Light blue
    "CH 2-10%": "#3478C0",  # Deeper blue
    "CH > 10%": "#002E6E",  # Dark navy blue
    "CH- and low VAF": "#9AC7E8",  # Muted blue, distinct from CH negative
    "DDR CH": "cornflowerblue",
}

km_data = {
    "LuPSMA": {key: df[df["Arm"] == "LuPSMA"] for key, df in groups.items()},
    "Cabazitaxel": {key: df[df["Arm"] == "Cabazitaxel"] for key, df in groups.items()}
}

############################ SET UP FIGURE
def plot_figure2_forest_plots_survival(gs, km_data, event, duration, groups, curve_colors=curve_colors):
    """
    Sets up the GS layout and plots the forest plot after running COX PH.
    gs should have this layout: 2, 4, width_ratios=[0.3,1,0.15,0.3], height_ratios=[1, 0.3]
    curve_colors defined in survival_analysis_utilities.py
    """
    ax_forest = plt.subplot(gs[0, 1])
    ax_annots_or = plt.subplot(gs[0, 2], sharey=ax_forest)
    ax_annots_p = plt.subplot(gs[0, 3], sharey=ax_forest)
    ax_x_annots=plt.subplot(gs[1,1], sharex=ax_forest)
    
    dict_list=[]
    for group in groups:
        surv_df_combined=pd.concat([km_data["LuPSMA"][group], km_data["Cabazitaxel"][group]])
        surv_df_combined["Arm coded"]=surv_df_combined["Arm"].map({"Cabazitaxel": 0, "LuPSMA": 1})
        km_dict=run_cox_proportional_hazards(surv_df_combined, duration, event, "Arm coded", ax = None)
        km_dict["Label"]=group
        km_dict["color"] = curve_colors.get(group, "black")
        
        dict_list.append(km_dict)
        
    plot_forest(*dict_list, ax=ax_forest)
    
    ax_forest.set_xlim([-0.1, 1.85])
    ax_forest.set_xticks([0, 0.5, 1, 1.5])
    ax_forest.set_xticklabels(["0", "0.5", "1", "1.5"])
    
    ax_annots_or.tick_params("both", bottom=False, labelbottom=False, left=False, labelleft=False)
    ax_annots_p.tick_params("both", bottom=False, labelbottom=False, left=False, labelleft=False)
    ax_x_annots.axis("off")
    ax_annots_or.axis("off")
    ax_annots_p.axis("off")
    ax_annots_or.set_title("HR (95% CI)", fontsize=8)
    ax_annots_p.set_title("p", fontsize=8)
    
    for i, pfs_dict in enumerate(dict_list):
        hr=format_p_value(float(pfs_dict["HR"]))
        lower=format_p_value(float(pfs_dict["CI_lower"]))
        upper=format_p_value(float(pfs_dict["CI_upper"]))
        p_val=format_p_value(float(pfs_dict["p"]))
        
        ax_annots_or.text(0, i, f"{hr}({lower}-{upper})", ha="left", va="center", fontsize=6)
        ax_annots_p.text(0, i, p_val, ha="left", va="center", fontsize=6)
    
    return(ax_forest, ax_annots_or, ax_annots_p, ax_x_annots)

def plot_survival_curves(km_data, metric, ylabel, filename, dir_figures, curve_colors, groups=None, run_logrank_test=True, suptitle=None, ci_show=False, annotate_HR=False, gs=None, fig=None, show_legend=True):
    """
    Plots Kaplan-Meier survival curves for specified groups and performs a log-rank test.
    Parameters:
    - km_data: dict containing survival data for each treatment arm and group.
    - metric: str, the survival time column name (e.g., "PFS (mo)" or "OS (mo)").
    - ylabel: str, label for y-axis.
    - filename: str, name of the output figure file.
    - dir_figures: str, directory where the figure should be saved.
    - curve_colors: dict, mapping of group names to colors.
    - groups: list of str, groups to plot (if None, defaults to all available groups in the dataset).
    - run_logrank_test: bool, whether to perform a log-rank test.
    """
    if gs is None:
        fig = plt.figure(figsize=(8, 4))
        gs = gridspec.GridSpec(1, 2, figure=fig)
    
    #
    for i, arm in enumerate(["LuPSMA", "Cabazitaxel"]):
        sharex = fig.axes[0] if i == 1 else None
        ax=fig.add_subplot(gs[i])
                
        ax.set_title(arm)
        kmf_list = []
        median_survivals = {}
        durations_list = []
        events_list = []
        groups_list = []
        curve_list=[]
        #
        for group in groups:
            curve = km_data[arm][group]
            kmf = KaplanMeierFitter()
            kmf.fit(durations=curve[metric], event_observed=curve["Progression" if metric == "PFS (mo)" else "Death"], label=group)
            kmf.plot_survival_function(ax=ax, ci_show=ci_show, show_censors=True, linewidth=1.2,
                                       censor_styles={'marker': '|', 'ms': 5, 'markerfacecolor': curve_colors.get(group, 'black'), 'mew': 0.8},
                                       color=curve_colors.get(group, 'black'))
            kmf_list.append(kmf)
            # Store median survival
            median_survivals[group] = kmf.median_survival_time_
            # Collect data for log-rank test
            durations_list.extend(curve[metric])
            events_list.extend(curve["Progression" if metric == "PFS (mo)" else "Death"])
            groups_list.extend([group] * len(curve[metric]))
            curve_list.append(curve.assign(group=group))
            
            if show_legend:
                if i==0:
                    ax.legend(frameon=False)
                else:
                    ax.legend().set_visible(False)
            else:
                ax.legend().set_visible(False)
            
            # Determine x ticks.
            if metric=="PFS (mo)":
                ax.set_xlim((-2, 35))
                ax.set_xticks([0, 10, 20, 30])
                ax.set_xticklabels(["0", "10", "20", "30"])
            else:
                ax.set_xlim((-2, 45))
                ax.set_xticks([0, 10, 20, 30, 40])
                ax.set_xticklabels(["0", "10", "20", "30", "40"])
        #
        add_at_risk_counts(*kmf_list, ax=ax, rows_to_show=["At risk"])
        #
        # Perform log-rank test (only if 2 or more groups are present)
        if run_logrank_test:
            results = multivariate_logrank_test(durations_list, groups_list, events_list)
            p_value = results.p_value
            #
            text_str = f"logrank p={p_value:.3f}\n"
            for group, median in median_survivals.items():
                text_str += f"{group}: {median:.1f} mo\n"
            y_pos_text=(0.55 if len(groups)==2 else 0.25)
            ax.text(0.95, y_pos_text, text_str, transform=ax.transAxes, ha='right', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        #
        # Run univariate Cox PH and annotate the hazard ratio on the plot.
        concatted_curves=pd.concat(curve_list, ignore_index=True)
        if annotate_HR:
            cph = CoxPHFitter()
            cox_dict=run_cox_proportional_hazards(concatted_curves, duration_col=metric, event_col="Progression" if metric == "PFS (mo)" else "Death", stratify_by="group")            
            hr_text = f"HR={cox_dict['HR']:.2f} (95% CI: {cox_dict['CI_lower']:.2f}-{cox_dict['CI_upper']:.2f})\np={cox_dict['p']}"
            ax.text(0.95, 0.15, hr_text, transform=ax.transAxes, ha='right', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        # Aesthetic settings
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim((-0.02, 1.02))
        ax.set_yticks([0, .2, .4, .6, .8, 1])
        ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])
        ax.set_xlabel("Time since randomization (mo)")
        
        if i==0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")
    #
    if suptitle is not None:
        fig.suptitle(suptitle)
    #
    if gs is None:
        gs.tight_layout(fig)
        fig.savefig(os.path.join(dir_figures, filename))
    else: 
        return(gs)

def format_p_value(p, sig_digits=2):
    if p == 0:
        return "0.00"  # Edge case
    exponent = math.floor(math.log10(abs(p)))  # Get exponent
    decimal_places = max(0, sig_digits - exponent - 1)  # Ensure at least 0 decimals
    return f"{p:.{decimal_places}f}"

def return_cox_analysis_within_arm_two_groups(km_data, metric="PFS (mo)", arm="LuPSMA", groups=["CH+", "CH-"]):
    """
    Runs Cox PH for within a single arm.
    """
    cox_dict={}
    arm_data_list=[]
    median_surv_dict={}
    for group in groups:
        arm_data=km_data[arm][group]
        if group=="All pts":
            arm_data["group"]="All pts"
        arm_data_list.append(arm_data)
        
        kmf = KaplanMeierFitter()
        kmf.fit(durations=arm_data[metric], event_observed=arm_data["Progression" if metric == "PFS (mo)" else "Death"], label=group)
        median_surv_dict[group] = kmf.median_survival_time_
        
    # Run Cox PH
    concatted_arm_data=pd.concat(arm_data_list, ignore_index=True)[["group", metric, "Progression" if metric=="PFS (mo)" else "Death"]]
    if groups==["CH+", "CH-"]:
        concatted_arm_data["group"] = concatted_arm_data["group"].map({"CH-": 0, "CH+": 1})
    else:
        concatted_arm_data["group"] = concatted_arm_data["group"].map({'All pts':0, 'CH < 2%':1, '2-10':2, '>10':3})
    
    cph = CoxPHFitter()
    # surv_df[stratify_by]=surv_df[stratify_by].astype(bool)
    # surv_df = pd.get_dummies(surv_df, columns=[stratify_by], drop_first=True)
    cph.fit(concatted_arm_data, duration_col=metric, event_col="Progression" if metric=="PFS (mo)" else "Death")
    
    hazard_ratio = cph.summary['exp(coef)'].values[0]
    ci_lower = cph.summary['exp(coef) lower 95%'].values[0]
    ci_upper = cph.summary['exp(coef) upper 95%'].values[0]
    p_value = cph.summary['p'].values[0]
    cox_dict={"HR": hazard_ratio, "CI_upper": ci_upper, "CI_lower": ci_lower, "p": p_value}
    
    return(cox_dict, median_surv_dict)

def plot_forest(*dicts, ax, xmin=None, xmax=None):
    """
    Plots a forest plot for the given dictionaries.
    Each dictionary should contain the following keys:
    - 'Label': Label for the data (e.g., gene name or category).
    - 'OR' or 'HR': Effect measure (Odds Ratio or Hazard Ratio).
    - 'CI lower': Lower bound of the confidence interval.
    - 'CI upper': Upper bound of the confidence interval.
    - Optional: 'color' for customizing error bar colors.
    """
    labels = []
    values = []
    CI_lowers = []
    CI_uppers = []
    colors = []
    
    for d in dicts:
        labels.append(d['Label'])
        measure_key = 'OR' if 'OR' in d else 'HR'  # Detect whether OR or HR is used
        values.append(d[measure_key])
        CI_lowers.append(d['CI_lower'])
        CI_uppers.append(d['CI_upper'])
        colors.append(d.get('color', 'black'))  # Default to black if no color is provided
    
    # Plotting
    y_positions = np.arange(len(labels))
    for i in range(len(labels)):
        ax.errorbar(values[i], y_positions[i], xerr=[[values[i] - CI_lowers[i]], 
                                                      [CI_uppers[i] - values[i]]],
                    fmt='o', color=colors[i], capsize=1.5, capthick=0.5, elinewidth=0.5, ms=3, zorder=99999)
    
    # Add vertical line for neutral effect (OR=1 or HR=1)
    ax.axvline(1, color='black', linestyle='--', label='Neutral (1)', linewidth=0.5, zorder=5)
    
    # Set y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(measure_key)
    
    if xmin is None and xmax is None:
        ax.set_xlim((-5, 32))
        ax.set_xticks([-5, 1, 10, 20, 30])
        ax.set_xticklabels(["-5", "1", "10", "20", "30"])
    else:
        ax.set_xlim((xmin, xmax))
        print("Label x-axis yourself pls.")
    
    ax.set_ylim((-1, len(labels)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax

# USE EXAMPLES     
# Without confidence intervals
# plot_survival_curves(km_data, "PFS (mo)", "Progression-Free Survival (%)", "PFS_ch_negative_vs_ch_positive.png", dir_figures, curve_colors, groups=["CH positive", "CH negative"])
# plot_survival_curves(km_data, "OS from cfDNA collection (mo)", "OS (%)", "OS_ch_negative_vs_ch_positive.png", dir_figures, curve_colors, groups=["CH positive", "CH negative"])

# plot_survival_curves(km_data, "PFS (mo)", "Progression-Free Survival (%)", "PFS_ch_3bins.png", dir_figures, curve_colors, groups=["CH < 2%", "CH 2-10%", "CH > 10%"], suptitle="CH<2% category includes CH- patients, too.")
# plot_survival_curves(km_data, "OS from cfDNA collection (mo)", "OS (%)", "OS_ch_3bins.png", dir_figures, curve_colors, groups=["CH < 2%", "CH 2-10%", "CH > 10%"], suptitle="CH<2% category includes CH- patients, too.")

# plot_survival_curves(km_data, "PFS (mo)", "Progression-Free Survival (%)", "PFS_ch_gene_groups.png", dir_figures, curve_colors, groups=["DTA CH", "DDR CH", "Splicing CH"])
# plot_survival_curves(km_data, "OS from cfDNA collection (mo)", "OS (%)", "OS_ch_gene_groups.png", dir_figures, curve_colors, groups=["DTA CH", "DDR CH", "Splicing CH"])

# # With confidence intervals
# plot_survival_curves(km_data, "PFS (mo)", "Progression-Free Survival (%)", "PFS_ch_negative_vs_ch_positive_withCI.png", dir_figures, curve_colors, groups=["CH positive", "CH negative"], ci_show=True, annotate_HR=True)
# plot_survival_curves(km_data, "OS from cfDNA collection (mo)", "OS (%)", "OS_ch_negative_vs_ch_positive_withCI.png", dir_figures, curve_colors, groups=["CH positive", "CH negative"], ci_show=True, annotate_HR=True)

# plot_survival_curves(km_data, "PFS (mo)", "Progression-Free Survival (%)", "PFS_ch_3bins_withCI.png", dir_figures, curve_colors, groups=["CH < 2%", "CH 2-10%", "CH > 10%"], suptitle="CH<2% category includes CH- patients, too.", ci_show=True, annotate_HR=True)
# plot_survival_curves(km_data, "OS from cfDNA collection (mo)", "OS (%)", "OS_ch_3bins_withCI.png", dir_figures, curve_colors, groups=["CH < 2%", "CH 2-10%", "CH > 10%"], suptitle="CH<2% category includes CH- patients, too.", ci_show=True, annotate_HR=True)

# plot_survival_curves(km_data, "PFS (mo)", "Progression-Free Survival (%)", "PFS_ch_gene_groups_withCI.png", dir_figures, curve_colors, groups=["DTA CH", "DDR CH", "Splicing CH"], ci_show=True, annotate_HR=True)
# plot_survival_curves(km_data, "OS from cfDNA collection (mo)", "OS (%)", "OS_ch_gene_groups_withCI.png", dir_figures, curve_colors, groups=["DTA CH", "DDR CH", "Splicing CH"], ci_show=True, annotate_HR=True)