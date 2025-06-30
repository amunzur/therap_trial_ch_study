import pandas as pd
import numpy as np
import os
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu
from scipy.stats import fisher_exact
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import ttest_ind
import math
from decimal import Decimal
from matplotlib.lines import Line2D

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

project_dir = os.environ.get("project_dir")

ch_color_dict={"CH-": "royalblue", "CH+": "r"}
path_therap_clinical_data=f"{project_dir}/clinical_data/TheraP supplementary tables - Clinical data (10-Dec-2023).tsv"
baseline_ch_path=f"{project_dir}/CH_baseline.csv"
path_source_make_kms=f"{project_dir}/plotting_scripts/survival_analysis_utilities.py"
dir_figures=f"{project_dir}/figures/main"

for path in [path_source_make_kms]:
    with open(path, 'r') as file:
        script_code = file.read()
    
    exec(script_code)

# LOAD DATASETS
baseline_ch = pd.read_csv(baseline_ch_path)

# PLOTTING
# Set up gridspec
fig = plt.figure(figsize=(8, 7))
gs_outer = gridspec.GridSpec(2, 1, height_ratios=[1, 0.5], hspace=0.03, wspace=0.15)

gs_row0 = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[1, 1], subplot_spec=gs_outer[0], wspace=0.2, hspace=0.1)

gs_row0_left = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 0.4], subplot_spec=gs_row0[0], wspace=0.15, hspace=1.5)
gs_row0_left_top_KM = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[1, 1], subplot_spec=gs_row0_left[0], wspace=0.15)
gs_row0_left_bottom_forest = gridspec.GridSpecFromSubplotSpec(2, 4, width_ratios=[0.3, 1, 0.15, 0.3], height_ratios=[1, 0.3], subplot_spec=gs_row0_left[1], wspace=0.7, hspace=0.25)

gs_row0_right = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 0.4], subplot_spec=gs_row0[1], wspace=0.15, hspace=1.5)
gs_row0_right_top_KM = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[1, 1], subplot_spec=gs_row0_right[0], wspace=0.15)
gs_row0_right_bottom_forest = gridspec.GridSpecFromSubplotSpec(2, 4, width_ratios=[0.3, 1, 0.15, 0.3], height_ratios=[1, 0.3], subplot_spec=gs_row0_right[1], wspace=0.7, hspace=0.25)

# Survival outcomes and CH
gs_row0_left_top_KM=plot_survival_curves(km_data, "PFS (mo)", "Progression-Free Survival (%)", None, dir_figures, curve_colors, groups=["CH-", "CH+"], gs=gs_row0_left_top_KM, fig=fig, run_logrank_test=True)
gs_row0_right_top_KM=plot_survival_curves(km_data, "OS from cfDNA collection (mo)", "OS (%)", None, dir_figures, curve_colors, groups=["CH-", "CH+"], gs=gs_row0_right_top_KM, fig=fig, run_logrank_test=True)

# Forest plots for PFS and OS
PFS_ax_forest, PFS_ax_annots_or, PFS_ax_annots_p, PFS_ax_x_annots=plot_figure2_forest_plots_survival(gs_row0_left_bottom_forest, km_data, event="Progression", duration="PFS (mo)", groups=["CH-", "CH+", "All pts"])
OS_ax_forest, OS_ax_annots_or, OS_ax_annots_p, OS_ax_x_annots=plot_figure2_forest_plots_survival(gs_row0_right_bottom_forest, km_data, event="Death", duration="OS from cfDNA collection (mo)", groups=["CH-", "CH+", "All pts"])

# This next section prints HRs, median survival etc to add the figures later on manually via Affinity.
lu_pfs_cox, lu_pfs_surv=return_cox_analysis_within_arm_two_groups(km_data, metric="PFS (mo)", arm="LuPSMA", groups=["CH+", "CH-"])
caba_pfs_cox, caba_pfs_surv=return_cox_analysis_within_arm_two_groups(km_data, metric="PFS (mo)", arm="Cabazitaxel", groups=["CH+", "CH-"])
lu_os_cox, lu_os_surv=return_cox_analysis_within_arm_two_groups(km_data, metric="OS from cfDNA collection (mo)", arm="LuPSMA", groups=["CH+", "CH-"])
caba_os_cox, caba_os_surv=return_cox_analysis_within_arm_two_groups(km_data, metric="OS from cfDNA collection (mo)", arm="Cabazitaxel", groups=["CH+", "CH-"])

# gs_outer.tight_layout(fig)
fig.savefig(f"{dir_figures}/KMs.png")
fig.savefig(f"{dir_figures}/KMs.pdf")