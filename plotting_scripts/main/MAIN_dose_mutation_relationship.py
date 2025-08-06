

"""
Bins number of cycles of treatment received into 3 bins.
Constructs a logistic regression correlating that with number of new CH mutations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats

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

def fit_normal_glm(df, x_name, y_name):
    """
    Akin to Gaussian LM fit. Can handle negative values in response variable.
    """
    model = smf.glm(f"{y_name} ~ {x_name}", data=df, family=sm.families.Gaussian()).fit()
    print(model.summary())
    
    return(model)

def plot_glm_predictions(result, plot_data, x_col, y_col, y_label, title=None, output_path=None, formula_based=True, fitted_glmcolor="black", dotcolor="black"):
    """
    Plots observed data and fitted GLM predictions over treatment cycles.
    
    Parameters:
        result (GLMResults): Fitted statsmodels GLM result object.
        plot_data (DataFrame): Data to plot (must include x_col and y_col).
        x_col (str): Column name for x-axis (e.g., treatment cycle).
        y_col (str): Column name for y-axis (e.g., VAF%).
        y_label (str): Label for the y-axis.
        title (str): Plot title.
        output_path (str or None): Path to save the figure. If None, figure is not saved.
    """
    
    # Prepare prediction range
    X_plot = np.arange(plot_data[x_col].min(), plot_data[x_col].max() + 1)
    
    if formula_based:
        # For formula-based models like normal basic GLM, just pass a DataFrame with x_col to predict
        X_pred = pd.DataFrame({x_col: X_plot})
    else:
        # For exog-based models like gamma regression, must add constant manually
        X_pred = pd.DataFrame({x_col: X_plot})
        X_pred = sm.add_constant(X_pred)
        
    # Predict from model
    y_pred = result.predict(X_pred)
    
    # Jitter x values slightly for visualization
    plot_data = plot_data.copy()
    np.random.seed(1)
    plot_data[f'{x_col}_jittered'] = plot_data[x_col] + np.random.normal(0, 0.1, size=plot_data.shape[0])
    
    # Plotting
    fig, ax = plt.subplots(figsize=(3.2, 1.8))
    sns.scatterplot(x=f'{x_col}_jittered', y=y_col, data=plot_data, label='Observed', ax=ax, size=5, color=dotcolor)
    ax.plot(X_plot, y_pred, color=fitted_glmcolor, marker='o', linestyle='-', label='Fitted GLM', markersize=3, linewidth=0.5)
    
    # Annotate number of observations at each cycle
    cycle_counts = plot_data[x_col].value_counts().sort_index()
    
    # Formatting
    ax.set_xlabel('Number of treatment cycles received')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    # ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.2))
    # fig.tight_layout()
    
    # Add stats from model fit
    try:
        coef = round(result.params[1])  # usually index 1 for the predictor after intercept
        pval = round(result.pvalues[1], 5)
    except Exception as e:
        coef = None
        pval = None
    
    # Format stats text
    # if coef is not None and pval is not None:
    #     stats_text = f'Coef: {coef:.3f}\nP-value: {pval:.3e}'
    #     # Put stats text in upper right corner with a box
    #     ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
    #             verticalalignment='top', horizontalalignment='right',
    #             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7),
    #             fontsize=8)
    
    if title=="Cabazitaxel":
        ax.set_xticks(range(3, 11))
        ax.set_xticklabels([str(i) for i in range(3, 11)])
    
    ax.set_ylim((-28, 30))
    ax.set_yticks([-20, -10, 0, 10, 20, 30])
    ax.set_yticklabels(["-20", "-10", "0", "10", "20", "30"])
    
    for cycle, count in cycle_counts.items():
        ax.text(x=cycle, y=ax.get_ylim()[1], s=f'n={count}',ha='center', va='bottom', fontsize=6, color='black'
        )
    
    # Horizontal line through y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    # Save if requested
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    
    return fig, ax

# LOAD DATASETS
project_dir = os.environ.get("project_dir")

baseline_ch_path=f"{project_dir}/CH_baseline.csv"
path_sample_info = f"{project_dir}/resources/sample_info.tsv"
progression_ch_path=f"{project_dir}/CH_progression.csv"
path_cycles="/groups/wyattgrp/users/amunzur/lu_chip/resources/clinical_tables/TheraP supplementary tables - Treatment dates (22-Feb-2025).csv"

sample_info = pd.read_csv(path_sample_info, sep= "\t")
pts_with_progression_samples = sample_info[sample_info["Timepoint"] == "FirstProgression"][["Patient_id", "Arm"]].drop_duplicates()
baseline_ch = pd.read_csv(baseline_ch_path)
baseline_ch_subset = baseline_ch[baseline_ch["Patient_id"].isin(pts_with_progression_samples)]
progression_ch = pd.read_csv(progression_ch_path)
progression_ch_only=progression_ch[progression_ch["Independently detected at baseline"]==False].reset_index(drop=True)

utilities=f"{project_dir}/plotting_scripts/utilities.py"
with open(utilities, 'r') as file:
    script_code = file.read()

exec(script_code)

baseline_ch=harmonize_vaf_columns(baseline_ch, timepoint="Baseline")
progression_ch=harmonize_vaf_columns(progression_ch, timepoint="FirstProgression")

# Generate one df of number of progression mutations, include the samples that developed 0 mutations
nmuts_df=progression_ch["Patient_id"].value_counts().reset_index().rename(columns={"index": "Patient_id", "Patient_id": "nmuts"})
nmuts_df=nmuts_df.merge(pts_with_progression_samples, how="outer").fillna(0)

# Generate number of cycles of treatment df
ncycles_df=pd.read_csv(path_cycles)
ncycles_df=ncycles_df.groupby(["Patient", "Treatment arm"])["Cycle"].max().reset_index()
ncycles_df.columns=["Patient_id", "Arm", "Cycle"]

# Bin number of cycles of treatment into discrete bins
bins=[1,3,5,7,9,11,13]
labels=["1-2", "3-4", "5-6", "7-8", "9-10", "11-12"]
ncycles_df['cycle_bin'] = pd.cut(ncycles_df['Cycle'], bins=bins, labels=labels, right=False)
ncycles_df['cycle_bin'] = pd.Categorical(ncycles_df['cycle_bin'], categories=labels, ordered=True)
ncycles_df['ncycles_bin_numeric'] = ncycles_df['cycle_bin'].cat.codes

####################################################
# Model 3. This time correlating the number of cycles of treatment with change in max vaf from baseline to progression.
# max_vaf_df_base=baseline_ch.groupby(["Patient_id", "Arm"])["VAF%"].max().reset_index().merge(pts_with_progression_samples, how="right").fillna(0.25).rename(columns={"VAF%": "Max VAF baseline"})
# max_vaf_df_prog=progression_ch.groupby(["Patient_id", "Arm"])["VAF%"].max().reset_index().merge(pts_with_progression_samples, how="right").fillna(0.25).rename(columns={"VAF%": "Max VAF progression"})

# max_vaf_diff_df=max_vaf_df_base.merge(max_vaf_df_prog)
# max_vaf_diff_df["vaf_diff"]=max_vaf_diff_df["Max VAF progression"]-max_vaf_diff_df["Max VAF baseline"]
# max_vaf_diff_df=max_vaf_diff_df.merge(ncycles_df)

# max_vaf_diff_df_lu=max_vaf_diff_df[max_vaf_diff_df["Arm"]=="LuPSMA"]
# max_vaf_diff_df_caba=max_vaf_diff_df[max_vaf_diff_df["Arm"]=="Cabazitaxel"]

median_vaf_df_base=baseline_ch.groupby(["Patient_id", "Arm"])["VAF%"].median().reset_index().merge(pts_with_progression_samples, how="right").fillna(0.25).rename(columns={"VAF%": "Median VAF baseline"})
median_vaf_df_prog=progression_ch.groupby(["Patient_id", "Arm"])["VAF%"].median().reset_index().merge(pts_with_progression_samples, how="right").fillna(0.25).rename(columns={"VAF%": "Median VAF progression"})

median_vaf_diff_df=median_vaf_df_base.merge(median_vaf_df_prog)
median_vaf_diff_df["vaf_diff"]=median_vaf_diff_df["Median VAF progression"]-median_vaf_diff_df["Median VAF baseline"]
median_vaf_diff_df=median_vaf_diff_df.merge(ncycles_df)

median_vaf_diff_df_lu=median_vaf_diff_df[median_vaf_diff_df["Arm"]=="LuPSMA"]
median_vaf_diff_df_caba=median_vaf_diff_df[median_vaf_diff_df["Arm"]=="Cabazitaxel"]

# max_vaf_diff_df_lu_no_outlier=max_vaf_diff_df_lu[max_vaf_diff_df_lu["Cycle"]>1]

# lu_normalglm_model_vaf=fit_normal_glm(max_vaf_diff_df_lu, "Cycle", "vaf_diff")
# caba_normalglm_model_vaf=fit_normal_glm(max_vaf_diff_df_caba, "Cycle", "vaf_diff")

lu_normalglm_model_vaf=fit_normal_glm(median_vaf_diff_df_lu, "Cycle", "vaf_diff")
caba_normalglm_model_vaf=fit_normal_glm(median_vaf_diff_df_caba, "Cycle", "vaf_diff")

# Plotting findings
arm_color_dict={"LuPSMA": "#6f1fff", "Cabazitaxel": "#3e3939"}
arm_color_dict_lighter={"LuPSMA": "#B38AFF", "Cabazitaxel": "#857A7A"}


lu_normal_glm_path="/groups/wyattgrp/users/amunzur/lu_chip/results/figures/progression/lu_normal_glm_vaf_change_median_vaf.pdf"
plot_glm_predictions(
    lu_normalglm_model_vaf, 
    median_vaf_diff_df_lu, 
    x_col="Cycle", 
    y_col="vaf_diff", 
    y_label="ΔMax CH VAF%", 
    title="LuPSMA", 
    output_path=lu_normal_glm_path, 
    formula_based=False, 
    fitted_glmcolor=arm_color_dict["LuPSMA"], 
    dotcolor=arm_color_dict_lighter["LuPSMA"])

caba_normal_glm_path="/groups/wyattgrp/users/amunzur/lu_chip/results/figures/progression/caba_normal_glm_vaf_change_median_vaf.pdf"
plot_glm_predictions(
    caba_normalglm_model_vaf, 
    median_vaf_diff_df_caba, 
    x_col="Cycle", 
    y_col="vaf_diff", 
    y_label="ΔMax CH VAF%", 
    title="Cabazitaxel", 
    output_path=caba_normal_glm_path, 
    formula_based=False,
    fitted_glmcolor=arm_color_dict["Cabazitaxel"], 
    dotcolor=arm_color_dict_lighter["Cabazitaxel"])





