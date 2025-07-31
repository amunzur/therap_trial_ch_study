

"""
Bins number of cycles of treatment received into 3 bins.
Constructs a logistic regression correlating that with number of new CH mutations.
"""

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
import matplotlib.patches as patches
from matplotlib.path import Path
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial

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


def run_neg_binomial(df, x_name, y_name):
    """
    Fits negative binomial model. 
    """
    # Data seems to be overdispersed, fit negative binomial model
    X = sm.add_constant(df[x_name])  # adds intercept term
    y = df[y_name]    
    
    # Fit Negative Binomial model
    nb_model = sm.GLM(y, X, family=NegativeBinomial()) # Neg binomial is ideal for overdispersed discrete count data.
    nb_result = nb_model.fit()  
    
    print(nb_result.summary())
    return(nb_result)

def run_gamma_regression_glm(df, x_name, y_name):
    """
    Runs gamma regression. 
    """
    X = sm.add_constant(df[x_name])
    y = df[y_name]
    
    mask = y > 0
    model = sm.GLM(y[mask], X[mask], family=Gamma(link=sm.genmod.families.links.log()))
    result = model.fit()
    
    print(result.summary()) 
    return(result)
    
def fit_normal_glm(df, x_name, y_name):
    """
    Akin to Gaussian LM fit. Can handle negative values in response variable.
    """
    model = smf.glm(f"{y_name} ~ {x_name}", data=df, family=sm.families.Gaussian()).fit()
    print(model.summary())
    
    return(model)
    
    
    
    plot_glm_predictions(lu_normalglm_model_vaf, max_vaf_diff_df_lu, x_col="Cycle", y_col="vaf_diff", y_label="Difference in max CH VAF%", title="LuPSMA", output_path=lu_normal_glm_path)


def plot_glm_predictions(result, plot_data, x_col, y_col, y_label, title=None, output_path=None, formula_based=True):
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
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.scatterplot(x=f'{x_col}_jittered', y=y_col, data=plot_data,alpha=0.6, label='Observed', ax=ax)
    ax.plot(X_plot, y_pred, color='red', marker='o', linestyle='-', label='Fitted GLM', markersize=4)
    
    # Annotate number of observations at each cycle
    cycle_counts = plot_data[x_col].value_counts().sort_index()
    
    # Find max y value from observed and predicted to determine y-limit
    max_y_observed = plot_data[y_col].max()
    max_y_pred = y_pred.max()
    max_y = max(max_y_observed, max_y_pred)
    
    min_y_observed = plot_data[y_col].min()
    min_y_pred = y_pred.min()
    min_y = min(min_y_observed, min_y_pred)-5
    
    # Set space for annotations — add 10% of max_y or at least 5 units
    y_buffer = max(5, max_y * 0.1)
    y_lim_upper = max_y + y_buffer
    
    for cycle, count in cycle_counts.items():
        ax.text(x=cycle, y=90, s=f'n={count}',ha='center', va='bottom', fontsize=8, color='black'
        )
    
    # Formatting
    ax.set_ylim(min_y, y_lim_upper)
    ax.set_xlabel('Number of treatment cycles received')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.2))
    # fig.tight_layout()
    
    # Add stats from model fit
    try:
        coef = round(result.params[1])  # usually index 1 for the predictor after intercept
        pval = round(result.pvalues[1], 5)
    except Exception as e:
        coef = None
        pval = None
    
    # Format stats text
    if coef is not None and pval is not None:
        stats_text = f'Coef: {coef:.3f}\nP-value: {pval:.3e}'
        # Put stats text in upper right corner with a box
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7),
                fontsize=8)
    
    # Save if requested
    if output_path:
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    
    return fig, ax

def plot_neg_binomial_fit(df, x_col, y_col, model, arm_label, color, path_output):
    """
    Plots scatter + predicted fit line from a negative binomial model.
    
    Parameters:
        df (DataFrame): Input data.
        x_col (str): Name of the independent variable column.
        y_col (str): Name of the dependent variable column.
        model (GLMResultsWrapper): Fitted negative binomial model.
        arm_label (str): Label to show in the legend.
        color (str): Color for points and line.
        ax (matplotlib.axes.Axes): Axes object to plot on.
    """
    
    plot_df = df.copy()
    np.random.seed(42)
    plot_df[x_col + "_jittered"] = plot_df[x_col] + np.random.uniform(-0.1, 0.1, size=len(plot_df))
    
    # Scatter plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=plot_df, x=x_col + "_jittered", y=y_col, color=color, alpha=0.6, s=15, ax=ax)
    
    # Prediction line
    x_vals = np.sort(df[x_col].unique())
    x_pred = pd.DataFrame({x_col: x_vals})
    x_pred = sm.add_constant(x_pred)
    y_pred = model.predict(x_pred)
        
    ax.plot(x_vals, y_pred, color=color, linestyle='-', linewidth=2)
    
    # Labels
    ax.set_xlabel("Number of treatment cycles")
    ax.set_ylabel("Number of new CH mutations")
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_yticklabels(['0', '5', '10', '15', '20'])
    ax.set_title(f"CH mutations vs. treatment cycles, neg binomial model\n{arm_label}")
    ax.spines[["top", "right"]].set_visible(False)
    
    # Add some stats to the plot from the model fit:
    coef = round(model.params[x_col], 3)
    pval = round(model.pvalues[x_col], 3)
    
    # Format as text
    textstr = (
        f"Slope = {coef:.2f}\n"
        f"p = {pval:.3g}\n"
    )
    
    # Add to your plot
    ax.text(
        0.05, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8)
    )
    
    fig.savefig(path_output)


project_dir = os.environ.get("project_dir")
path_sample_info = f"{project_dir}/resources/sample_info.tsv"
path_cycles="/groups/wyattgrp/users/amunzur/lu_chip/resources/clinical_tables/TheraP supplementary tables - Treatment dates (22-Feb-2025).csv"
baseline_ch_path=f"{project_dir}/CH_baseline.csv"
progression_ch_path=f"{project_dir}/CH_progression.csv"
dir_figures=f"{project_dir}/figures/main"

path_utilities=f"{project_dir}/plotting_scripts/utilities.py"
with open(path_utilities, 'r') as file:
    script_code = file.read()

exec(script_code)

# LOAD DATASETS
sample_info = pd.read_csv(path_sample_info, sep= "\t")
pts_with_progression_samples = sample_info[sample_info["Timepoint"] == "FirstProgression"][["Patient_id", "Arm"]].drop_duplicates()
baseline_ch = pd.read_csv(baseline_ch_path)
baseline_ch_subset = baseline_ch[baseline_ch["Patient_id"].isin(pts_with_progression_samples)]
progression_ch = pd.read_csv(progression_ch_path)
progression_ch_only=progression_ch[progression_ch["Independently detected at baseline"]==False].reset_index(drop=True)

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

######################
# Model 1. Do patients acquire more new CH mutations with increasing number of treatment cycles?
nmuts_df=nmuts_df.merge(ncycles_df)

nmuts_lu_df=nmuts_df[nmuts_df["Arm"]=="LuPSMA"]
nmuts_caba_df=nmuts_df[nmuts_df["Arm"]=="Cabazitaxel"]

lu_negbinom_model_nmuts=run_neg_binomial(nmuts_lu_df, "Cycle", "nmuts")
caba_negbinom_model_nmuts=run_neg_binomial(nmuts_caba_df, "Cycle", "nmuts")

# Plotting
neg_binom_lu_path="/groups/wyattgrp/users/amunzur/lu_chip/results/figures/progression/lu_neg_binom_nmuts_and_cycles.png"
plot_neg_binomial_fit(nmuts_lu_df, "Cycle", "nmuts", lu_negbinom_model_nmuts, "LuPSMA", "dodgerblue", neg_binom_lu_path)

neg_binom_caba_path="/groups/wyattgrp/users/amunzur/lu_chip/results/figures/progression/caba_neg_binom_nmuts_and_cycles.png"
plot_neg_binomial_fit(nmuts_caba_df, "Cycle", "nmuts", caba_negbinom_model_nmuts, "Cabazitaxel", "darkorange", neg_binom_caba_path)


######################
# Model 2. Do patients acquire higher VAF treatment-emergent CH mutations with increasing number of treatment cycles?
# Max VAF and number of cycles of treatment

# Max CH VAF at progression and number of cycles of treatment for lu?
max_vaf_df=progression_ch.groupby(["Patient_id", "Arm"])["VAF%"].max().reset_index()
# max_vaf_df=max_vaf_df.merge(pts_with_progression_samples, how="outer").fillna(0)
max_vaf_df=max_vaf_df.merge(ncycles_df)

max_vaf_df_lu=max_vaf_df[max_vaf_df["Arm"]=="LuPSMA"]
max_vaf_df_caba=max_vaf_df[max_vaf_df["Arm"]=="Cabazitaxel"]

lu_gammareg_model_vaf=run_gamma_regression_glm(max_vaf_df_lu, "Cycle", "VAF%")
caba_gammareg_model_vaf=run_gamma_regression_glm(max_vaf_df_caba, "Cycle", "VAF%")

# Visualize
lu_gamma_path="/groups/wyattgrp/users/amunzur/lu_chip/results/figures/progression/lu_gamma_regression.png"
plot_glm_predictions(lu_gammareg_model_vaf, max_vaf_df_lu, x_col="Cycle", y_col="VAF%", y_label="Max treatment-emergent CH VAF%", title="LuPSMA", output_path=lu_gamma_path)

caba_gamma_path="/groups/wyattgrp/users/amunzur/lu_chip/results/figures/progression/caba_gamma_regression.png"
plot_glm_predictions(caba_gammareg_model_vaf, max_vaf_df_caba, x_col="Cycle", y_col="VAF%", y_label="Max treatment-emergent CH VAF%", title="Cabazitaxel", output_path=caba_gamma_path)

######################
# Model 3. This time correlating the number of cycles of treatment with change in max vaf from baseline to progression.
max_vaf_df_base=baseline_ch.groupby(["Patient_id", "Arm"])["VAF%"].max().reset_index().merge(pts_with_progression_samples, how="right").fillna(0.25).rename(columns={"VAF%": "Max VAF baseline"})
max_vaf_df_prog=progression_ch.groupby(["Patient_id", "Arm"])["VAF%"].max().reset_index().merge(pts_with_progression_samples, how="right").fillna(0.25).rename(columns={"VAF%": "Max VAF progression"})

max_vaf_diff_df=max_vaf_df_base.merge(max_vaf_df_prog)
max_vaf_diff_df["vaf_diff"]=max_vaf_diff_df["Max VAF progression"]-max_vaf_diff_df["Max VAF baseline"]
max_vaf_diff_df=max_vaf_diff_df.merge(ncycles_df)

max_vaf_diff_df_lu=max_vaf_diff_df[max_vaf_diff_df["Arm"]=="LuPSMA"]
max_vaf_diff_df_caba=max_vaf_diff_df[max_vaf_diff_df["Arm"]=="Cabazitaxel"]

lu_normalglm_model_vaf=fit_normal_glm(max_vaf_diff_df_lu, "Cycle", "vaf_diff")
caba_normalglm_model_vaf=fit_normal_glm(max_vaf_diff_df_caba, "Cycle", "vaf_diff")

# Plotting findings
lu_normal_glm_path="/groups/wyattgrp/users/amunzur/lu_chip/results/figures/progression/lu_normal_glm_vaf_change.png"
plot_glm_predictions(lu_normalglm_model_vaf, max_vaf_diff_df_lu, x_col="Cycle", y_col="vaf_diff", y_label="Difference in max CH VAF%", title="LuPSMA", output_path=lu_normal_glm_path, formula_based=False)

caba_normal_glm_path="/groups/wyattgrp/users/amunzur/lu_chip/results/figures/progression/caba_normal_glm_vaf_change.png"
plot_glm_predictions(caba_normalglm_model_vaf, max_vaf_diff_df_caba, x_col="Cycle", y_col="vaf_diff", y_label="Difference in max CH VAF%", title="Cabazitaxel", output_path=caba_normal_glm_path, formula_based=False)






X = sm.add_constant(max_vaf_diff_df_lu['Cycle'])
y = max_vaf_diff_df_lu['vaf_diff']

model = sm.GLM(y, X, family=Gamma(link=sm.genmod.families.links.log()))
result = model.fit()
print(result.summary())





# Prepare data
X_plot = np.arange(nmuts_lu_df['Cycle'].min(), nmuts_lu_df['Cycle'].max() + 1)
X_plot_const = sm.add_constant(X_plot)

# Predict expected mean VAF% from model (on original scale)
y_pred_log = result.predict(X_plot_const)  # predictions on link scale (log), but predict func returns predictions already in whatever scale was used, so non-log in our case

fig, ax = plt.subplots(figsize=(4, 3))

plot_data = max_vaf_df_lu[mask].copy()
np.random.seed(1)
plot_data['Cycle_jittered'] = plot_data['Cycle'] + np.random.normal(0, 0.1, size=plot_data.shape[0])

sns.scatterplot(x='Cycle_jittered', y='VAF%', data=plot_data, alpha=0.6, label='Observed', ax=ax) # plot the observed data
ax.plot(X_plot, y_pred_log, color='red', marker='o', linestyle='-', label='Fitted Gamma GLM', markersize=4) # Plot prediction data from GLM

# Annotate number of cycles in plot
cycle_counts = plot_data['Cycle'].value_counts().sort_index()

# Annotate each cycle with 'n='
for cycle, count in cycle_counts.items():
    ax.text(
        x=cycle,
        y=90,  # just above the highest point
        s=f'n={count}',
        ha='center',
        va='bottom',
        fontsize=8,
        color='black'
    )



# Labeling
ax.set_ylim((-5, 100))
ax.set_xlabel('Number of treatment cycles received')
ax.set_ylabel('Maximum CH VAF at progression(%)')
ax.set_title('LuPSMA')
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.2))
fig.tight_layout()

fig.savefig("/groups/wyattgrp/users/amunzur/lu_chip/results/figures/progression/gamma_reg.png")

##############################
# Another try, correlating the change in max VAF from baseline to progression, then adding the number of cycles of treatment into the equation
max_vaf_df_base=baseline_ch.groupby(["Patient_id", "Arm"])["VAF%"].max().reset_index().merge(pts_with_progression_samples, how="right").fillna(0.25).rename(columns={"VAF%": "Max VAF baseline"})
max_vaf_df_prog=progression_ch.groupby(["Patient_id", "Arm"])["VAF%"].max().reset_index().merge(pts_with_progression_samples, how="right").fillna(0.25).rename(columns={"VAF%": "Max VAF progression"})

max_vaf_diff_df=max_vaf_df_base.merge(max_vaf_df_prog)
max_vaf_diff_df["vaf_diff"]=max_vaf_diff_df["Max VAF progression"]-max_vaf_diff_df["Max VAF baseline"]
max_vaf_diff_df=max_vaf_diff_df.merge(ncycles_df)

max_vaf_diff_df_lu=max_vaf_diff_df[max_vaf_diff_df["Arm"]=="LuPSMA"]
max_vaf_diff_df_caba=max_vaf_diff_df[max_vaf_diff_df["Arm"]=="Cabazitaxel"]





X = sm.add_constant(max_vaf_diff_df_lu['Cycle'])
y = max_vaf_diff_df_lu['vaf_diff']

model = sm.GLM(y, X, family=Gamma(link=sm.genmod.families.links.log()))
result = model.fit()
print(result.summary())



    

import statsmodels.api as sm
import statsmodels.formula.api as smf

# Example assumes a column 'Cycle' and 'vaf_diff'
model = smf.glm("vaf_diff ~ Cycle", data=max_vaf_diff_df_lu, family=sm.families.Gaussian()).fit()
print(model.summary())

X_plot = np.linspace(max_vaf_diff_df_lu['Cycle'].min(), max_vaf_diff_df_lu['Cycle'].max(), 100)
y_pred = model.predict(pd.DataFrame({'Cycle': X_plot}))

plt.figure(figsize=(6, 4))
sns.scatterplot(x='Cycle', y='vaf_diff', data=max_vaf_diff_df_lu, alpha=0.5)
plt.plot(X_plot, y_pred, color='red', label='Fitted GLM')
plt.legend()
plt.xlabel('Number of Treatment Cycles')
plt.ylabel('Change in VAF (%)')
plt.savefig("/groups/wyattgrp/users/amunzur/lu_chip/results/figures/progression/gamma_reg_vaf_diff.png")
