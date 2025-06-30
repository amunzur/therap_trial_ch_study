import pandas as pd
import numpy as np
import os
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def CH_presence_line_thresholds(muts_df, age_df, PATH_sample_information, ax, axforest, fontsize = 10):
    """
    Plos the fraction of the cohort that is CH positive in each age.
    """
    import scipy.stats as stats
    sample_info = pd.read_csv(PATH_sample_information, sep = "\t")
    sample_info = sample_info[sample_info["Timepoint"] == "Baseline"].reset_index(drop = True)
    
    # Add arm info
    path_arm = "/groups/wyattgrp/users/amunzur/lu_chip/resources/clinical_tables/arm_info.tsv"
    arm_df = pd.read_csv(path_arm, sep = "\t")
    
    # Generate dfs of varying thresholds and annotate their CHIP mutation status
    ch1 = annotate_mutation_status_lu(muts_df[(muts_df["VAF_n"] <= 2)], PATH_sample_information, annotate_what = "CHIP", timepoint= "Baseline").merge(age_df, how = "left").dropna().merge(arm_df)
    ch2 = annotate_mutation_status_lu(muts_df[(muts_df["VAF_n"] > 2) & (muts_df["VAF_n"] <= 10)], PATH_sample_information, annotate_what = "CHIP", timepoint= "Baseline").merge(age_df, how = "left").dropna().merge(arm_df)
    ch3 = annotate_mutation_status_lu(muts_df[muts_df["VAF_n"] > 10], PATH_sample_information, annotate_what = "CHIP", timepoint= "Baseline").merge(age_df, how = "left").dropna().merge(arm_df)
    
    ch1 = ch1[ch1["Timepoint_t"] == "Baseline"].reset_index(drop = True)
    ch2 = ch2[ch2["Timepoint_t"] == "Baseline"].reset_index(drop = True)
    ch3 = ch3[ch3["Timepoint_t"] == "Baseline"].reset_index(drop = True)
    
    colors = {"0.25%-2%": "limegreen", "2%-10%": "#709bd0", ">10%": "#224193"}
    
    bins = [40, 60, 70, 80, 90]
    labels = ['40-59', '60-69', '70-79', '80-89']
    
    for j, (df, df_annot) in enumerate(zip([ch3, ch2, ch1], [">10%", "2%-10%", "0.25%-2%"])):
        # Bin the ages
        df['Age_bin'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
        df['Age_bin'] = pd.Categorical(df['Age_bin'], categories=labels, ordered=True)
        
        # Calculate fraction that is CH+
        fractions = []
        for i, group in df.groupby("Age_bin"):
            n_pos = group[group["CHIP status"] == "Positive"].shape[0]
            n_neg = group[group["CHIP status"] == "Negative"].shape[0]
            perc_pos = n_pos/(n_pos+n_neg)
            fractions.append(perc_pos)
        
        # LOGISTIC REGRESSOIN
        df['CHIP_binary'] = df['CHIP status'].apply(lambda x: 1 if x == 'Positive' else 0)
        df['Age_bin'] = pd.Categorical(df['Age_bin'], ordered=True) # If 'Age_Bin' is already categorical, convert it to numeric codes (ordered)
        df['Age_bin_numeric'] = df['Age_bin'].cat.codes  # Convert Age Bin to numeric
        
        # Step 1: Prepare X (predictor) and y (outcome)
        X = df['Age_bin_numeric']  # Age bins as predictor
        y = df['CHIP_binary']      # Binary CHIP status as outcome
        
        # Step 2: Add constant to X for the intercept
        X = sm.add_constant(X)
        
        # Step 3: Fit logistic regression model
        logit_model = sm.Logit(y, X)
        result = logit_model.fit()
        
        odds_ratios = pd.DataFrame({"Odds Ratio": np.exp(result.params),"p-value": result.pvalues})
        print(f"CH category: {df_annot}")
        print(odds_ratios)
        
        # Step 4: Calculate Odds Ratios and Confidence Intervals
        odds_ratios = pd.DataFrame({
            "Odds Ratio": np.exp(result.params),
            "p-value": result.pvalues,
            "Lower 95% CI": np.exp(result.params - 1.96 * result.bse),  # Lower bound
            "Upper 95% CI": np.exp(result.params + 1.96 * result.bse)   # Upper bound
        })
        OR = odds_ratios[odds_ratios.index == "Age_bin_numeric"]["Odds Ratio"][0]
        p_value = odds_ratios[odds_ratios.index == "Age_bin_numeric"]["p-value"][0]
        lower_ci = odds_ratios[odds_ratios.index == "Age_bin_numeric"]["Lower 95% CI"][0]
        upper_ci = odds_ratios[odds_ratios.index == "Age_bin_numeric"]["Upper 95% CI"][0]
        
        # Step 4. Plotting LR results as an inset plot
        age_bins_numeric_range = np.linspace(df['Age_bin_numeric'].min(), df['Age_bin_numeric'].max(), 100)
        X_plot = sm.add_constant(age_bins_numeric_range)
        y_pred_probs = result.predict(X_plot)
        ax.scatter(sorted(df['Age_bin_numeric'].unique()), fractions, color=colors[df_annot], label='Actual Fraction Positive', s = 5) #Plot the actual fraction of CH-positive individuals by age bin
        ax.plot(age_bins_numeric_range, y_pred_probs, color=colors[df_annot], label='Logistic Regression Fit', linewidth = 0.8) # Step 4: Plot the logistic regression predicted probabilities
        
        # Calculating error bars for each dot
        age_bin_stats = df.groupby('Age_bin').agg(count_positive=('CHIP_binary', 'sum'),count_total=('CHIP_binary', 'size')).reset_index()
        age_bin_stats['proportion_positive'] = age_bin_stats['count_positive'] / age_bin_stats['count_total']
        conf_intervals = []
        for _, row in age_bin_stats.iterrows():
            n = row['count_total']
            p = row['proportion_positive']
            
            if n > 0:
                ci_low, ci_high = stats.binom.interval(0.95, n=n, p=p)
                # Calculate the error margins
                conf_intervals.append((p - ci_low/n, ci_high/n - p))
            else:
                conf_intervals.append((0, 0))  # No error if no total
        conf_intervals = np.array(conf_intervals).T
        
        # Plotting the error bar
        df_plotting = pd.DataFrame({'Age_bin': age_bin_stats['Age_bin'],'Positive': age_bin_stats['proportion_positive']})
        ax.errorbar(df_plotting.index, df_plotting['Positive'], yerr=conf_intervals, fmt='none', color=colors[df_annot], capsize=2, elinewidth=0.5, capthick = 0.5)
        
        # Plotting the forest
        if axforest is not None:
            axforest.scatter(OR, j, color=colors[df_annot], s = 4)
            axforest.text(2.7, j, round(p_value, 4), ha='left', va='center', fontsize=5, color = "black")
            axforest.errorbar(OR, j, xerr = [[OR - lower_ci], [upper_ci - OR]], fmt='none', color=colors[df_annot], capsize=2, elinewidth=0.5, capthick = 0.5)
            axforest.set_ylim((-1, 3))
    
    # Annotate the number of people in each age group below the x-tick labels
    patient_ages = sample_info.merge(age_df).drop_duplicates(["Patient_id", "Age"])
    patient_ages["Age_bin"] = pd.cut(patient_ages['Age'], bins=bins, labels=labels, right=False)
    patient_ages['Age_bin'] = pd.Categorical(patient_ages['Age_bin'], categories=labels, ordered=True)
    patient_ages = patient_ages["Age_bin"].value_counts().reset_index()
    patient_ages.columns=["Age_bin", "count"]
    patient_ages['index'] = pd.Categorical(patient_ages['Age_bin'], categories=labels, ordered=True)
    patient_ages = patient_ages.sort_values('index')
    x_ticklabellist = patient_ages["index"].astype(str) + "\nn=" + patient_ages["count"].astype(str).tolist()    
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(x_ticklabellist, fontsize=8)
    # AES
    ax.set_ylim((-0.1, 1.1))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(["0", "20", "40", "60", "80", "100"])
    ax.set_xlabel('Age at baseline draw')
    ax.set_ylabel('% of patients with CH')
    ax.spines[["top", "right"]].set_visible(False)
    
    # add legend
    legend_colors = colors.values()
    legend_labels = colors.keys()
    legend_handles = [plt.Line2D([0], [0], marker='o', color=color, label=label, markersize=1, linestyle='') for color, label in zip(legend_colors, legend_labels)]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, handletextpad=0.1, title = "CH VAF interval")
    
    if axforest is not None:
        axforest.axvline(1, color='black', linestyle='--', linewidth = 0.5)
        # axforest.set_ylim((-0.1, 2.1))
        axforest.set_yticks([0, 1, 2])
        axforest.tick_params(axis='y', left=False, labelleft=False)
        axforest.spines[["top", "right", "left"]].set_visible(False)
        axforest.set_yticklabels(["0.25%-2%", "2%-10%", ">10%"])
        axforest.set_xlabel("OR")
        return(ax, axforest)
    else:
        return(ax)

fig, ax = plt.subplots(figsize=(3, 3))  # adjust figsize as needed

project_dir = os.environ.get("project_dir")
baseline_ch_path=f"{project_dir}/CH_baseline.csv"
age_path=f"{project_dir}/resources/age.csv"
dir_figures=f"{project_dir}/figures/supp"
path_sample_info = f"{project_dir}/resources/sample_info.tsv"

baseline_ch=pd.read_csv(baseline_ch_path)
age_df = pd.read_csv(age_path)

dict_box_colors = {"LuPSMA": (220/255, 20/255, 60/255, 0.3), "Cabazitaxel": (154/255, 205/255, 50/255, 0.3)}
ax_forest = inset_axes(ax, width="25%", height="17%", loc='upper left')
ax = CH_presence_line_thresholds(baseline_ch, age_df, PATH_sample_information, ax, axforest = ax_forest, fontsize = 8)

fig.tight_layout()
fig.savefig(f"{dir_figures}/SUPP_age_and_CH.png")
fig.savefig(f"{dir_figures}/SUPP_age_and_CH.pdf", transparent=True)