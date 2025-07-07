import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu
from scipy import stats
import os
from scipy.stats import fisher_exact

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
path_sample_info = f"{project_dir}/resources/sample_info.tsv"
baseline_ch_path=f"{project_dir}/CH_baseline.csv"
progression_ch_path=f"{project_dir}/CH_progression.csv"
dir_figures=f"{project_dir}/figures/supp"

path_utilities=f"{project_dir}/plotting_scripts/utilities.py"
with open(path_utilities, 'r') as file:
    script_code = file.read()

exec(script_code)

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
combined_muts=pd.concat([baseline_ch_subset, progression_ch], ignore_index=True)

combined_muts.loc[combined_muts["Baseline VAF"]==0, "Baseline VAF"]=0.25
combined_muts.loc[combined_muts["Progression VAF"]==0, "Progression VAF"]=0.25

comparison_table = []

DTA_genes=["DNMT3A", "TET2", "ASXL1"]
DDR_genes=["TP53", "PPM1D", "ATM", "CHEK2"]

gene_sets = {
    "ALL genes": sorted(combined_muts["Gene"].dropna().unique()),
    "DTA genes combined": DTA_genes,
    "DDR genes combined": DDR_genes
}

# Add individual genes
for gene in DTA_genes + DDR_genes:
    gene_sets[gene] = [gene]

gene_sets = {
    "ALL genes": sorted(combined_muts["Gene"].dropna().unique()),
    "DTA genes combined": DTA_genes,
    "DNMT3A": ["DNMT3A"],
    "TET2": ["TET2"],
    "ASXL1": ["ASXL1"],
    "DDR genes combined": DDR_genes,
    "PPM1D": ["PPM1D"],
    "TP53": ["TP53"],
    "ATM": ["ATM"],
    "CHEK2": ["CHEK2"]
}

p_values={}
numbers={}

for group_name, genes_list in gene_sets.items():
    row = {"Gene group": group_name}
    
    mydict={}
    for arm in ["LuPSMA", "Cabazitaxel"]:
        subset = combined_muts[(combined_muts["Gene"].isin(genes_list)) & (combined_muts["Arm"] == arm)]
        subset = binomial_test_vaf(subset)
        
        total = len(subset)
        incr = subset[(subset["significant"]) & (subset["Progression VAF"] > subset["Baseline VAF"])]
        decr = subset[(subset["significant"]) & (subset["Progression VAF"] < subset["Baseline VAF"])]
        stable = subset[~subset["significant"]]
        
        def fmt(n, total):
            perc = (n / total * 100) if total > 0 else 0
            return f"{n}/{total} ({perc:.1f}%)"
        
        row[f"{arm}_INCR"] = fmt(len(incr), total)
        row[f"{arm}_STABLE"] = fmt(len(stable), total)
        row[f"{arm}_DECR"] = fmt(len(decr), total)
        
        mydict[arm]=len(incr)
        # numbers[group_name]
        
    comparison_table.append(row)

# Convert to DataFrame
comparison_df = pd.DataFrame(comparison_table)
# print(comparison_df.to_string(index=False))

# Here run the Fisher's exact test on prevalence of increasing mutations
for gene_group in ["ALL genes", "DTA genes combined", "DDR genes combined"]:
    n_lu_incr=comparison_df[comparison_df["Gene group"]==gene_group]["LuPSMA_INCR"].values[0].split("/")[0]
    n_lu_total=comparison_df[comparison_df["Gene group"]==gene_group]["LuPSMA_INCR"].values[0].split("/")[1].split(" ")[0]
    n_caba_incr=comparison_df[comparison_df["Gene group"]==gene_group]["Cabazitaxel_INCR"].values[0].split("/")[0]
    n_caba_total=comparison_df[comparison_df["Gene group"]==gene_group]["Cabazitaxel_INCR"].values[0].split("/")[1].split(" ")[0]
    
    lu_not_incr = int(n_lu_total) - int(n_lu_incr)
    caba_not_incr = int(n_caba_total) - int(n_caba_incr)
    contingency = [[int(n_lu_incr), lu_not_incr], [int(n_caba_incr), caba_not_incr]]
    
    odds_ratio, p_value = fisher_exact(contingency, alternative="two-sided")
    
    print(f"Gene group: {gene_group}")
    print(f"Contingency table: {contingency}")
    print(f"Odds ratio: {odds_ratio:.2f}, P-value: {p_value}")
    print("------------")


# Extract percentage values
def extract_percentage(s):
    return float(s.split("(")[1].split("%")[0])

percent_df = comparison_df.copy()
for col in comparison_df.columns[1:]:
    percent_df[col] = comparison_df[col].apply(extract_percentage)

# Insert spacer column
percent_df.insert(4, 'Spacer', np.nan)

# Insert blank rows after 'ALL genes' and 'ASXL1'
def insert_blank_after(df, row_label):
    idx = df.index[df['Gene group'] == row_label][0]
    blank_row = pd.Series([np.nan] * len(df.columns), index=df.columns)
    return pd.concat([df.iloc[:idx+1], pd.DataFrame([blank_row]), df.iloc[idx+1:]], ignore_index=True)

percent_df = insert_blank_after(percent_df, 'ALL genes')
percent_df = insert_blank_after(percent_df, 'ASXL1')
comparison_df = insert_blank_after(comparison_df, 'ALL genes')
comparison_df = insert_blank_after(comparison_df, 'ASXL1')

# Reverse index so that genes appear in correct order
percent_df = percent_df.iloc[::-1].reset_index(drop=True)
comparison_df = comparison_df.iloc[::-1].reset_index(drop=True)

# Custom colormaps
incr_cmap = LinearSegmentedColormap.from_list("incr_cmap", ["white", "orangered"])
stable_cmap = LinearSegmentedColormap.from_list("stable_cmap", ["white", "mediumseagreen"])
decr_cmap = LinearSegmentedColormap.from_list("decr_cmap", ["white", "royalblue"])

col_cmaps = {
    'Cabazitaxel_INCR': incr_cmap,
    'Cabazitaxel_STABLE': stable_cmap,
    'Cabazitaxel_DECR': decr_cmap,
    'LuPSMA_INCR': incr_cmap,
    'LuPSMA_STABLE': stable_cmap,
    'LuPSMA_DECR': decr_cmap
}

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
col_widths = [1] * (len(percent_df.columns) - 1)
col_widths[3]=0.2  # Make spacer column narrower
col_heights=[1] * (len(percent_df.index))
col_heights[5]=col_heights[10]=0.2

x_positions = np.cumsum([0] + col_widths)
y_positions = np.cumsum([0] + col_heights[::-1])[::-1][1:]

for row_idx, row in enumerate(percent_df.itertuples(index=False)):
    for col_idx, col_name in enumerate(percent_df.columns[1:]):
        xpos = x_positions[col_idx]
        width = col_widths[col_idx]
        height = col_heights[row_idx]
        val = getattr(row, col_name)
        if pd.isna(val):
            color = 'white'
            annotation = ""
        else:
            percent = extract_percentage(getattr(comparison_df.iloc[row_idx], col_name))
            cmap = col_cmaps.get(col_name, plt.cm.Greys)
            normed = percent / 100
            color = cmap(normed)
            annotation = getattr(comparison_df.iloc[row_idx], col_name)
        
        ypos = y_positions[row_idx]
        ax.add_patch(plt.Rectangle((xpos, ypos), width, height, color=color))
        if annotation:
            ax.text(xpos + width / 2, ypos + height / 2, annotation, ha='center', va='center', fontsize=9)

# Formatting
ax.set_xlim(0, x_positions[-1])
ax.set_ylim(0, sum(col_heights))
ax.set_xticks([(x_positions[i] + x_positions[i+1]) / 2 for i in range(len(col_widths))])
ax.set_xticklabels(percent_df.columns[1:], rotation=45, ha='right')
yticks = [y + h / 2 for y, h in zip(y_positions, col_heights)]
ax.set_yticks(yticks)
ax.set_yticklabels(percent_df['Gene group'].replace(np.nan, ""))
ax.invert_yaxis()
ax.tick_params(length=0)
ax.set_xticks([], minor=True)
ax.set_yticks([], minor=True)

plt.tight_layout()

ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
plt.savefig(f"{dir_figures}/SUPP_dta_ddr_vaf_change_heatmap.pdf")