import numpy as np
import pandas as pd
import os
from scipy.stats import fisher_exact
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import fisher_exact

project_dir = os.environ.get("project_dir")
path_utilities=f"{project_dir}/plotting_scripts/utilities.py"
with open(path_utilities, 'r') as file:
    script_code = file.read()

exec(script_code)

path_ae=f"{project_dir}/resources/TheraP Adverse Events - Adverse events (safety cohort; n=183).csv"
path_ch_baseline=f"{project_dir}/CH_baseline.csv"
path_sample_info = f"{project_dir}/resources/sample_info.tsv"
dir_figures=f"{project_dir}/figures/supp"

arm_color_dict={"LuPSMA": "#6f1fff", "Cabazitaxel": "#3e3939"}
arm_color_dict_lighter={"LuPSMA": "#d3bbfe", "Cabazitaxel": "#b2b0b0"}

sample_info=pd.read_csv(path_sample_info, sep="\t")
baseline_ch=pd.read_csv(path_ch_prog)

# Annotate CH status
baseline_ch=harmonize_vaf_columns(baseline_ch, timepoint="FirstProgression")
del baseline_ch["VAF_n"]
baseline_ch=baseline_ch.rename(columns={"VAF%": "VAF_n"})
# baseline_ch=baseline_ch.rename(columns={"Timepoint": "Timepoint_t"})
# baseline_ch=baseline_ch[baseline_ch["VAF_n"]>=2]
mut_status = annotate_mutation_status_lu(baseline_ch, path_sample_info, annotate_what="CHIP", timepoint="Baseline", annotate_gene=False)
mut_status = mut_status.merge(sample_info[['Patient_id', 'Arm']].drop_duplicates())
mut_status["CHIP status"]=mut_status["CHIP status"].replace("Positive", True).replace("Negative", False)

# Clean up AE df
ae_df=pd.read_csv(path_ae)
ae_df.columns=["Patient_id", "Arm", "System_Organ_Class", "Adverse_event", "Highest_grade_event"]
del ae_df["System_Organ_Class"]
ae_df["Patient_id"] = "TheraP-" + ae_df["Patient_id"].astype(str).str.zfill(3)
ae_df=ae_df[ae_df["Patient_id"].isin(sample_info["Patient_id"].unique())]

ae_focus=["Thromboembolic event", "Anemia", "Neutropenia (+/- fever)", "Platelet count decreased", "White blood cell decreased"]
ae_df=ae_df[ae_df["Adverse_event"].isin(ae_focus)].reset_index(drop=True)

ae_wide = ae_df.pivot_table(
    index=["Patient_id", "Arm"],
    columns="Adverse_event",
    values="Highest_grade_event",
    aggfunc=lambda x: x.iloc[0]  # in case of duplicates, take the first
).reset_index()

ae_wide.columns.name = None
ae_wide=ae_wide.merge(sample_info[["Patient_id", "Arm"]].drop_duplicates(), how="right")

ae_wide=ae_wide.merge(mut_status, how="right").reset_index(drop=True).drop("Timepoint_t", axis=1)
n_total_ch_status=ae_wide[["Arm", "CHIP status"]].value_counts()

ae_dict = {}
for ae in ae_focus:
    ae_dict[ae] = {}
    for arm in ["LuPSMA", "Cabazitaxel"]:
        ae_dict[ae][arm] = {}
        for ch_status in [True, False]:
            n_high=ae_wide[(ae_wide[ae].isin(["Grade 1", "Grade 2"])) & (ae_wide["CHIP status"]==ch_status) & (ae_wide["Arm"]==arm)].shape[0]
            n_low=ae_wide[(ae_wide[ae].isin(["Grade 3", "Grade 4", "Grade 5"])) & (ae_wide["CHIP status"]==ch_status) & (ae_wide["Arm"]==arm)].shape[0]
            n_absent=ae_wide[(ae_wide[ae].isna()) & (ae_wide["CHIP status"]==ch_status) & (ae_wide["Arm"]==arm)].shape[0]
            
            perc_high=n_high/n_total_ch_status[arm][ch_status]*100
            perc_low=n_low/n_total_ch_status[arm][ch_status]*100
            perc_absent=n_absent/n_total_ch_status[arm][ch_status]*100
            
            ae_dict[ae][arm][ch_status] = {
                "n_high": n_high,
                "perc_high": perc_high,
                "n_low": n_low,
                "perc_low": perc_low,
                "n_absent": n_absent,
                "perc_absent": perc_absent,
            }

ae_df = pd.DataFrame(
    [(ae, arm, ch, *vals.values())
     for ae, arm_dict in ae_dict.items()
     for arm, ch_dict in arm_dict.items()
     for ch, vals in ch_dict.items()],
    columns=["AE", "Arm", "CHIP", "n_high", "perc_high", "n_low", "perc_low", "n_absent", "perc_absent"]
)

# Plotting:
fig = plt.figure(figsize=(7, 4))
gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
ax_lu=fig.add_subplot(gs[0])
ax_caba=fig.add_subplot(gs[1])

ytick_list=[]
yticklabel_list=[]

for i, ae in enumerate(ae_focus):
    for factor, arm, ax in zip([-1, 1], ["LuPSMA", "Cabazitaxel"], [ax_lu, ax_caba]):
        n_fishers_exact={}
        for offset, ch_status in zip([-0.15, 0.15], [True, False]): 
            perc_high=ae_dict[ae][arm][ch_status]["perc_high"]
            perc_low=ae_dict[ae][arm][ch_status]["perc_low"]
            perc_absent=ae_dict[ae][arm][ch_status]["perc_absent"]
            
            n_high=ae_dict[ae][arm][ch_status]["n_high"]
            n_low=ae_dict[ae][arm][ch_status]["n_low"]
            
            ax.barh(i+offset, factor*perc_high, color=arm_color_dict[arm], height=0.2)
            ax.barh(i+offset, factor*perc_low, left=factor*perc_high, color=arm_color_dict_lighter[arm], height=0.2)
            # ax.barh(i+offset, factor*perc_absent, left=factor*(perc_high+perc_low), color="lightgrey", height=0.2)
            
            # Annotate patient numbers
            ax.text(factor*perc_high, i+offset, n_high, fontsize=6, color="white", va="center")
            ax.text(factor*(perc_high+perc_low), i+offset, n_low, fontsize=6, color="black", va="center")
            
            # Run Fisher's exact
            n_with_ae=ae_dict[ae][arm][ch_status]["n_high"]+ae_dict[ae][arm][ch_status]["n_low"]
            n_without_ae=ae_dict[ae][arm][ch_status]["n_absent"]
            
            n_fishers_exact[ch_status]={"with_ae": n_with_ae, "without_ae": n_without_ae}
            
            ytick_list.append(i+offset)
            yticklabel_list.append(ch_status)
        
        # Run fisher's exact
        contingency_table = [
            [n_fishers_exact[True]["with_ae"], n_fishers_exact[True]["without_ae"]],
            [n_fishers_exact[False]["with_ae"], n_fishers_exact[False]["without_ae"]],
        ]
        
        # Run Fisher's exact test
        oddsratio, p_value = fisher_exact(contingency_table)
        print(arm, ae, p_value)
        ax.text(40*factor, i, f"p = {p_value:.2f}", ha='center', va='center', fontsize=8)

yticklabel_list = ["CH−" if x is False else "CH+" for x in yticklabel_list]
ax_caba.set_yticks(ytick_list)
ax_caba.set_yticklabels(yticklabel_list)
ax_caba.tick_params(left=False)
ax_lu.tick_params(left=False)
ax_lu.set_yticks(range(0, len(ae_focus)))
ax_lu.set_yticklabels(ae_focus, fontsize=8)

ax_lu.set_title("LuPSMA")
ax_caba.set_title("Cabazitaxel")

# ax_lu.set_xlim((-100, 0))
# ax_lu.set_xticks([-100, -75, -50, -25, 0])
# ax_lu.set_xticklabels(["100", "75", "50", "25", "0"])

# ax_caba.set_xlim((0, 100))
# ax_caba.set_xticks([0, 25, 50, 75, 100])
# ax_caba.set_xticklabels(["0", "25", "50", "75", "100"])

ax_lu.set_xlim((-60, 0))
ax_lu.set_xticks([-50, -25, 0])
ax_lu.set_xticklabels(["50", "25", "0"])

ax_caba.set_xlim((0, 60))
ax_caba.set_xticks([0, 25, 50])
ax_caba.set_xticklabels(["0", "25", "50"])


ax_lu.spines[["top", "left"]].set_visible(False)
ax_caba.spines[["top", "right"]].set_visible(False)

# Add legend
legend_colors = [arm_color_dict["LuPSMA"], arm_color_dict_lighter["LuPSMA"]]
legend_labels = ["Grade≥3", "Grade<3"]
legend_handles = [plt.Line2D([0], [0], marker='s', color=color, label=label, markeredgecolor="none", markersize=10, linestyle='') for color, label in zip(legend_colors, legend_labels)]
ax_lu.legend(handles=legend_handles, loc="lower left", frameon=False, handlelength=2, handletextpad = 0.1)

legend_colors = [arm_color_dict["Cabazitaxel"], arm_color_dict_lighter["Cabazitaxel"]]
legend_labels = ["Grade≥3", "Grade<3"]
legend_handles = [plt.Line2D([0], [0], marker='s', color=color, label=label, markeredgecolor="none", markersize=10, linestyle='') for color, label in zip(legend_colors, legend_labels)]
ax_caba.legend(handles=legend_handles, loc="lower right", frameon=False, handlelength=2, handletextpad = 0.1)

fig.tight_layout()

fig.savefig(f"{dir_figures}/AE_and_CH.png")
fig.savefig(f"{dir_figures}/AE_and_CH.pdf")
# fig.savefig(f"{dir_figures}/AE_and_CH_2perc.png")
# fig.savefig(f"{dir_figures}/AE_and_CH_2perc.pdf")