import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportions_ztest
from matplotlib.lines import Line2D

mpl.rcParams['hatch.linewidth'] = 0.3
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

def plot_ch_prevalence_by_arm(muts_df, path_sample_information, color_dict, ax, annotate_what, add_legend= True, lu_pts_n_total = None, caba_pts_n_total = None, add_p_values = True, minvaf=0):
    """
    Plots the presence / absence of CH in the cohort as bar charts. Separates into two groups.
    """
    # Use WBC VAF where possible, if it isn't available use cfDNA VAF
    col_pairs = [
    ("VAF_n", "VAF_t")]
    
    # Loop through each pair and create a new combined column
    for primary, fallback in col_pairs:
        combined_col_name = primary.replace("_n", "").replace("WBC ", "")
        muts_df[combined_col_name] = muts_df[primary].combine_first(muts_df[fallback])
    
    sample_info = pd.read_csv(path_sample_information, sep = "\t")    
    
    if lu_pts_n_total is None and caba_pts_n_total is None:
        lu_pts_n_total = sample_info[sample_info["Arm"] == "LuPSMA"]["Patient_id"].drop_duplicates().shape[0]
        caba_pts_n_total = sample_info[sample_info["Arm"] == "Cabazitaxel"]["Patient_id"].drop_duplicates().shape[0]
        
    # if annotate_what.lower() == "chip":
    #     col_to_filter = "VAF_n"
    # else:
    #     col_to_filter = "VAF_t"
    
    col_to_filter="VAF"
    
    results_dict = {}
    for min_vaf in [minvaf, 2, 10]:
        muts_df_filtered  = muts_df[muts_df[col_to_filter] >= min_vaf]
        
        lu_pts = muts_df_filtered[muts_df_filtered["Arm"] == "LuPSMA"]["Patient_id"].unique().shape[0]
        lu_perc = round((lu_pts/lu_pts_n_total)*100)
        
        caba_pts = muts_df_filtered[muts_df_filtered["Arm"] == "Cabazitaxel"]["Patient_id"].unique().shape[0]
        caba_perc = round((caba_pts/caba_pts_n_total)*100)
        
        print(f"For min vaf {min_vaf} nlu_pts in {lu_pts}, ncaba_pts is {caba_pts}")
        
        # save results
        results_dict[min_vaf] = {"LuPSMA perc": lu_perc, "Cabazitaxel perc": caba_perc, "LuPSMA npts": lu_pts, "Cabazitaxel npts": caba_pts}
        # sum_perc = (lu_pts + caba_pts)/(group1_ntotal+group2_ntotal)
        # print(f"Min vaf={min_vaf}, {sum_perc}")
    
    # Now plotting the bar charts    
    df = pd.DataFrame.from_dict(results_dict, orient='index').reset_index().rename(columns = {"index": "min_vaf"})
    for i, row in df.iterrows():
        ax.bar(i-0.2, row["LuPSMA perc"], color = color_dict["LuPSMA"], edgecolor = "none", width = 0.4)
        ax.bar(i+0.2, row["Cabazitaxel perc"], color = color_dict["Cabazitaxel"], edgecolor = "none", width = 0.4)
        
        # Annotate the number of patients on top of the bars
        ax.text(i - 0.2, row["LuPSMA perc"]+1, str(row["LuPSMA npts"]), ha='center', va='bottom', fontsize=6, color='black')
        ax.text(i + 0.2, row["Cabazitaxel perc"]+1, str(row["Cabazitaxel npts"]), ha='center', va='bottom', fontsize=6, color='black')
    
    # Z test
    if add_p_values:
        p_values_ztest = []
        nobs = np.array([lu_pts_n_total, caba_pts_n_total])
        for i, (group1_count, group2_count) in enumerate(zip(df["LuPSMA npts"], df["Cabazitaxel npts"])):
            counts = np.array([group1_count, group2_count])
            _, p_val = proportions_ztest(counts, nobs, alternative="larger")  # alternative="larger" for one-sided test
            p_values_ztest.append(p_val)
        
        for i, p_val in enumerate(p_values_ztest):
            ypos=97
            # Get the bladder count, the p values will be annotated right on top of it.
            count_value = df.loc[i, "LuPSMA perc"]
            if p_val < 0.001:
                ax.text(i, ypos, "***", ha='center', va='bottom', color='black')
                print(f"p value at x position {i} is {p_val}")
            elif p_val < 0.01:
                ax.text(i, ypos, "**", ha='center', va='bottom', color='black')
                print(f"p value at x position {i} is {p_val}")
            elif p_val < 0.05:
                ax.text(i, ypos, "*", ha='center', va='bottom', color='black')
                print(f"p value at x position {i} is {p_val}")
            elif p_val>=0.05:
                ax.text(i, ypos, "ns", ha='center', va='bottom', color='black')
                print(f"p value at x position {i} is {p_val}")
            
    ax.set_xlim((-0.7, 2.5))
    ax.set_xticks([0, 1, 2])
    # ax.tick_params(axis='x', bottom=False)
    # ax.tick_params(axis='x', pad=2)
    ax.set_xticklabels(["≥0.25", "≥2", "≥10"])
    ax.set_ylabel("% of patients")
    ax.set_xlabel("Minimum CH VAF%")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax.set_ylim((0, 100))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    
    # ADD LEGEND
    if add_legend:
        legend_colors = color_dict.values()
        legend_labels = color_dict.keys()
        legend_handles = [plt.Line2D([0], [0], marker='s', color=color, label=label, markersize=5, linestyle='') for color, label in zip(legend_colors, legend_labels)]
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, handlelength=2, handletextpad = 0.1)
    
    if min_vaf!=0:
        print("Dont forget to rename the x axis ticklabels.")
    return(ax)

def plot_genewise_ch_mutations_by_arm(muts_df, genes_list, path_sample_info, gene_color_dict, ax, timepoint, raw_counts=False, add_vaf_swarm=False, vafcolname=None, pvalue=False, arm_color_dict=None, color_arm_instead_of_gene=False, add_legend=True):
    
    """
    If color_arm_instead_of_gene is set to true, colors bars based on Lu and Caba colors.
    """
    if color_arm_instead_of_gene and arm_color_dict is None:
        raise ValueError("When color_arm_instead_of_gene is set to True, must provide argument for arm_color_dict.")
    # vaf_ax=ax.twiny()
    if raw_counts: 
        division_factor_dict={'LuPSMA': 1, 'Cabazitaxel': 1}
    else:
        sample_info=pd.read_csv(path_sample_info, sep="\t")
        ntotal_lu=sample_info[(sample_info["Timepoint"]==timepoint) & (sample_info["Arm"]=="LuPSMA") & (sample_info["Cohort"]=="TheraP")].shape[0]
        ntotal_caba=sample_info[(sample_info["Timepoint"]==timepoint) & (sample_info["Arm"]=="Cabazitaxel") & (sample_info["Cohort"]=="TheraP")].shape[0]
        division_factor_dict={'LuPSMA': ntotal_lu, 'Cabazitaxel': ntotal_caba}
    
    xtick_list=[]
    xticklabel_list=[]
    
    for i, gene in enumerate(genes_list):
        contingency_table = []  # For Fisher's exact test
        y_positions=[]  # Store y positions for later 
        
        for offset, arm in zip([-0.2, 0.2], ["LuPSMA", "Cabazitaxel"]):
            
            xticklabel_list.append(arm[0])
            xtick_list.append(i+offset)
            
            muts_subset_df = muts_df[(muts_df["Gene"] == gene) & (muts_df["Arm"] == arm)]
            if color_arm_instead_of_gene:
                bar_color = arm_color_dict[arm] 
            else:
                bar_color = gene_color_dict[gene]   
            
            # For Fisher's exact test
            nmuts_raw = muts_subset_df.shape[0]
            npts = muts_subset_df["Patient_id"].unique().shape[0]
            total_patients = ntotal_lu+ntotal_caba
            contingency_table.append([npts, total_patients - npts])   
            
            # Determine nmuts for plotting
            if raw_counts:
                nmuts = nmuts_raw
            else:
                nmuts = muts_subset_df["Patient_id"].nunique() / division_factor_dict[arm]  
                
            gene_name_pos = 80  
                
            ax.bar(i+offset, nmuts, color=bar_color, width=0.4, edgecolor="white", linewidth=0.3)
            ax.text(i+offset, nmuts, str(npts), fontsize=6, va="bottom", ha="center")
            y_positions.append(nmuts)
            
        # Run Fisher's exact test
        ymax=ax.get_ylim()[1]
        if len(contingency_table) == 2:
            odds_ratio, p_value = fisher_exact(contingency_table)
            # Find the correct y-position for p-value annotation
            pval_y_pos = max(y_positions)  # Place on the upper side   
            if pvalue:
                pval_y_pos=0.41
                if p_value>=0.05:
                    to_print="ns"
                elif p_value<=0.05:
                    to_print="*"
                elif p_value<=0.01:
                    to_print="**"
                elif p_value<=0.001:
                    to_print="***"
                ax.text(i, ymax, to_print, ha="center", va="top", fontsize=8, color="black")      
                print(f"p value at x position {i} is {p_value}")        
    
    if raw_counts:
        ax.set_ylim((0, 80))
        ax.set_yticks([0, 20, 40, 60])
        ax.set_yticklabels(["0", "20", "40", "60"])
    else:
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_yticklabels(["0", "10", "20", "30", "40", "50"])
    ax.spines[["top", "right"]].set_visible(False)
    print(xticklabel_list)
    
    if color_arm_instead_of_gene:
        ax.set_xticks(range(0, len(genes_list)))
        ax.set_xticklabels(genes_list, rotation=90, fontstyle="italic")
    else:
        ax.set_xticks(xtick_list)
        ax.set_xticklabels(xticklabel_list)
    
    # ax.tick_params(left=False)
    # [ax.axhline(ytick, color="grey", linewidth=0.5, zorder=-2) for ytick in ax.get_yticks()]
    ax.set_ylabel("% patients")
    
    # ADD THE VAF SWARM
    if add_vaf_swarm and vafcolname is not None:
        ax_vaf=ax.twiny()
        for i, gene in enumerate(genes_list):
            for offset, arm in zip([-0.2, 0.2], ["Cabazitaxel", "LuPSMA"]):
                muts_vafs=np.log10(muts_df[(muts_df["Gene"] == gene) & (muts_df["Arm"] == arm)][vafcolname])
                if reflect_across_y:
                    muts_vafs=muts_vafs*-1
                ax_vaf.scatter(muts_vafs, np.random.uniform(-0.07, 0.07, len(muts_vafs))+i+offset, alpha=0.4, color="black", s=3)
        
        ax_vaf.set_ylim((np.log10(0.2), np.log10(100)))
        ax_vaf.set_yticks([np.log10(0.25), np.log10(1), np.log10(2), np.log10(10), np.log10(50), np.log10(90)])
        ax_vaf.set_yticklabels(["0.25", "1", "2", "10", "50", "90"])
        ax_vaf.spines[["left", "right"]].set_visible(False)
        ax_vaf.set_xlabel("VAF%")
        ax_vaf.xaxis.set_label_position("top")  # Moves the label to the top
        return(ax, ax_vaf)
    
    # # Add legend
    if color_arm_instead_of_gene:
        if add_legend:
            legend_colors = arm_color_dict.values()
            legend_labels = arm_color_dict.keys()
            legend_handles=[plt.Line2D([0], [0], marker='s', color=color, label=label, markersize=5, linestyle='', markeredgecolor='none') for color, label in zip(legend_colors, legend_labels)]
            ax.legend(handles=legend_handles, loc="upper right", frameon=False, handlelength=2, handletextpad = 0.1, ncol=1, bbox_to_anchor=(1, 0.95)) 
    
    return(ax)

def plot_stacked_gene_mutation_counts(muts_df, ax, gene_color_dict, path_sample_info, timepoint, raw_counts=False, reflect_across_y=False, add_bar_labels=True):
    """
    Barchart showing absolute mutation counts in genes of interest. Genes outside of the color_dict will be in the other category shown in grey.
    """
    if raw_counts: 
        division_factor_dict={'LuPSMA': 1, 'Cabazitaxel': 1}
    else:
        sample_info=pd.read_csv(path_sample_info, sep="\t")
        ntotal_lu=sample_info[(sample_info["Timepoint"]==timepoint) & (sample_info["Arm"]=="LuPSMA") & (sample_info["Cohort"]=="TheraP")].shape[0]
        ntotal_caba=sample_info[(sample_info["Timepoint"]==timepoint) & (sample_info["Arm"]=="Cabazitaxel") & (sample_info["Cohort"]=="TheraP")].shape[0]
        division_factor_dict={'LuPSMA': ntotal_lu, 'Cabazitaxel': ntotal_caba}
    
    non_other_genes=list(gene_color_dict.keys())
    muts_df.loc[~muts_df["Gene"].isin(non_other_genes), "Gene"] = "Other"
    
    for i, arm in enumerate(['LuPSMA', 'Cabazitaxel']):
        arm_muts=muts_df[muts_df["Arm"]==arm]
        if raw_counts:
            arm_muts_counts=arm_muts["Gene"].value_counts().reset_index()
            arm_muts_counts.columns=["Gene", "Counts"]
        else:
            arm_muts_counts=arm_muts[["Patient_id", "Gene"]].drop_duplicates()["Gene"].value_counts().reset_index()
            arm_muts_counts.columns=["Gene", "Counts"]
            arm_muts_counts["Counts"]=arm_muts_counts["Counts"]/division_factor_dict[arm]
                
        arm_muts_counts['Gene'] = pd.Categorical(arm_muts_counts['Gene'], categories=list(gene_color_dict.keys()), ordered=True)
        arm_muts_counts = arm_muts_counts.sort_values('Gene').reset_index(drop=True)
        
        bottom = 0
        for j, row in arm_muts_counts.iterrows():
            gene_counts = row["Counts"]
            bar_color = gene_color_dict.get(row["Gene"], 'grey')  # Default to grey for "Other"
            
            ax.bar(i, gene_counts, bottom=bottom, color=bar_color)
            bottom += gene_counts  # Increase bottom to stack the next positive bar
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["LuPSMA", "Cabazitaxel"])
            if raw_counts:
                ax.set_yticks([0, 50, 100, 150, 200])
                ax.set_yticklabels(["0", "50", "100", "150", "200"])
            else:
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_yticklabels(["0", "25", "50", "75", "100"])
        
    ax.set_title("")
    ax.set_ylabel("Number of mutations")
    return(ax)

def add_gene_name_legend(gene_color_dict, ax):
    gene_order = ["TP53", "PPM1D", "CHEK2", "ATM", "DNMT3A", "TET2", "ASXL1", "Other"]
    legend_labels = [rf"$\it{{{x}}}$" if x != "Other" else x for x in gene_order]
    legend_colors = [gene_color_dict[x] for x in gene_order]
    legend_handles = [plt.Line2D([0], [0], marker='s', color=color, label=label, markersize=5, linestyle='', markeredgecolor='none') for color, label in zip(legend_colors, legend_labels)]
    ax.legend(handles=legend_handles, loc='upper right', frameon=False, handlelength=0.9, handleheight=1, handletextpad=0.2, labelspacing=0.3, borderaxespad=0.1, ncol=1, fontsize=6)
    ax.axis("off")
    return ax

project_dir = os.environ.get("project_dir")

path_sample_info = f"{project_dir}/resources/sample_info.tsv"
baseline_ch_path=f"{project_dir}/CH_baseline.csv"
progression_ch_path=f"{project_dir}/CH_progression.csv"
dir_figures=f"{project_dir}/figures/main"

arm_color_dict={"LuPSMA": "#6f1fff", "Cabazitaxel": "#3e3939"}

gene_color_dict = {
    "Other": "Silver",

    "ATM":"#ECAFAF",
    "CHEK2":"#E08B8B",
    "PPM1D":"#D46666",
    "TP53": "#943434",

    "ASXL1": "#5cadad",
    "TET2": "#008080",
    "DNMT3A": "#004c4c",
}

# LOAD DATASETS
sample_info = pd.read_csv(path_sample_info, sep= "\t")
pts_with_progression_samples = sample_info[sample_info["Timepoint"] == "FirstProgression"]["Patient_id"].drop_duplicates()
baseline_ch = pd.read_csv(baseline_ch_path)
baseline_ch_subset = baseline_ch[baseline_ch["Patient_id"].isin(pts_with_progression_samples)]
progression_ch = pd.read_csv(progression_ch_path)
progression_ch=progression_ch[progression_ch["Independently detected at baseline"]==False]

# PLOTTING
fig = plt.figure(figsize=(8, 4.5))

gs_outer=gridspec.GridSpec(2, 4, width_ratios=[0.5, 1, 0.5, 0.1], height_ratios=[1,1], hspace=0.38, wspace=0.3)

ax0_row0=plt.subplot(gs_outer[0,0])
ax1_row0=plt.subplot(gs_outer[0,1])
ax2_row0=plt.subplot(gs_outer[0,2])
ax2_row0_legend=plt.subplot(gs_outer[0,3])

ax0_row1=plt.subplot(gs_outer[1,0], sharex=ax0_row0)
ax1_row1=plt.subplot(gs_outer[1,1])
ax2_row1=plt.subplot(gs_outer[1,2])
ax2_row1_legend=plt.subplot(gs_outer[1,3])

ax0_row0=plot_ch_prevalence_by_arm(baseline_ch, path_sample_info, arm_color_dict, ax0_row0, annotate_what = "CHIP")
ax0_row1=plot_ch_prevalence_by_arm(progression_ch, path_sample_info, arm_color_dict, ax0_row1, lu_pts_n_total = 60, caba_pts_n_total = 47, annotate_what = "CHIP")

genes_list=['DNMT3A', 'TET2', 'ASXL1', 'ATM', 'CHEK2', 'PPM1D', 'TP53']

ax1_row0.set_ylim((0, 0.50))
ax1_row1.set_ylim((0, 0.55))
ax1_row0=plot_genewise_ch_mutations_by_arm(baseline_ch, genes_list, path_sample_info, gene_color_dict, ax1_row0, timepoint="Baseline", raw_counts=False, add_vaf_swarm=False, vafcolname=None, pvalue=True, arm_color_dict=arm_color_dict, color_arm_instead_of_gene=True)
ax1_row1=plot_genewise_ch_mutations_by_arm(progression_ch, genes_list, path_sample_info, gene_color_dict, ax1_row1, timepoint="FirstProgression", raw_counts=False, add_vaf_swarm=False, vafcolname=None, pvalue=True, arm_color_dict=arm_color_dict, color_arm_instead_of_gene=True, add_legend=False)

ax2_row0=plot_stacked_gene_mutation_counts(baseline_ch, ax2_row0, gene_color_dict, path_sample_info, timepoint="Baseline", raw_counts=True, add_bar_labels=True)
ax2_row1=plot_stacked_gene_mutation_counts(progression_ch, ax2_row1, gene_color_dict, path_sample_info, timepoint="FirstProgression", raw_counts=True, add_bar_labels=True)
ax2_row1.set_ylim(0, 150)
ax2_row1.set_yticks([0, 50, 100, 150])
ax2_row1.set_yticklabels(["0", "50", "100", "150"])

ax2_row0_legend=add_gene_name_legend(gene_color_dict, ax2_row0_legend)
ax2_row1_legend=add_gene_name_legend(gene_color_dict, ax2_row1_legend)

fig.savefig(f"{dir_figures}/baseline_and_prog_ch_prevalence.png")
fig.savefig(f"{dir_figures}/baseline_and_prog_ch_prevalence.pdf", facecolor='none', transparent=True)