def harmonize_vaf_columns(muts_df, timepoint):
    """
    Merges _n and _t columns. Prioritizes _n where possible.
    timepoint is the timepoint where these mutations come from.
    """
    timepoint=timepoint.capitalize()
    
    if timepoint not in ["Baseline", "Firstprogression"]:
        raise ValueError(f"Invalid timepoint: {timepoint}. Must be 'Baseline' or 'FirstProgression'.")
    
    if timepoint=="Baseline":
        key="Progression"
    elif timepoint=="Firstprogression":
        key="Baseline"
    
    col_pairs = [
        ("VAF_n", "VAF_t"),
        ("Alt_forward_n", "Alt_forward_t"),
        ("Ref_forward_n", "Ref_forward_t"),
        ("Alt_reverse_n", "Alt_reverse_t"),
        ("Ref_reverse_n", "Ref_reverse_t"),
        ("Depth_n", "Depth_t"),
        (f"{key} WBC alt count", f"{key} cfDNA alt count"),
        (f"{key} WBC depth", f"{key} cfDNA depth"),
        (f"{key} WBC vaf%", f"{key} cfDNA vaf%")
    ]
    
    # Loop through each pair and create a new combined column
    for primary, fallback in col_pairs:
        combined_col_name = primary.replace("_n", "").replace("WBC ", "")
        if primary=="VAF_n":
            combined_col_name="VAF%"
        muts_df[combined_col_name] = muts_df[primary].combine_first(muts_df[fallback])
    
    # muts_df["Alt"]=muts_df["Alt_forward"]+muts_df["Alt_reverse"]
    # muts_df=muts_df.drop(["Alt_forward", "Alt_reverse"], axis=1)
    
    return(muts_df)

def binomial_test_vaf(df):
    results = []
    
    for _, row in df.iterrows():
        baseline_alt = int(row["Baseline alt"])  # Observed alt reads at Baseline
        baseline_depth = int(row["Baseline depth"])  # Total reads at Baseline
        progression_vaf = row["Progression VAF"] / 100  # Convert VAF% to probability
        if not pd.isnull(progression_vaf):
            if baseline_depth == 0 or progression_vaf == 0:
                p_value = 1.0  # If no coverage, we can't test
            else:
                p_value = stats.binomtest(baseline_alt, baseline_depth, progression_vaf, alternative='two-sided').pvalue
            
            results.append(p_value)
    
    df["p_value"] = results
    df["significant"] = df["p_value"] < 0.05  # Mark significant changes
    return df

def annotate_mutation_status_lu(mutations_df, PATH_sample_information, annotate_what, timepoint, annotate_gene = False): 
    """
    Given a mutation mutations_df and a list of all pts in the cohort, annotate the status of patients in the mutation list.
    Intended for survival analysis.
    annotate_what: Either provide ctDNA or CHIP.
    annotate_gene: If a gene name is provided it will annotate the status of that gene. A list can also be given.
    """
    if isinstance(annotate_gene, str):
        mutations_df = mutations_df[mutations_df["Gene"] == annotate_gene].reset_index(drop = True)
    elif isinstance(annotate_gene, list):
        mutations_df = mutations_df[mutations_df["Gene"].isin(annotate_gene)].reset_index(drop=True)
    
    mutations_df = mutations_df[["Patient_id", "Timepoint_t", "Sample_name_t", "Gene", "VAF_n", "Protein_annotation"]]
    mutations_df[annotate_what + " status"] = "Positive"
    mutations_df = mutations_df[["Patient_id", annotate_what + " status"]].drop_duplicates().reset_index(drop = True)
    all_pts = pd.read_csv(PATH_sample_information, sep="\t")
    all_pts = all_pts[all_pts["Timepoint"] == timepoint.capitalize()].rename(columns = {"Timepoint": "Timepoint_t"})
    mutations_df = all_pts[["Patient_id", "Timepoint_t"]].drop_duplicates().merge(mutations_df, how = "left")
    mutations_df[annotate_what + " status"] = mutations_df[annotate_what + " status"].fillna("Negative")
    mutations_df = mutations_df.drop_duplicates().reset_index(drop = True)[["Patient_id", "Timepoint_t", annotate_what+" status"]]
    return(mutations_df)


def calculate_OR_and_p(muts, gene, ntotal_pt_lu, n_total_pt_caba, apply_correction=False):
    """
    For a given gene calculates the OR of having a mutation in that gene in LuPSMA vs Cabazitaxel.
    """
    if gene is not None and isinstance(gene, str):
        muts_gene = muts[muts["Gene"] == gene]
        label = gene
    elif gene is not None and isinstance(gene, list):
        pattern = "|".join(map(re.escape, gene))  # Ensures special characters are treated as literals
        muts_gene = muts[muts["Gene"].str.contains(pattern, regex=True, na=False)]
        label = ", ".join(gene)  # Convert list to a readable label
    else: 
        muts_gene = muts.copy()
        label = "All genes"
    
    # Number of patients mutated vs not in each arm
    npts_mutated_lu = muts_gene[muts_gene["Arm"] == "LuPSMA"]["Patient_id"].drop_duplicates().shape[0]
    npts_mutated_caba = muts_gene[muts_gene["Arm"] == "Cabazitaxel"]["Patient_id"].drop_duplicates().shape[0]      
    
    npts_nonmutated_lu=ntotal_pt_lu-npts_mutated_lu
    npts_nonmutated_caba=n_total_pt_caba-npts_mutated_caba
    
    # Apply continuity correction if requested or if needed due to zero cells
    a, b, c, d = npts_mutated_lu, npts_nonmutated_lu, npts_mutated_caba, npts_nonmutated_caba
    if apply_correction or 0 in [a, b, c, d]:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5
        corrected = True
    else:
        corrected = False
    
    # Create the contingency table for Fisher's exact test (always uncorrected values)
    contingency_table = [[npts_mutated_lu, npts_nonmutated_lu],
                         [npts_mutated_caba, npts_nonmutated_caba]]
    _, p_value = stats.fisher_exact(contingency_table)
    
    # Calculate Odds Ratio and CI
    OR = (a / b) / (c / d)
    se_log_or = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    log_or = math.log(OR)
    z = 1.96  # for 95% CI
    ci_lower = math.exp(log_or - z * se_log_or)
    ci_upper = math.exp(log_or + z * se_log_or)
    
    return {
        "Label": label,
        "OR": OR,
        "p Fisher": p_value,
        "CI_lower": ci_lower,
        "CI_upper": ci_upper,
        "Correction applied": corrected
    }

def plot_vaf_change(df, gene, ax, plot_delta=True, plot_days=True): 
    """
    Plots vaf change from baseline to progression.
    """
    if gene is not None:
        df_gene=df[df["Gene"]==gene]
    else:
        df_gene=df.copy()
    
    df_gene=binomial_test_vaf(df_gene)
    
    nmuts=df_gene.shape[0]
        
    for _, row in df_gene.iterrows():
        base_vaf=row["Baseline VAF"]
        prog_vaf=row["Progression VAF"]
        if row["significant"]:
            if prog_vaf>base_vaf: # VAF incr
                color="orangered"
            else:
                color="royalblue"
        else:
            color="mediumseagreen"
        
        if plot_days:
            xpos_prog=row["Date diff in months"]
            # ax.set_xlabel("Days")
        else:
            xpos_prog=2
            ax.set_xlim(0.5, 2.5)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Base", "Prog"])
        
        if plot_delta:
            ax.scatter(1, 0, s = 5, edgecolor = None, color = color)
            ax.scatter(xpos_prog, prog_vaf-base_vaf, s = 2, edgecolor = None, color = color)
            ax.plot([1, xpos_prog], [0, prog_vaf-base_vaf], color=color, linewidth=0.5)
        else:
            ax.scatter(1, base_vaf, s = 5, edgecolor = None, color = color)
            ax.scatter(xpos_prog, prog_vaf, s =2, edgecolor = None, color = color)
            ax.plot([1, xpos_prog], [base_vaf, prog_vaf], color=color, linewidth=0.7)
    
    # ax.set_xlabel("Timepoint")
    ax.set_ylabel("Δ VAF%")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return(ax, nmuts)


def compare_growth_rate_per_gene_BOX(df, ax, gene=None, protein_alt=None):
    df_gene=df[df["Gene"]==gene]
    nmuts=df_gene.shape[0]
    
    # Boxp aes
    flierprops_dict = dict(marker='o', markersize=5, markeredgecolor='black', linestyle='None')
    whiskerprops_dict =dict(color='black')
    medianprops_dict = dict(color='black')
    capprops_dict = dict(color='black')
    
    # We color by timepoint
    timepoint_color_dict = {"Baseline": "blue", "FirstProgression": "darkorange"}
    df_gene["Timepoint color"]=df["Timepoint"].map(timepoint_color_dict)
    
    growth_rate_list=[]
    for i, x_offset, arm in zip([0, 1], [-0.4, 0.4], ["LuPSMA", "Cabazitaxel"]):
        gr=df_gene[df_gene["Arm"]==arm]["Growth rate"]
        colors=df_gene[df_gene["Arm"]==arm]["Timepoint color"].tolist()
        boxprops_dict = dict(facecolor="white", edgecolor='black', linewidth = 0.7)  
        boxplot = ax.boxplot(gr, positions = [i+x_offset], flierprops = flierprops_dict, boxprops=boxprops_dict, medianprops = medianprops_dict, capprops = capprops_dict, widths = 0.8, showfliers = False, patch_artist = True)
        ax.scatter(np.random.uniform(i+x_offset-0.25, i+x_offset+0.25, len(gr)), gr, s = 8, color = colors, edgecolor="black", linewidths=0.5, zorder = 100)
        
        gr_raw=df_gene[df_gene["Arm"]==arm]["Growth rate"].tolist()
        growth_rate_list.append(gr_raw)
        
        n=len(gr)
        ax.text(i+x_offset, -0.75, f"n={n}", va="bottom", ha="center", fontsize=6)
    
    # Run MWU on VAFs
    if len(growth_rate_list[0])>0 and growth_rate_list[1]:
        p = ttest_ind(growth_rate_list[0], growth_rate_list[1], equal_var=False).pvalue
        p_formatted = f"{p:.1g}"
        ax.text(i, 0.9, str(f"t-test\np={p_formatted}"), ha='center', va='top', fontsize=7, color='black')
    
    ax.set_ylabel("Natural log growth rate")
    ax.set_xticklabels(["LuPSMA", "Cabazitaxel"])    
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim((-0.2, 0.2))
    ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax.set_xlim([-1.2, 2.2])
    return(ax, nmuts)

