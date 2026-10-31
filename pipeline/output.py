# 1. Estimate Average Causal Effect (ACE) using Pearl's Backdoor Adjustment:
#    ACE(X_i -> Y) = E[Y | do(X_i = 1)] - E[Y | do(X_i = 0)]
ace_results = {}

for feat in candidate_causal_features:
    x_vals = X_sample[feat].values.astype(float)
    x_bin = (x_vals >= np.median(x_vals)).astype(int)  # Binarize via median split
    
    # Identify confounders: Z = parents(feat) ∩ parents(Target)
    confounders = get_adjustment_set(feat_idx, TARGET_IDX, adj_matrix, node_names)
    
    if len(confounders) == 0:
        # Naive ACE when no unobserved confounding exists
        p1 = y_sample[x_bin == 1].mean()
        p0 = y_sample[x_bin == 0].mean()
        ace = float(p1 - p0)
    else:
        # Stratified backdoor adjustment across confounder strata
        ace = calculate_backdoor_adjusted_ace(x_bin, y_sample, confounders, X_sample)
        
    ace_results[feat] = {"ace": ace, "abs_ace": abs(ace)}

# 2. Filter features: keep only invariant causal drivers above threshold (|ACE| >= 0.05)
selected_features = [
    feat for feat, val in sorted(ace_results.items(), key=lambda x: x[1]["abs_ace"], reverse=True)
    if val["abs_ace"] >= 0.05
][:30]

# Writes data/causal/selected_features.json:
# ['HTTPS', 'ack_flag_number', 'psh_flag_number', 'rst_flag_number', 'rst_count', 'Variance', 'Duration']
