// kernels.cl - The Complete Suite

// --- HELPER: Log-Sum-Exp Trick ---
// Prevents overflow when adding log-probabilities
float log_add(float log_a, float log_b) {
    float max_val = max(log_a, log_b);
    float min_val = min(log_a, log_b);
    return max_val + log1p(exp(min_val - max_val));
}

// --- SECTION 1: Basic Markov Operations ---

// 1. Standard Markov Step: Next = Current * Matrix
__kernel void markov_step(
    const int N,
    __global const float *current_state,
    __global const float *transition_mat,
    __global float *next_state)
{
    int id = get_global_id(0);
    if (id < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += current_state[k] * transition_mat[k * N + id];
        }
        next_state[id] = sum;
    }
}

// 2. Standard HMM Filter (Probability Space)
// Used for simple "What state am I in?" queries without log-space
__kernel void hmm_forward_step(
    const int N,
    __global const float *alpha_prev,
    __global const float *trans_mat,
    __global const float *emissions,
    __global float *alpha_new)
{
    int id = get_global_id(0);
    if (id < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += alpha_prev[k] * trans_mat[k * N + id];
        }
        alpha_new[id] = sum * emissions[id];
    }
}

// --- SECTION 2: Advanced Log-Space Operations (Stable) ---

// 3. Log-Space Forward (For Viterbi & Training)
__kernel void hmm_forward_log(
    const int N,
    __global const float *log_alpha_prev, 
    __global const float *log_trans_mat,  
    __global const float *log_emissions,  
    __global float *log_alpha_new)        
{
    int id = get_global_id(0);
    if (id < N) {
        float log_sum = -INFINITY;
        for (int k = 0; k < N; k++) {
            float val = log_alpha_prev[k] + log_trans_mat[k * N + id];
            if (k == 0) log_sum = val;
            else log_sum = log_add(log_sum, val);
        }
        log_alpha_new[id] = log_sum + log_emissions[id];
    }
}

// 4. Log-Space Backward (For Training)
__kernel void hmm_backward_log(
    const int N, 
    __global const float *beta_future, 
    __global const float *trans, 
    __global const float *emis_future, 
    __global float *beta_curr) 
{
    int id = get_global_id(0); // State 'i'
    if (id < N) {
        float log_sum = -INFINITY;
        for (int j=0; j<N; j++) {
            // transition i->j + emission(t+1) + beta(t+1)
            float val = trans[id*N + j] + emis_future[j] + beta_future[j];
            if (j==0) log_sum = val;
            else log_sum = log_add(log_sum, val);
        }
        beta_curr[id] = log_sum;
    }
}

// 5. Viterbi Algorithm (Finds best path)
__kernel void viterbi_step(
    const int N,
    __global const float *log_delta_prev,
    __global const float *log_trans_mat,
    __global const float *log_emissions,
    __global float *log_delta_new,       
    __global int *backpointers)          
{
    int id = get_global_id(0);
    if (id < N) {
        float max_prob = -INFINITY;
        int best_prev_state = 0;

        for (int k = 0; k < N; k++) {
            float prob = log_delta_prev[k] + log_trans_mat[k * N + id];
            if (prob > max_prob) {
                max_prob = prob;
                best_prev_state = k;
            }
        }
        log_delta_new[id] = max_prob + log_emissions[id];
        backpointers[id] = best_prev_state; 
    }
}

// --- SECTION 3: Learning Accumulators (Baum-Welch) ---

// 6. Accumulate Transitions (E-Step)
// Condenses time T into N*N summary matrix
__kernel void accumulate_transitions(
    const int T, const int N,
    __global const float *alpha_full, 
    __global const float *beta_full,  
    __global const float *emis_full,  
    __global const float *trans_mat,  
    __global float *new_trans_counts) 
{
    int row = get_global_id(1); // From State i
    int col = get_global_id(0); // To State j

    if (row < N && col < N) {
        float log_sum_xi = -INFINITY;
        float log_trans_val = trans_mat[row * N + col];

        // Loop over time 0 to T-2
        for (int t = 0; t < T - 1; t++) {
            float log_xi = alpha_full[t*N + row] + 
                           log_trans_val + 
                           emis_full[(t+1)*N + col] + 
                           beta_full[(t+1)*N + col];
            
            if (t == 0) log_sum_xi = log_xi;
            else log_sum_xi = log_add(log_sum_xi, log_xi);
        }
        new_trans_counts[row * N + col] = log_sum_xi;
    }
}

// 7. Accumulate Gammas (E-Step)
// Condenses time T into N summary counts
__kernel void accumulate_gammas(
    const int T, const int N,
    __global const float *alpha_full,
    __global const float *beta_full,
    __global float *log_gamma_sums) 
{
    int id = get_global_id(0);
    if (id < N) {
        float log_sum_gamma = -INFINITY;
        for (int t = 0; t < T; t++) {
            float val = alpha_full[t*N + id] + beta_full[t*N + id];
            if (t == 0) log_sum_gamma = val;
            else log_sum_gamma = log_add(log_sum_gamma, val);
        }
        log_gamma_sums[id] = log_sum_gamma;
    }
}