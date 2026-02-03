// kernels.cl - Fixed Write Permissions

// --- HELPER: Log-Sum-Exp Trick ---
float log_add(float log_a, float log_b) {
    float max_val = max(log_a, log_b);
    float min_val = min(log_a, log_b);
    return max_val + log1p(exp(min_val - max_val));
}

// --- SECTION 1: Basic Operations ---

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

// --- SECTION 2: Advanced Log-Space Operations ---

// 3. Log-Space Forward (FIXED: Removed 'const' from log_alpha_full)
__kernel void hmm_forward_log(
    const int N,
    __global float *log_alpha_full,        // <--- FIX: Removed 'const' here
    const int prev_offset,
    const int curr_offset,
    __global const float *log_trans_mat,
    __global const float *log_emissions,
    const int emis_offset)
{
    int id = get_global_id(0);
    if (id < N) {
        float log_sum = -INFINITY;
        // Read from 'prev_offset' in the giant buffer
        for (int k = 0; k < N; k++) {
            float val = log_alpha_full[prev_offset + k] + log_trans_mat[k * N + id];
            if (k == 0) log_sum = val;
            else log_sum = log_add(log_sum, val);
        }
        // Write to 'curr_offset'
        // Read emission from 'emis_offset'
        log_alpha_full[curr_offset + id] = log_sum + log_emissions[emis_offset + id];
    }
}

// 4. Log-Space Backward
__kernel void hmm_backward_log(
    const int N,
    __global float *beta_full,            
    const int future_offset,              
    const int curr_offset,                
    __global const float *trans,
    __global const float *emis_full,      
    const int future_emis_offset)         
{
    int id = get_global_id(0); // State 'i'
    if (id < N) {
        float log_sum = -INFINITY;
        for (int j=0; j<N; j++) {
            // trans(i->j) + emis(t+1, j) + beta(t+1, j)
            float val = trans[id*N + j] + 
                        emis_full[future_emis_offset + j] + 
                        beta_full[future_offset + j];
            
            if (j==0) log_sum = val;
            else log_sum = log_add(log_sum, val);
        }
        beta_full[curr_offset + id] = log_sum;
    }
}

// 5. Viterbi Algorithm
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

// --- SECTION 3: Learning Accumulators ---

// 6. Accumulate Transitions (E-Step)
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