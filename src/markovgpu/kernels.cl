// kernels.cl - Memory Optimized (Transposed Access) + Fixed Write Permissions

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
    __global const float *trans_mat_T, // EXPECTS TRANSPOSED MATRIX
    __global float *next_state)
{
    int id = get_global_id(0); // Target State (Row in Transposed Mat)
    if (id < N) {
        float sum = 0.0f;
        int row_start = id * N; // Coalesced Start (Optimization)
        
        for (int k = 0; k < N; k++) {
            // Read sequentially: P_T[id][k] corresponds to P[k][id]
            sum += current_state[k] * trans_mat_T[row_start + k];
        }
        next_state[id] = sum;
    }
}

__kernel void hmm_forward_step(
    const int N,
    __global const float *alpha_prev,
    __global const float *trans_mat_T, // EXPECTS TRANSPOSED MATRIX
    __global const float *emissions,
    __global float *alpha_new)
{
    int id = get_global_id(0);
    if (id < N) {
        float sum = 0.0f;
        int row_start = id * N; 

        for (int k = 0; k < N; k++) {
            sum += alpha_prev[k] * trans_mat_T[row_start + k];
        }
        alpha_new[id] = sum * emissions[id];
    }
}

// --- SECTION 2: Advanced Log-Space Operations ---

// 3. Log-Space Forward (Memory Optimized)
__kernel void hmm_forward_log(
    const int N,
    __global float *log_alpha_full,        // NO CONST (Write Permission Fix Preserved)
    const int prev_offset,
    const int curr_offset,
    __global const float *log_trans_mat_T, // EXPECTS TRANSPOSED MATRIX
    __global const float *log_emissions,
    const int emis_offset)
{
    int id = get_global_id(0); // Target State (Row in Transposed Mat)
    if (id < N) {
        float log_sum = -INFINITY;
        int row_start = id * N; 
        
        // Loop 'k' (Previous State). 
        // In Transposed Matrix, 'id' is the Row, 'k' is the Column.
        // So we read P_T[id][k] which corresponds to P[k][id]
        for (int k = 0; k < N; k++) {
            float val = log_alpha_full[prev_offset + k] + log_trans_mat_T[row_start + k];
            if (k == 0) log_sum = val;
            else log_sum = log_add(log_sum, val);
        }
        
        // Write to 'curr_offset'
        log_alpha_full[curr_offset + id] = log_sum + log_emissions[emis_offset + id];
    }
}

// 4. Log-Space Backward (Memory Optimized - Uses ORIGINAL Matrix)
// Note: Backward pass needs P[i][j], which is naturally Row-Major.
// So we DO NOT use the Transposed matrix here. It is already optimized!
__kernel void hmm_backward_log(
    const int N,
    __global float *beta_full,            
    const int future_offset,              
    const int curr_offset,                
    __global const float *trans, // ORIGINAL MATRIX (Row-Major)
    __global const float *emis_full,      
    const int future_emis_offset)         
{
    int id = get_global_id(0); // State 'i'
    if (id < N) {
        float log_sum = -INFINITY;
        int row_start = id * N;

        for (int j=0; j<N; j++) {
            // Read sequentially: trans[row_start + j]
            float val = trans[row_start + j] + 
                        emis_full[future_emis_offset + j] + 
                        beta_full[future_offset + j];
            
            if (j==0) log_sum = val;
            else log_sum = log_add(log_sum, val);
        }
        beta_full[curr_offset + id] = log_sum;
    }
}

// 5. Viterbi Algorithm (Memory Optimized)
__kernel void viterbi_step(
    const int N,
    __global const float *log_delta_prev,
    __global const float *log_trans_mat_T, // EXPECTS TRANSPOSED MATRIX
    __global const float *log_emissions,
    __global float *log_delta_new,
    __global int *backpointers)
{
    int id = get_global_id(0);
    if (id < N) {
        float max_prob = -INFINITY;
        int best_prev_state = 0;
        int row_start = id * N;

        for (int k = 0; k < N; k++) {
            // Read sequentially: P_T[id][k]
            float prob = log_delta_prev[k] + log_trans_mat_T[row_start + k];
            if (prob > max_prob) {
                max_prob = prob;
                best_prev_state = k;
            }
        }
        log_delta_new[id] = max_prob + log_emissions[id];
        backpointers[id] = best_prev_state;
    }
}

// --- SECTION 3: Learning Accumulators (Unchanged) ---

// 6. Accumulate Transitions (E-Step)
__kernel void accumulate_transitions(
    const int T, const int N,
    __global const float *alpha_full, 
    __global const float *beta_full,  
    __global const float *emis_full,  
    __global const float *trans_mat, // Original Matrix
    __global float *new_trans_counts) 
{
    int row = get_global_id(1); 
    int col = get_global_id(0); 

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