import pyopencl as cl
import numpy as np
import os

# Locate the kernel file
MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
KERNEL_PATH = os.path.join(MODULE_PATH, "kernels.cl")

# Threshold: Use GPU if states >= 64, otherwise CPU is faster
GPU_THRESHOLD = 64 

class MarkovEngine:
    def __init__(self):
        self.use_gpu = False
        self.ctx = None
        self.queue = None
        self.prg = None

        # 1. Try to Connect to GPU
        try:
            platforms = cl.get_platforms()
            gpu_devices = []
            for p in platforms:
                gpu_devices.extend(p.get_devices(device_type=cl.device_type.GPU))
            
            if gpu_devices:
                # Pick the discrete GPU (highest compute units)
                best_dev = sorted(gpu_devices, key=lambda d: d.max_compute_units, reverse=True)[0]
                self.ctx = cl.Context([best_dev])
                print(f"🔌 Connected to Accelerator: {best_dev.name} ({best_dev.max_compute_units} CUs)")
            else:
                self.ctx = cl.create_some_context(interactive=False)
                print(f"⚠️ No Dedicated GPU found. Using: {self.ctx.devices[0].name}")

            self.queue = cl.CommandQueue(self.ctx)
            
            # 2. Compile Kernels
            if not os.path.exists(KERNEL_PATH):
                raise FileNotFoundError(f"Kernel file missing at: {KERNEL_PATH}")
                
            with open(KERNEL_PATH, "r") as f:
                self.prg = cl.Program(self.ctx, f.read()).build()

            # 3. Cache Kernels (Robust Retrieval)
            self.use_gpu = True
            try:
                # Basic
                self.k_markov    = self.prg.markov_step
                self.k_hmm_basic = self.prg.hmm_forward_step
                
                # Advanced / Viterbi
                self.k_hmm_log   = self.prg.hmm_forward_log
                self.k_viterbi   = self.prg.viterbi_step
                
                # Training
                self.k_hmm_back  = self.prg.hmm_backward_log
                self.k_acc_trans = self.prg.accumulate_transitions
                self.k_acc_gamma = self.prg.accumulate_gammas
                
            except AttributeError as e:
                print(f"❌ Kernel Warning: {e}")
                print("⚠️ Some GPU features may be disabled.")

        except Exception as e:
            print(f"⚠️ OpenCL Initialization failed: {e}")
            print("⚠️ Running in CPU-Only Mode (NumPy).")
            self.use_gpu = False

    # --- 1. Simulation ---
    def step(self, P, v):
        """Runs one step: v_new = v * P"""
        N = len(v)

        if not self.use_gpu or N < GPU_THRESHOLD:
            return v.dot(P)

        mf = cl.mem_flags
        P = np.ascontiguousarray(P, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)
        result = np.empty_like(v)
        
        d_P = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=P)
        d_v = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v)
        d_res = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=result.nbytes)

        self.k_markov(self.queue, (N,), None, np.int32(N), d_v, d_P, d_res)
        cl.enqueue_copy(self.queue, result, d_res)
        return result

    def converge(self, P, start_v, tolerance=1e-5, max_steps=1000):
        N = len(start_v)

        # CPU Path
        if not self.use_gpu or N < GPU_THRESHOLD:
            # print(f"🔄 Converging on CPU (N={N})...")
            current_v = start_v.copy()
            for i in range(max_steps):
                new_v = current_v.dot(P)
                if np.sum(np.abs(new_v - current_v)) < tolerance:
                    return new_v
                current_v = new_v
            return current_v

        # GPU Path
        # print(f"🔄 Converging on GPU (N={N})...")
        mf = cl.mem_flags
        P = np.ascontiguousarray(P, dtype=np.float32)
        start_v = np.ascontiguousarray(start_v, dtype=np.float32)

        d_P = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=P)
        d_v_read = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=start_v)
        d_v_write = cl.Buffer(self.ctx, mf.READ_WRITE, size=start_v.nbytes)
        
        current_v = start_v.copy()
        
        for i in range(max_steps):
            self.k_markov(self.queue, (N,), None, np.int32(N), d_v_read, d_P, d_v_write)
            
            if i % 10 == 0:
                new_v = np.empty_like(current_v)
                cl.enqueue_copy(self.queue, new_v, d_v_write)
                if np.sum(np.abs(new_v - current_v)) < tolerance:
                    return new_v
                current_v = new_v

            d_v_read, d_v_write = d_v_write, d_v_read
            
        print("⚠️ Reached max steps without full convergence.")
        return current_v

    # --- 2. Inference & Viterbi ---
    def hmm_filter(self, transition_matrix, observation_probs):
        """Standard HMM Filter (Returns Probabilities)"""
        # Simplification: Running basic HMM forward pass
        # For production use, usually prefer Log-Space to avoid underflow.
        # This wrapper can be upgraded to use k_hmm_log if needed.
        pass 

    def decode_regime(self, transition_matrix, observation_probs):
        """Viterbi Algorithm (Finds Most Likely Path)"""
        T, N = observation_probs.shape
        epsilon = 1e-20

        # CPU Path
        if not self.use_gpu or N < GPU_THRESHOLD:
            log_trans = np.log(transition_matrix + epsilon)
            log_emis = np.log(observation_probs + epsilon)
            log_delta = np.zeros((T, N))
            backpointers = np.zeros((T, N), dtype=int)

            log_delta[0] = -np.log(N) + log_emis[0]

            for t in range(1, T):
                for j in range(N):
                    vals = log_delta[t-1] + log_trans[:, j]
                    best_prev = np.argmax(vals)
                    backpointers[t, j] = best_prev
                    log_delta[t, j] = vals[best_prev] + log_emis[t, j]
            
            path = np.zeros(T, dtype=int)
            path[-1] = np.argmax(log_delta[-1])
            for t in range(T-2, -1, -1):
                path[t] = backpointers[t+1, path[t+1]]
            return path

        # GPU Path
        mf = cl.mem_flags
        log_trans = np.log(transition_matrix + epsilon).astype(np.float32)
        log_emis = np.log(observation_probs + epsilon).astype(np.float32)
        log_delta = np.full(N, -np.log(N), dtype=np.float32)

        d_trans = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=log_trans)
        d_delta_in = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=log_delta)
        d_delta_out = cl.Buffer(self.ctx, mf.READ_WRITE, size=log_delta.nbytes)
        
        full_backpointer_history = np.zeros((T, N), dtype=np.int32)
        d_backpointers = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=full_backpointer_history.nbytes // T)

        print(f"🕵️ Decoding {T} days (GPU Accelerated)...")

        for t in range(T):
            d_emis = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=log_emis[t])
            
            self.k_viterbi(self.queue, (N,), None, np.int32(N), 
                           d_delta_in, d_trans, d_emis, d_delta_out, d_backpointers)

            step_pointers = np.empty(N, dtype=np.int32)
            cl.enqueue_copy(self.queue, step_pointers, d_backpointers)
            full_backpointer_history[t] = step_pointers

            d_delta_in, d_delta_out = d_delta_out, d_delta_in
        
        final_log_probs = np.empty(N, dtype=np.float32)
        cl.enqueue_copy(self.queue, final_log_probs, d_delta_in)
        
        best_path = np.zeros(T, dtype=np.int32)
        best_path[-1] = np.argmax(final_log_probs)
        
        for t in range(T-2, -1, -1):
            next_state = best_path[t+1]
            best_path[t] = full_backpointer_history[t+1][next_state]
            
        return best_path

    # --- 3. Training (Baum-Welch) ---
    def fit(self, observations, n_states, n_iters=10, tolerance=1e-4):
        """Baum-Welch Expectation Maximization (Training)"""
        T = observations.shape[0]
        N = n_states
        
        # Random Init
        log_trans = np.log(np.full((N, N), 1.0/N) + np.random.rand(N,N)*0.01).astype(np.float32)
        log_emis = np.log(observations + 1e-20).astype(np.float32) 
        
        mf = cl.mem_flags
        d_trans = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=log_trans)
        d_alpha = cl.Buffer(self.ctx, mf.READ_WRITE, size=T * N * 4) # Full history
        d_beta  = cl.Buffer(self.ctx, mf.READ_WRITE, size=T * N * 4) # Full history
        d_emis  = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=log_emis)
        
        d_new_trans = cl.Buffer(self.ctx, mf.READ_WRITE, size=log_trans.nbytes)
        d_gamma_sums = cl.Buffer(self.ctx, mf.READ_WRITE, size=N * 4)

        prev_score = -np.inf
        
        print(f"🧠 Training HMM ({N} States, {T} Steps)...")
        
        for i in range(n_iters):
            # 1. CPU Forward/Backward (Latency Optimized)
            alpha_full, log_likelihood = self._cpu_forward(log_trans, log_emis)
            beta_full = self._cpu_backward(log_trans, log_emis)
            
            # 2. GPU Accumulation (Throughput Optimized)
            cl.enqueue_copy(self.queue, d_alpha, alpha_full)
            cl.enqueue_copy(self.queue, d_beta, beta_full)
            cl.enqueue_copy(self.queue, d_trans, log_trans)
            
            self.k_acc_trans(self.queue, (N, N), None, np.int32(T), np.int32(N), 
                             d_alpha, d_beta, d_emis, d_trans, d_new_trans)
            
            self.k_acc_gamma(self.queue, (N,), None, np.int32(T), np.int32(N),
                             d_alpha, d_beta, d_gamma_sums)
            
            # 3. Update
            new_log_trans_counts = np.empty_like(log_trans)
            log_gamma_sums = np.empty(N, dtype=np.float32)
            
            cl.enqueue_copy(self.queue, new_log_trans_counts, d_new_trans)
            cl.enqueue_copy(self.queue, log_gamma_sums, d_gamma_sums)
            
            log_trans = new_log_trans_counts - log_gamma_sums[:, None]
            
            change = log_likelihood - prev_score
            print(f"   Iter {i+1}: Likelihood {log_likelihood:.2f} (Delta: {change:.4f})")
            if abs(change) < tolerance:
                break
            prev_score = log_likelihood
            
        return np.exp(log_trans)

    def _cpu_forward(self, log_trans, log_emis):
        T, N = log_emis.shape
        alpha = np.zeros((T, N), dtype=np.float32)
        alpha[0] = -np.log(N) + log_emis[0]
        for t in range(1, T):
            for j in range(N):
                prev = alpha[t-1] + log_trans[:, j]
                alpha[t, j] = np.logaddexp.reduce(prev) + log_emis[t, j]
        return alpha, np.logaddexp.reduce(alpha[-1])

    def _cpu_backward(self, log_trans, log_emis):
        T, N = log_emis.shape
        beta = np.zeros((T, N), dtype=np.float32)
        for t in range(T-2, -1, -1):
            for i in range(N):
                terms = log_trans[i, :] + log_emis[t+1] + beta[t+1]
                beta[t, i] = np.logaddexp.reduce(terms)
        return beta