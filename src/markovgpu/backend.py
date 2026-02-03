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
                best_dev = sorted(
                    gpu_devices, key=lambda d: d.max_compute_units, reverse=True
                )[0]
                self.ctx = cl.Context([best_dev])
                print(
                    f"🔌 Connected to Accelerator: {best_dev.name} ({best_dev.max_compute_units} CUs)"
                )
            else:
                self.ctx = cl.create_some_context(interactive=False)
                print(f"⚠️ No Dedicated GPU found. Using: {self.ctx.devices[0].name}")

            self.queue = cl.CommandQueue(self.ctx)

            # 2. Compile Kernels
            if not os.path.exists(KERNEL_PATH):
                raise FileNotFoundError(f"Kernel file missing at: {KERNEL_PATH}")

            # OPTIMIZATION: Fast Math Build Options
            build_options = "-cl-mad-enable -cl-fast-relaxed-math"

            with open(KERNEL_PATH, "r") as f:
                self.prg = cl.Program(self.ctx, f.read()).build(options=build_options)

            # 3. Cache Kernels (Robust Retrieval)
            self.use_gpu = True
            try:
                # Basic
                self.k_markov = self.prg.markov_step
                self.k_hmm_basic = self.prg.hmm_forward_step

                # Advanced / Viterbi
                self.k_hmm_log = self.prg.hmm_forward_log
                self.k_viterbi = self.prg.viterbi_step

                # Training
                self.k_hmm_back = self.prg.hmm_backward_log
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
        # OPTIMIZATION: Transpose P for coalesced access
        # The kernel expects P_T[id][k] which maps to P[k][id]
        P_T = np.ascontiguousarray(P.T, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)
        result = np.empty_like(v)

        d_P_T = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=P_T)
        d_v = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v)
        d_res = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=result.nbytes)

        self.k_markov(self.queue, (N,), None, np.int32(N), d_v, d_P_T, d_res)
        cl.enqueue_copy(self.queue, result, d_res)
        return result

    def converge(self, P, start_v, tolerance=1e-5, max_steps=1000):
        # Note: 'converge' currently uses the iterative step approach.
        # For maximum optimization, this loop should ideally be moved to a kernel,
        # but for now, we rely on the optimized 'step' logic implicitly or CPU fallback.
        # Below is the robust hybrid implementation.
        N = len(start_v)

        # CPU Path
        if not self.use_gpu or N < GPU_THRESHOLD:
            current_v = start_v.copy()
            for i in range(max_steps):
                new_v = current_v.dot(P)
                if np.sum(np.abs(new_v - current_v)) < tolerance:
                    return new_v
                current_v = new_v
            return current_v

        # GPU Path
        # We reuse the specific buffers to avoid reallocation overhead in loop
        mf = cl.mem_flags
        P_T = np.ascontiguousarray(P.T, dtype=np.float32)
        start_v = np.ascontiguousarray(start_v, dtype=np.float32)

        d_P_T = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=P_T)
        d_v_read = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=start_v)
        d_v_write = cl.Buffer(self.ctx, mf.READ_WRITE, size=start_v.nbytes)

        current_v = start_v.copy()

        for i in range(max_steps):
            # Use k_markov with Transposed Matrix
            self.k_markov(self.queue, (N,), None, np.int32(N), d_v_read, d_P_T, d_v_write)

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
                    vals = log_delta[t - 1] + log_trans[:, j]
                    best_prev = np.argmax(vals)
                    backpointers[t, j] = best_prev
                    log_delta[t, j] = vals[best_prev] + log_emis[t, j]

            path = np.zeros(T, dtype=int)
            path[-1] = np.argmax(log_delta[-1])
            for t in range(T - 2, -1, -1):
                path[t] = backpointers[t + 1, path[t + 1]]
            return path

        # GPU Path
        mf = cl.mem_flags
        # OPTIMIZATION: Transpose Log-Transition Matrix
        log_trans = np.log(transition_matrix + epsilon).astype(np.float32)
        log_trans_T = np.ascontiguousarray(log_trans.T, dtype=np.float32)
        
        log_emis = np.log(observation_probs + epsilon).astype(np.float32)
        log_delta = np.full(N, -np.log(N), dtype=np.float32)

        d_trans_T = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=log_trans_T)
        d_delta_in = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=log_delta)
        d_delta_out = cl.Buffer(self.ctx, mf.READ_WRITE, size=log_delta.nbytes)

        full_backpointer_history = np.zeros((T, N), dtype=np.int32)
        d_backpointers = cl.Buffer(
            self.ctx, mf.WRITE_ONLY, size=full_backpointer_history.nbytes // T
        )

        print(f"🕵️ Decoding {T} days (GPU Optimized)...")

        for t in range(T):
            d_emis = cl.Buffer(
                self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=log_emis[t]
            )

            self.k_viterbi(
                self.queue,
                (N,),
                None,
                np.int32(N),
                d_delta_in,
                d_trans_T, # Pass Transposed Matrix
                d_emis,
                d_delta_out,
                d_backpointers,
            )

            step_pointers = np.empty(N, dtype=np.int32)
            cl.enqueue_copy(self.queue, step_pointers, d_backpointers)
            full_backpointer_history[t] = step_pointers

            d_delta_in, d_delta_out = d_delta_out, d_delta_in

        final_log_probs = np.empty(N, dtype=np.float32)
        cl.enqueue_copy(self.queue, final_log_probs, d_delta_in)

        best_path = np.zeros(T, dtype=np.int32)
        best_path[-1] = np.argmax(final_log_probs)

        for t in range(T - 2, -1, -1):
            next_state = best_path[t + 1]
            best_path[t] = full_backpointer_history[t + 1][next_state]

        return best_path

    # --- 3. Training (Baum-Welch) ---
    def fit(self, observations, n_states, n_iters=10, tolerance=1e-4):
        """Baum-Welch Expectation Maximization (Training)"""
        T = observations.shape[0]
        N = n_states
        mf = cl.mem_flags

        # 1. Initialize Params (Log Space)
        log_trans = np.log(
            np.full((N, N), 1.0 / N) + np.random.rand(N, N) * 0.01
        ).astype(np.float32)
        log_emis = np.log(observations + 1e-20).astype(np.float32)

        # 2. Allocate GPU Memory (VRAM)
        # We need TWO transition buffers for optimization:
        # A. Original (Row-Major) for Backward Pass & Accumulation
        # B. Transposed (Col-Major) for Forward Pass
        d_trans = cl.Buffer(self.ctx, mf.READ_WRITE, size=log_trans.nbytes)
        d_trans_T = cl.Buffer(self.ctx, mf.READ_WRITE, size=log_trans.nbytes)
        
        # Initial Copy
        cl.enqueue_copy(self.queue, d_trans, log_trans)
        cl.enqueue_copy(self.queue, d_trans_T, np.ascontiguousarray(log_trans.T))

        d_emis = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=log_emis)
        
        d_alpha = cl.Buffer(self.ctx, mf.READ_WRITE, size=T * N * 4) # float32 = 4 bytes
        d_beta = cl.Buffer(self.ctx, mf.READ_WRITE, size=T * N * 4)
        
        d_new_trans = cl.Buffer(self.ctx, mf.READ_WRITE, size=log_trans.nbytes)
        d_gamma_sums = cl.Buffer(self.ctx, mf.READ_WRITE, size=N * 4)

        prev_score = -np.inf

        print(f"🧠 Training HMM ({N} States, {T} Steps) on GPU...")

        # Host buffers for initial checks and final readback
        init_alpha = np.zeros(N, dtype=np.float32)
        final_alpha_T = np.zeros(N, dtype=np.float32)

        for i in range(n_iters):
            
            # --- A. Forward Pass (GPU Loop) ---
            # Uses Transposed Matrix (d_trans_T) for coalesced reads
            init_alpha[:] = -np.log(N) + log_emis[0]
            cl.enqueue_copy(self.queue, d_alpha, init_alpha, is_blocking=False)

            for t in range(1, T):
                prev_offset = (t - 1) * N
                curr_offset = t * N
                emis_offset = t * N
                
                self.k_hmm_log(
                    self.queue, (N,), None,
                    np.int32(N),
                    d_alpha,            
                    np.int32(prev_offset), 
                    np.int32(curr_offset),
                    d_trans_T, # <--- Optimized Read
                    d_emis,             
                    np.int32(emis_offset)
                )

            # --- B. Backward Pass (GPU Loop) ---
            # Uses Original Matrix (d_trans) because Backward pass logic matches Row-Major
            init_beta_end = np.zeros(N, dtype=np.float32) 
            beta_end_offset = (T - 1) * N * 4 
            cl.enqueue_copy(self.queue, d_beta, init_beta_end, dst_offset=beta_end_offset, is_blocking=False)

            for t in range(T - 2, -1, -1):
                curr_offset = t * N
                future_offset = (t + 1) * N
                future_emis_offset = (t + 1) * N

                self.k_hmm_back(
                    self.queue, (N,), None,
                    np.int32(N),
                    d_beta,            
                    np.int32(future_offset),
                    np.int32(curr_offset),
                    d_trans, # <--- Optimized Read (Backward needs Row-Major)
                    d_emis,
                    np.int32(future_emis_offset)
                )

            # --- C. Accumulation (GPU) ---
            self.queue.finish()

            self.k_acc_trans(
                self.queue, (N, N), None,
                np.int32(T), np.int32(N),
                d_alpha, d_beta, d_emis, d_trans, d_new_trans
            )

            self.k_acc_gamma(
                self.queue, (N,), None,
                np.int32(T), np.int32(N),
                d_alpha, d_beta, d_gamma_sums
            )

            # --- D. Update & Check Convergence (CPU) ---
            new_log_trans_counts = np.empty_like(log_trans)
            log_gamma_sums = np.empty(N, dtype=np.float32)

            cl.enqueue_copy(self.queue, new_log_trans_counts, d_new_trans)
            cl.enqueue_copy(self.queue, log_gamma_sums, d_gamma_sums)
            
            # Calc Likelihood
            alpha_T_offset = (T - 1) * N * 4
            cl.enqueue_copy(self.queue, final_alpha_T, d_alpha, src_offset=alpha_T_offset)
            log_likelihood = np.logaddexp.reduce(final_alpha_T)

            # M-Step: Normalize
            log_trans = new_log_trans_counts - log_gamma_sums[:, None]
            
            # Update BOTH GPU Buffers for next iteration
            cl.enqueue_copy(self.queue, d_trans, log_trans)
            cl.enqueue_copy(self.queue, d_trans_T, np.ascontiguousarray(log_trans.T))

            change = log_likelihood - prev_score
            print(f"   Iter {i + 1}: Likelihood {log_likelihood:.2f} (Delta: {change:.4f})")
            
            if abs(change) < tolerance:
                break
            prev_score = log_likelihood

        return np.exp(log_trans)