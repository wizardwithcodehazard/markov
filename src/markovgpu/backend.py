import pyopencl as cl
import numpy as np
import os

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
KERNEL_PATH = os.path.join(MODULE_PATH, "kernels.cl")
GPU_THRESHOLD = 64


class MarkovEngine:
    def __init__(self):
        self.use_gpu = False
        self.ctx = None
        self.queue = None
        self.prg = None
        try:
            platforms = cl.get_platforms()
            gpu_devices = []
            for p in platforms:
                gpu_devices.extend(p.get_devices(device_type=cl.device_type.GPU))

            if gpu_devices:
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
            if not os.path.exists(KERNEL_PATH):
                raise FileNotFoundError(f"Kernel file missing at: {KERNEL_PATH}")

            with open(KERNEL_PATH, "r") as f:
                self.prg = cl.Program(self.ctx, f.read()).build()

            self.use_gpu = True
            try:
                self.k_markov = self.prg.markov_step
                self.k_hmm_basic = self.prg.hmm_forward_step
                self.k_hmm_log = self.prg.hmm_forward_log
                self.k_viterbi = self.prg.viterbi_step_optimized
                self.k_hmm_back = self.prg.hmm_backward_log
                self.k_acc_trans = self.prg.accumulate_transitions
                self.k_acc_gamma = self.prg.accumulate_gammas
            except AttributeError as e:
                print(f"❌ Kernel Warning: {e}")
                print("⚠️ Some GPU features may be disabled.")

        except Exception as e:
            print(f"⚠️ OpenCL Initialization failed: {e}")
            self.use_gpu = False

    def step(self, P, v):
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

    def decode_regime(self, transition_matrix, observation_probs):
        """Viterbi Algorithm"""
        T, N = observation_probs.shape
        epsilon = 1e-20

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

        mf = cl.mem_flags
        log_trans = np.log(transition_matrix + epsilon).astype(np.float32)
        log_emis_full = np.log(observation_probs + epsilon).astype(np.float32).ravel()
        log_delta = np.full(N, -np.log(N), dtype=np.float32)

        d_trans = cl.Buffer(
            self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=log_trans
        )
        d_all_emis = cl.Buffer(
            self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=log_emis_full
        )
        d_delta_in = cl.Buffer(
            self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=log_delta
        )
        d_delta_out = cl.Buffer(self.ctx, mf.READ_WRITE, size=log_delta.nbytes)
        d_all_backpointers = cl.Buffer(self.ctx, mf.READ_WRITE, size=T * N * 4)

        for t in range(1, T):
            self.k_viterbi(
                self.queue,
                (N,),
                None,
                np.int32(N),
                np.int32(t),
                np.int32(T),
                d_delta_in,
                d_trans,
                d_all_emis,
                d_delta_out,
                d_all_backpointers,
            )
            d_delta_in, d_delta_out = d_delta_out, d_delta_in

        full_backpointer_history = np.empty((T, N), dtype=np.int32)
        final_log_probs = np.empty(N, dtype=np.float32)
        cl.enqueue_copy(self.queue, full_backpointer_history, d_all_backpointers)
        cl.enqueue_copy(self.queue, final_log_probs, d_delta_in)

        best_path = np.zeros(T, dtype=np.int32)
        best_path[-1] = np.argmax(final_log_probs)
        for t in range(T - 1, 0, -1):
            next_state = best_path[t]
            best_path[t - 1] = full_backpointer_history[t][next_state]

        return best_path

    def fit(self, observations, n_states, n_iters=10, tolerance=1e-4):
        T = observations.shape[0]
        N = n_states

        # --- 1. CPU Fallback (The Missing Piece) ---
        if not self.use_gpu or N < GPU_THRESHOLD:
            print(f"🐢 Training HMM ({N} States) on CPU (NumPy)...")
            # Random Init
            log_trans = np.log(np.full((N, N), 1.0 / N) + np.random.rand(N, N) * 0.01)
            log_emis = np.log(observations + 1e-20)
            prev_score = -np.inf

            for i in range(n_iters):
                # Pure NumPy Forward-Backward
                alpha, log_likelihood = self._cpu_forward(log_trans, log_emis)
                beta = self._cpu_backward(log_trans, log_emis)

                # E-Step: Accumulate (Standard Baum-Welch)
                log_xi_sum = np.full((N, N), -np.inf)
                log_gamma_sum = np.full(N, -np.inf)

                # Vectorized Accumulation (Optimized NumPy)
                for t in range(T - 1):
                    # log_xi = alpha[t] + trans + emis[t+1] + beta[t+1]
                    temp = alpha[t][:, None] + log_trans + log_emis[t + 1] + beta[t + 1]
                    log_xi_sum = np.logaddexp(log_xi_sum, temp)

                # Gamma is just Alpha + Beta
                log_gamma = alpha + beta
                log_gamma_sum = np.logaddexp.reduce(log_gamma[:-1], axis=0)

                # M-Step
                log_trans = log_xi_sum - log_gamma_sum[:, None]

                change = log_likelihood - prev_score
                print(
                    f"   Iter {i + 1}: Likelihood {log_likelihood:.2f} (Delta: {change:.4f})"
                )
                if abs(change) < tolerance:
                    break
                prev_score = log_likelihood
            return np.exp(log_trans)

        # --- 2. GPU Path (Large Models) ---
        mf = cl.mem_flags
        log_trans = np.log(
            np.full((N, N), 1.0 / N) + np.random.rand(N, N) * 0.01
        ).astype(np.float32)
        log_emis_full = np.log(observations + 1e-20).astype(np.float32).ravel()

        d_trans = cl.Buffer(
            self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=log_trans
        )
        d_all_emis = cl.Buffer(
            self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=log_emis_full
        )
        d_alpha = cl.Buffer(self.ctx, mf.READ_WRITE, size=T * N * 4)
        d_beta = cl.Buffer(self.ctx, mf.READ_WRITE, size=T * N * 4)
        d_new_trans = cl.Buffer(self.ctx, mf.READ_WRITE, size=log_trans.nbytes)
        d_gamma_sums = cl.Buffer(self.ctx, mf.READ_WRITE, size=N * 4)

        prev_score = -np.inf
        print(f"🧠 Training HMM ({N} States, {T} Steps) on GPU...")

        for i in range(n_iters):
            log_likelihood = self._gpu_forward(d_trans, d_all_emis, d_alpha, T, N)
            self._gpu_backward(d_trans, d_all_emis, d_beta, T, N)

            self.k_acc_trans(
                self.queue,
                (N, N),
                None,
                np.int32(T),
                np.int32(N),
                d_alpha,
                d_beta,
                d_all_emis,
                d_trans,
                d_new_trans,
            )

            self.k_acc_gamma(
                self.queue,
                (N,),
                None,
                np.int32(T),
                np.int32(N),
                d_alpha,
                d_beta,
                d_gamma_sums,
            )

            new_log_trans = np.empty_like(log_trans)
            log_gamma = np.empty(N, dtype=np.float32)
            cl.enqueue_copy(self.queue, new_log_trans, d_new_trans)
            cl.enqueue_copy(self.queue, log_gamma, d_gamma_sums)

            log_trans = new_log_trans - log_gamma[:, None]
            cl.enqueue_copy(self.queue, d_trans, log_trans)

            change = log_likelihood - prev_score
            print(
                f"   Iter {i + 1}: Likelihood {log_likelihood:.2f} (Delta: {change:.4f})"
            )
            if abs(change) < tolerance:
                break
            prev_score = log_likelihood

        return np.exp(log_trans)

    def _gpu_forward(self, d_trans, d_all_emis, d_alpha, T, N):
        """Runs Forward Algorithm purely in GPU memory (Alignment Safe)"""
        # 1. Init t=0
        emis_0 = np.empty(N, dtype=np.float32)
        cl.enqueue_copy(self.queue, emis_0, d_all_emis)
        alpha_0 = -np.log(N) + emis_0
        cl.enqueue_copy(self.queue, d_alpha, alpha_0)

        # 2. Loop with INTEGER OFFSETS
        for t in range(1, T):
            off_prev = (t - 1) * N
            off_curr = t * N
            off_emis = t * N

            self.k_hmm_log(
                self.queue,
                (N,),
                None,
                np.int32(N),
                np.int32(off_prev),
                np.int32(off_emis),
                np.int32(off_curr),
                d_alpha,  # Pass FULL buffer
                d_trans,
                d_all_emis,  # Pass FULL buffer
                d_alpha,  # Pass FULL buffer
            )

        # 3. Retrieve Result
        alpha_T = np.empty(N, dtype=np.float32)
        cl.enqueue_copy(self.queue, alpha_T, d_alpha, src_offset=(T - 1) * N * 4)
        return np.logaddexp.reduce(alpha_T)

    def _gpu_backward(self, d_trans, d_all_emis, d_beta, T, N):
        """Runs Backward Algorithm purely in GPU memory (Alignment Safe)"""
        # t=T-1: Initialize with 0.0 (log(1))
        beta_T = np.zeros(N, dtype=np.float32)
        cl.enqueue_copy(self.queue, d_beta, beta_T, dst_offset=(T - 1) * N * 4)

        # Loop t=T-2 down to 0
        for t in range(T - 2, -1, -1):
            off_curr = t * N
            off_future = (t + 1) * N
            off_emis = (t + 1) * N

            self.k_hmm_back(
                self.queue,
                (N,),
                None,
                np.int32(N),
                np.int32(off_future),
                np.int32(off_emis),
                np.int32(off_curr),
                d_beta,  # Pass FULL buffer
                d_trans,
                d_all_emis,  # Pass FULL buffer
                d_beta,  # Pass FULL buffer
            )

    def _cpu_forward(self, log_trans, log_emis):
        """Standard Forward Algorithm (NumPy)"""
        T, N = log_emis.shape
        alpha = np.zeros((T, N))
        alpha[0] = -np.log(N) + log_emis[0]
        for t in range(1, T):
            for j in range(N):
                prev = alpha[t - 1] + log_trans[:, j]
                alpha[t, j] = np.logaddexp.reduce(prev) + log_emis[t, j]
        return alpha, np.logaddexp.reduce(alpha[-1])

    def _cpu_backward(self, log_trans, log_emis):
        """Standard Backward Algorithm (NumPy)"""
        T, N = log_emis.shape
        beta = np.zeros((T, N))
        for t in range(T - 2, -1, -1):
            for i in range(N):
                terms = log_trans[i, :] + log_emis[t + 1] + beta[t + 1]
                beta[t, i] = np.logaddexp.reduce(terms)
        return beta
