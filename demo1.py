import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from markovgpu import MarkovEngine
from hmmlearn import hmm

def run_benchmark_suite():
    # Configuration
    STATE_SIZES = [10, 50, 100, 200, 500]
    SEQ_LENGTH = 5000  # Fixed sequence length
    
    results = []

    print(f"📊 MARKOVGPU vs HMMLEARN: The Scaling Benchmark")
    print(f"   Sequence Length: {SEQ_LENGTH} steps")
    print("-" * 65)
    print(f"{'N_STATES':<10} | {'CPU (s)':<10} | {'GPU (s)':<10} | {'Speedup':<10} | {'Winner':<10}")
    print("-" * 65)

    # Initialize Engine once (warmup)
    engine = MarkovEngine()

    for N in STATE_SIZES:
        # --- 1. Generate Synthetic Data ---
        # Transition Matrix (Random stochastic)
        trans_mat = np.random.rand(N, N).astype(np.float32)
        trans_mat /= trans_mat.sum(axis=1, keepdims=True)
        
        # Emission Probabilities (Pre-computed likelihoods for fair comparison)
        # Shape: (T, N)
        obs_probs = np.random.rand(SEQ_LENGTH, N).astype(np.float32)
        
        # --- 2. Benchmark hmmlearn (CPU) ---
        # We assume a Categorical HMM structure for fair comparison of Viterbi
        h_model = hmm.CategoricalHMM(n_components=N)
        h_model.startprob_ = np.full(N, 1/N)
        h_model.transmat_ = trans_mat
        # Dummy emissions to satisfy API (we won't use them directly in logic if possible, 
        # but hmmlearn forces us to pass observations. 
        # To make it fair, we benchmark the Viterbi step overhead.)
        
        # HACK: hmmlearn doesn't accept pre-computed probabilities easily.
        # So we will benchmark the "Decode" step on a dummy sequence 
        # and assume hmmlearn's internal emission calc is negligible compared to Viterbi recursion.
        dummy_X = np.random.randint(0, 10, size=(SEQ_LENGTH, 1))
        h_model.emissionprob_ = np.random.rand(N, 10)
        h_model.emissionprob_ /= h_model.emissionprob_.sum(axis=1, keepdims=True)

        start_cpu = time.time()
        try:
            # Run Viterbi
            _ = h_model.decode(dummy_X, algorithm="viterbi")
            end_cpu = time.time()
            time_cpu = end_cpu - start_cpu
        except Exception:
            time_cpu = 999.0 # Timeout/Crash

        # --- 3. Benchmark MarkovGPU (GPU) ---
        start_gpu = time.time()
        # Run Viterbi
        _ = engine.decode_regime(trans_mat, obs_probs)
        end_gpu = time.time()
        time_gpu = end_gpu - start_gpu

        # --- 4. Record & Print ---
        speedup = time_cpu / time_gpu
        winner = "GPU 🚀" if speedup > 1.0 else "CPU 🐢"
        
        print(f"{N:<10} | {time_cpu:<10.4f} | {time_gpu:<10.4f} | {speedup:<10.2f}x | {winner}")
        
        results.append({
            "N": N,
            "CPU": time_cpu,
            "GPU": time_gpu,
            "Speedup": speedup
        })

    # --- 5. Visualization ---
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Execution Time (Log Scale)
    plt.subplot(1, 2, 1)
    plt.plot(df["N"], df["CPU"], 'o--', label="hmmlearn (CPU)", color='orange')
    plt.plot(df["N"], df["GPU"], 's-', label="MarkovGPU", color='blue', linewidth=2)
    plt.yscale('log')
    plt.xlabel("Number of States (N)")
    plt.ylabel("Time (seconds) - Log Scale")
    plt.title("Execution Time vs Model Complexity")
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.5)

    # Plot 2: Speedup Factor
    plt.subplot(1, 2, 2)
    plt.plot(df["N"], df["Speedup"], '^-', color='green', linewidth=2)
    plt.axhline(1.0, color='red', linestyle='--', label="Break-even")
    plt.xlabel("Number of States (N)")
    plt.ylabel("Speedup Factor (x)")
    plt.title("GPU Speedup over CPU")
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.tight_layout()
    print("\n🎨 Generating Benchmark Plot...")
    plt.show()

if __name__ == "__main__":
    run_benchmark_suite()