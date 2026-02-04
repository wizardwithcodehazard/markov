import numpy as np
from markovgpu import MarkovEngine


def test_import():
    """Verifies the library can be imported and initialized."""
    engine = MarkovEngine()
    assert engine is not None
    # Check if fallback to CPU is correctly set (Github Actions usually has no GPU)
    # This assertion is soft; it just prints status, but we expect it to handle either case.
    print(f"Engine initialized. GPU Enabled: {engine.use_gpu}")


def test_simple_step():
    """Verifies the basic Markov step logic (CPU/GPU transparently)."""
    engine = MarkovEngine()

    # 1. Identity Matrix Test
    P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    v = np.array([0.5, 0.5], dtype=np.float32)
    res = engine.step(P, v)

    np.testing.assert_allclose(res, v, atol=1e-6)

    # 2. State Flip Test
    P_flip = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    v_start = np.array([1.0, 0.0], dtype=np.float32)
    res_flip = engine.step(P_flip, v_start)

    expected = np.array([0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(res_flip, expected, atol=1e-6)


def test_viterbi_decoding():
    """Verifies Viterbi algorithm (Finding most likely state sequence)."""
    engine = MarkovEngine()

    # Setup: 2 States (0=Sunny, 1=Rainy)
    # Transition: Sunny->Sunny (0.8), Rainy->Rainy (0.6)
    trans_mat = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=np.float32)

    # Observations (T=3):
    # Day 0: Strong signal for Sunny (0)
    # Day 1: Strong signal for Sunny (0)
    # Day 2: Strong signal for Rainy (1)
    # Format: P(Obs | State) -> Shape (T, N)
    obs_probs = np.array(
        [
            [0.9, 0.1],  # Day 0
            [0.8, 0.2],  # Day 1
            [0.1, 0.9],  # Day 2
        ],
        dtype=np.float32,
    )

    # Expected Path: Sunny(0) -> Sunny(0) -> Rainy(1)
    expected_path = np.array([0, 0, 1], dtype=np.int32)

    path = engine.decode_regime(trans_mat, obs_probs)

    np.testing.assert_array_equal(path, expected_path)


def test_training_fit():
    """Verifies Baum-Welch training runs and returns valid probabilities."""
    engine = MarkovEngine()

    # Synthetic Data: 3 States, 100 Time steps
    N = 3
    T = 100

    # Create fake likelihoods (random but valid)
    np.random.seed(42)
    obs_probs = np.random.rand(T, N).astype(np.float32)
    # Normalize rows to represent probabilities (roughly)
    obs_probs /= obs_probs.sum(axis=1, keepdims=True)

    # Run Training
    # We use a small iteration count for testing speed
    learned_trans = engine.fit(obs_probs, n_states=N, n_iters=5)

    # Checks:
    # 1. Shape must be (N, N)
    assert learned_trans.shape == (N, N)

    # 2. Rows must sum to approx 1.0 (Stochastic matrix property)
    row_sums = learned_trans.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-3)

    # 3. Values must be probabilities (0 <= p <= 1)
    assert (learned_trans >= 0).all()
    assert (learned_trans <= 1.0 + 1e-5).all()
