import numpy as np
from markovgpu import MarkovEngine


def test_import():
    # Verifies the library can at least be imported
    engine = MarkovEngine()
    assert engine is not None


def test_cpu_fallback():
    # Verifies CPU logic works (since GitHub Actions has no GPU)
    engine = MarkovEngine()
    P = np.array([[1.0]], dtype=np.float32)
    v = np.array([1.0], dtype=np.float32)
    res = engine.step(P, v)
    assert res[0] == 1.0
