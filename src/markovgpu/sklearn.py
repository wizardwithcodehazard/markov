import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.stats import norm
from .backend import MarkovEngine

class GpuHMM(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible Wrapper for MarkovGPU.
    Allows use in Pipelines, GridSearchCV, and Cross-Validation.
    """
    def __init__(self, n_states=2, n_iter=100, tolerance=1e-4, verbose=False):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.verbose = verbose
        self.engine = MarkovEngine()
        
        # Learned Parameters
        self.trans_mat_ = None
        self.start_prob_ = None
    
    def fit(self, X, y=None):
        """
        Trains the HMM on the GPU.
        X: array-like of shape (n_samples, n_features) OR (n_samples,)
           For now, we assume X represents 'Observation Probabilities' 
           OR raw data we can model as Gaussian emissions.
        """
        # 1. Input Validation
        X = check_array(X, ensure_2d=False)
        
        # 2. Heuristic: If X is 1D (Raw Data), we convert to Emission Probs
        # using a simple Gaussian mixture assumption for convenience.
        if X.ndim == 1 or X.shape[1] == 1:
            if self.verbose:
                print(f"ℹ️ Auto-converting raw data to {self.n_states} Gaussian states.")
            X_flat = X.ravel()
            obs_probs = self._auto_gaussian_emissions(X_flat)
        else:
            # Assume X is already [Probability of State 0, Prob of State 1, ...]
            if X.shape[1] != self.n_states:
                raise ValueError(f"Input has {X.shape[1]} columns, but n_states={self.n_states}. "
                                 "If passing raw probabilities, cols must match n_states.")
            obs_probs = X

        # 3. Train on GPU
        if self.verbose:
            print(f"🚀 Offloading to GPU: {X.shape[0]} samples, {self.n_states} states")
            
        self.trans_mat_ = self.engine.fit(
            obs_probs, 
            n_states=self.n_states, 
            n_iters=self.n_iter, 
            tolerance=self.tolerance
        )
        
        # Set is_fitted flag
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Returns the most likely hidden state path (Viterbi).
        """
        check_is_fitted(self, ['trans_mat_'])
        X = check_array(X, ensure_2d=False)
        
        if X.ndim == 1 or X.shape[1] == 1:
            obs_probs = self._auto_gaussian_emissions(X.ravel())
        else:
            obs_probs = X
            
        return self.engine.decode_regime(self.trans_mat_, obs_probs)

    def _auto_gaussian_emissions(self, data):
        """
        Helper: Splits data into N quantiles and assumes Gaussian emissions.
        This makes the class 'Just Work' for simple 1D data.
        """
        T = len(data)
        N = self.n_states
        
        # Smart Init: Sort data and split into N chunks to guess means
        sorted_data = np.sort(data)
        chunk_size = T // N
        means = [np.mean(sorted_data[i*chunk_size : (i+1)*chunk_size]) for i in range(N)]
        std = np.std(data) * 0.5 # Heuristic width
        
        probs = np.zeros((T, N), dtype=np.float32)
        for k in range(N):
            probs[:, k] = norm.pdf(data, loc=means[k], scale=std)
            
        return probs