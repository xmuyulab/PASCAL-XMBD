import logging
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LabelSpreading:
    """
    Label Spreading with an SNN (Shared Nearest Neighbor) graph.

    Parameters
    ----------
    k : int
        Number of neighbors for kNN.
    max_iter : int
        Maximum number of propagation iterations.
    alpha : float
        Clamping factor in (0, 1). Higher alpha -> more propagation, less clamping to initial labels.
    tol : float
        Convergence tolerance based on infinity norm of label distribution change.
    min_shared : int
        Minimum number of shared neighbors to keep an edge in SNN.
    metric : str
        Distance metric for kNN ('euclidean', 'cosine').
    """
    def __init__(self, k=10, max_iter=100, alpha=0.8, tol=1e-6, min_shared=1, metric='euclidean'):
        self.k = k
        self.max_iter = max_iter
        self.alpha = alpha
        self.tol = tol
        self.min_shared = min_shared
        self.metric = metric


    def _build_snn_graph(self):
        """
        Build an SNN (Shared Nearest Neighbor) graph.
        """
        X = self.X_
        n_samples = X.shape[0]
            
        nbrs = NearestNeighbors(n_neighbors=self.k+1, metric=self.metric).fit(X)
        distances, indices = nbrs.kneighbors(X)  # [n_samples, k+1]

        # Remove self-loop neighbor
        indices = indices[:, 1:]
        
        # Build a binary kNN adjacency matrix (sparse)
        adjacency_binary = csr_matrix((np.ones(n_samples * self.k), 
                                    (np.repeat(np.arange(n_samples), self.k), 
                                    indices.flatten())), 
                                    shape=(n_samples, n_samples))
        
        # Compute shared-neighbor counts
        intersection = adjacency_binary @ adjacency_binary.T
        intersection.setdiag(0) # remove self-loops
        intersection.eliminate_zeros()
        intersection = intersection.tocoo()
        
        mask = intersection.data >= self.min_shared
        row = intersection.row[mask]
        col = intersection.col[mask]
        data = intersection.data[mask]

        W_snn = csr_matrix((data, (row, col)), shape=(n_samples, n_samples))
        W_snn = 0.5 * (W_snn + W_snn.T)
        degrees = np.array(W_snn.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1e-8
        D_inv_sqrt = 1.0 / np.sqrt(degrees)
        # D_inv_sqrt_mat = csr_matrix(np.diag(D_inv_sqrt))
        D_inv_sqrt_mat = sp.diags(D_inv_sqrt, format='csr')

        S = D_inv_sqrt_mat @ W_snn @ D_inv_sqrt_mat

        return S
    
    

    def fit(self, X, y):
        """
        Fit the model.
        """
        self.X_ = np.array(X)

        print(f"[INFO] Original data dimension: {self.X_.shape[1]}")

        # --- Dimensionality Reduction with LDA ---
        labeled_mask = y != -1
        if np.sum(labeled_mask) > 0:
            # n_components cannot be larger than n_classes - 1
            n_classes = len(np.unique(y[labeled_mask]))
            n_comp = n_classes - 1
            if n_comp < 1:
                print("[WARNING] Not enough classes for LDA, skipping dimensionality reduction.")
            else:
                lda = LinearDiscriminantAnalysis(n_components=n_comp)
                lda.fit(self.X_[labeled_mask], y[labeled_mask])
                self.X_ = lda.transform(self.X_)
                
                print(f"[INFO] Data dimension after reduction: {self.X_.shape[1]}")
        else:
            print("[WARNING] No labeled data available for LDA, skipping dimensionality reduction.")

        # Build SNN graph
        self.graph_matrix = self._build_snn_graph()
        
        classes = np.unique(y)
        self.classes_ = (classes[classes != -1])

        n_samples, n_classes = len(y), len(self.classes_)

        alpha = self.alpha
        if alpha is None or alpha <= 0.0 or alpha >= 1.0:
            raise ValueError("alpha must be in (0, 1)")
        
        self.label_distributions_ = np.zeros((n_samples, n_classes))
        for label in self.classes_:
            self.label_distributions_[y == label, self.classes_ == label] = 1
        
        y_static = np.copy(self.label_distributions_)
        y_static *= 1 - alpha
        l_previous = np.zeros((self.X_.shape[0], n_classes))

        for self.n_iter_ in range(self.max_iter):
            if np.linalg.norm(self.label_distributions_ - l_previous, ord=np.inf) < self.tol:
                print(self.n_iter_)
                break
            l_previous = self.label_distributions_.copy()
            F_new = self.graph_matrix @ self.label_distributions_
            self.label_distributions_ = self.alpha * F_new + y_static
        else:
            logging.warning(
                'max_iter=%d was reached without convergence.' % self.max_iter)
            self.n_iter_ += 1

        # Save the unnormalized scores before row-wise normalization
        self.unnormalized_label_distributions_ = self.label_distributions_.copy()
        # normalized scores
        normalizer = np.sum(self.label_distributions_, axis=1, keepdims=True)
        normalizer[normalizer == 0] = 1
        self.label_distributions_ /= normalizer

        return self
