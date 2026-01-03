# backend/reconstruction.py
import numpy as np

def tikhonov_reconstruction(A, b, lam=1e-2):
    """
    Perform Tikhonov regularized reconstruction: (A^T A + λI)x = A^T b
    """
    AtA = A.T @ A
    Atb = A.T @ b
    M = AtA + lam * np.eye(A.shape[1])
    x = np.linalg.solve(M, Atb)
    return x

def least_squares(A, b):
    """
    Perform simple least-squares reconstruction.
    """
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x

def gradient_descent(A, b, lam=1e-2, steps=500, alpha=None):
    """
    Gradient Descent solver for (A^T A + λI)x = A^T b
    """
    n = A.shape[1]
    x = np.zeros(n)
    if alpha is None:
        # heuristic step size
        alpha = 1 / (np.linalg.norm(A, 2)**2 + lam)
    for _ in range(steps):
        grad = A.T @ (A @ x - b) + lam * x
        x -= alpha * grad
    return x

def conjugate_gradient(A, b, lam=1e-2, tol=1e-6, maxiter=1000):
    """
    Conjugate Gradient solver for (A^T A + λI)x = A^T b
    """
    AtA = A.T @ A
    Atb = A.T @ b
    n = A.shape[1]
    def matvec(v): return AtA @ v + lam * v

    x = np.zeros(n)
    r = Atb - matvec(x)
    p = r.copy()
    rsold = np.dot(r, r)

    for _ in range(maxiter):
        Ap = matvec(p)
        alpha = rsold / (np.dot(p, Ap) + 1e-12)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / (rsold + 1e-12)) * p
        rsold = rsnew

    return x
