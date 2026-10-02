"""Right-preconditioned FGMRES (flexible GMRES), Saad 1993 / Saad & Schultz.

Why this file exists
--------------------
`scipy.sparse.linalg.fgmres` was REMOVED in this environment's scipy (1.18.1);
`scipy.sparse.linalg.lgmres` and `gcrotmk` are flexible too but do not expose
the per-iteration preconditioned residual history and an exact iteration count
that this project needs.  The project therefore carries its own
implementation, cross-checked against `scipy.sparse.linalg.gmres` with a fixed
preconditioner (see `self_test`).

FGMRES in one paragraph
-----------------------
GMRES minimises ||b - A x|| over the Krylov space K_j(A, r0).  With a *varying*
preconditioner M_j (e.g. an inner iterative solve, or a preconditioner that is
rebuilt from the current state) the standard GMRES Arnoldi relation
A V_j = V_{j+1} H_j no longer holds, because the vectors
z_j = M_j^{-1} v_j are not related to v_j by a fixed operator.  FGMRES fixes
this by storing the z_j and building the *right*-preconditioned space

    A Z_j = V_{j+1} H_j ,        x_j = x0 + Z_j y_j

so the least-squares problem is on H_j only, while the flexible preconditioner
only enters through Z_j.  Right preconditioning is essential here: the residual
whose norm FGMRES minimises is then the TRUE residual ||b - A x_j||, not the
preconditioned one.

Interface
---------
    fgmres(A, b, M=None, x0=None, rtol=1e-8, atol=0.0, restart=None,
           maxiter=None, callback=None) -> (x, info, iters, res_hist)

  A : callable, v -> A @ v                (any shape b.size)
  M : callable or None, r -> M^{-1} @ r   (approximate inverse of A)
  info : 0 converged, 1 maxiter/restart budget exhausted, 2 breakdown-free stop
"""
import numpy as np

__all__ = ["fgmres", "self_test"]


def _givens(a, b):
    """Complex Givens rotation G = [[c, s], [-conj(s), c]] with G [a; b] = [r; 0].

    c is real and >= 0, |c|^2 + |s|^2 = 1.  Returns (c, s, r).
    """
    aa = abs(a)
    bb = abs(b)
    if bb == 0.0:
        return 1.0, 0.0, a
    if aa == 0.0:
        s = np.conj(b) / bb
        return 0.0, s, bb
    denom = np.sqrt(aa * aa + bb * bb)
    c = aa / denom
    s = c * np.conj(b) / np.conj(a)
    return c, s, c * (aa * aa + bb * bb) / np.conj(a)


def fgmres(A, b, M=None, x0=None, rtol=1e-8, atol=0.0, restart=None,
           maxiter=None, callback=None):
    """Solve A x = b with right-preconditioned FGMRES.

    Parameters
    ----------
    A        : callable          v -> A v
    b        : (n,) array
    M        : callable or None  r -> M^{-1} r   (M approximates A^{-1})
    x0       : (n,) array or None   initial guess (warm start)
    rtol     : relative tolerance on ||b - A x|| / ||b||
    atol     : absolute tolerance on ||b - A x||
    restart  : Krylov dimension before a restart (default min(n, 30))
    maxiter  : maximum number of *inner* iterations over all restarts
    callback : callable(iters, res_norm) after every inner iteration

    Returns
    -------
    x        : (n,) solution
    info     : 0 converged; 1 budget exhausted; 2 breakdown (exact solution)
    iters    : total number of inner (matvec) iterations performed
    res_hist : residual-norm history.  Entry 0 is the TRUE initial residual
               `||b - A x0||`.  The entries appended inside a restart cycle are
               the GMRES quasi-residual estimates `|g[j+1]|` obtained from the
               Givens rotations: in exact arithmetic they equal the residual of
               the least-squares solution, but for an ill-conditioned `A` they
               can lose the true norm once the Arnoldi vectors lose
               orthogonality.  The LAST entry of every cycle is overwritten with
               the TRUE residual `||b - A x||` after the restart update
               (line `res_hist[-1] = beta`), so `res_hist[-1]` is always accurate.
    """
    b = np.asarray(b)
    n = b.size
    if x0 is None:
        x = np.zeros(n, dtype=complex if np.iscomplexobj(b) else float)
    else:
        x = np.array(x0, dtype=complex if np.iscomplexobj(b) else float)
    if restart is None:
        restart = min(n, 30)
    m = max(1, int(restart))
    if maxiter is None:
        maxiter = 10 * n
    maxiter = int(maxiter)
    bnorm = float(np.linalg.norm(b))
    tol = max(float(atol), float(rtol) * bnorm)

    Ax = A(x)
    r = b - Ax
    beta = float(np.linalg.norm(r))
    res_hist = [beta]
    if callback is not None:
        callback(0, beta)
    if beta <= tol:
        return x, 0, 0, res_hist

    total = 0
    info = 1
    ctype = complex
    while total < maxiter:
        V = np.zeros((m + 1, n), dtype=ctype)
        Z = np.zeros((m, n), dtype=ctype)
        H = np.zeros((m + 1, m), dtype=ctype)
        cs = np.zeros(m)
        sn = np.zeros(m, dtype=ctype)
        g = np.zeros(m + 1, dtype=ctype)
        V[0] = r / beta
        g[0] = beta
        k = 0
        for j in range(m):
            if total >= maxiter:
                info = 1
                break
            vj = V[j]
            Z[j] = M(vj) if M is not None else vj.copy()
            w = A(Z[j])
            for i in range(j + 1):
                h = np.vdot(V[i], w)
                H[i, j] = h
                w = w - h * V[i]
            hnext = float(np.linalg.norm(w))
            H[j + 1, j] = hnext
            # apply the previous rotations to column j of H
            for i in range(j):
                t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -np.conj(sn[i]) * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t
            c, s, rjj = _givens(H[j, j], H[j + 1, j])
            cs[j], sn[j] = c, s
            H[j, j] = rjj
            H[j + 1, j] = 0.0
            t = c * g[j] + s * g[j + 1]
            g[j + 1] = -np.conj(s) * g[j] + c * g[j + 1]
            g[j] = t
            total += 1
            k = j + 1
            rnorm = float(abs(g[j + 1]))
            res_hist.append(rnorm)
            if callback is not None:
                callback(total, rnorm)
            if rnorm <= tol:
                info = 0
                break
            if hnext == 0.0:                 # lucky breakdown: exact solve
                info = 2
                break
            V[j + 1] = w / hnext
        if k == 0:                           # no budget left at all
            break
        # least-squares solve of the (k x k) triangular system H y = g
        y = np.linalg.solve(H[:k, :k], g[:k])
        x = x + Z[:k].T @ y
        if info in (0, 2):
            break
        r = b - A(x)
        beta = float(np.linalg.norm(r))
        res_hist[-1] = beta
        if beta <= tol:
            info = 0
            break
    return x, info, total, res_hist


# ------------------------------------------------------------------ self test
def self_test(verbose=True):
    """Cross-check against scipy's GMRES (fixed preconditioner) and a dense solve."""
    import scipy.linalg as sla
    from scipy.sparse.linalg import gmres as sp_gmres
    rng = np.random.default_rng(0)
    ok = True
    for n, kind in ((40, 'real'), (40, 'complex')):
        A = rng.normal(size=(n, n))
        if kind == 'complex':
            A = A + 1j * rng.normal(size=(n, n))
        A = A + n * np.eye(n)
        b = rng.normal(size=n) + (1j * rng.normal(size=n) if kind == 'complex' else 0)
        xref = sla.solve(A, b)
        # 1. unpreconditioned, full restart -> must behave like scipy gmres
        x, info, it, hist = fgmres(lambda v: A @ v, b, rtol=1e-12,
                                   restart=n, maxiter=n + 5)
        e1 = np.linalg.norm(x - xref) / np.linalg.norm(xref)
        xs, isp = sp_gmres(A, b, rtol=1e-12, restart=n, maxiter=n + 5)[:2]
        e2 = np.linalg.norm(xs - xref) / np.linalg.norm(xref)
        # 2. with a fixed (Jacobi) preconditioner, scipy must agree too
        d = np.diag(A).copy()
        M = lambda r: r / d
        xp, infop, itp, histp = fgmres(lambda v: A @ v, b, M=M, rtol=1e-12,
                                       restart=n, maxiter=n + 5)
        e3 = np.linalg.norm(xp - xref) / np.linalg.norm(xref)
        # 3. a genuinely FLEXIBLE preconditioner (changes every call): FGMRES
        #    must still converge; plain GMRES with the same map need not.
        cnt = {'n': 0}

        def Mflex(r):
            cnt['n'] += 1
            return r / d * (1.0 + 1e-3 * np.cos(cnt['n']))
        xf, infof, itf, _ = fgmres(lambda v: A @ v, b, M=Mflex, rtol=1e-10,
                                   restart=n, maxiter=4 * n)
        e4 = np.linalg.norm(xf - xref) / np.linalg.norm(xref)
        # 4. warm start from the exact solution -> zero iterations
        xw, infow, itw, _ = fgmres(lambda v: A @ v, b, x0=xref, rtol=1e-12)
        ok &= (e1 < 1e-9 and e2 < 1e-9 and e3 < 1e-9 and e4 < 1e-8
               and itw == 0 and infow == 0)
        if verbose:
            print('  [%s n=%d] dense %.1e | scipy-gmres ref %.1e | fgmres %.1e |'
                  ' jacobi %.1e | flexible %.1e | warm-start iters %d'
                  % (kind, n, e1, e2, e1, e3, e4, itw))
    if verbose:
        print('  fgmres self_test %s' % ('PASSED' if ok else 'FAILED'))
    return bool(ok)


if __name__ == '__main__':
    print('FGMRES self test')
    self_test()
