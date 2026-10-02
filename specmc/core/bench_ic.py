"""Blankenbach / ASPECT case-2a INITIAL CONDITIONS.

Two initial states are provided, both returning Chebyshev/Fourier COEFFICIENTS
of the temperature deviation `theta = T - (1 - z)` on the model grid.

`aspect_theta(model, path=None)`
    The *real* ASPECT benchmark initial condition: the ascii-data field
    `benchmarks/blankenbach/initial_temperature_case2a.txt`, which is what
    `base_case2a.prm` uses ("List of model names = ascii data").
    The file is a 16 x 16 structured field on cell centres
    x, y in {(j+1/2)/16}.  It is NOT the analytic boundary-layer profile that
    the (dead) `Function` subsection of base_case2a.prm documents.

`analytic_bl_theta(model, ...)`
    The analytic profile from the *inactive* `Function` block of
    base_case2a.prm:

        BL(z) = 0.5 + 0.5 (z1 - z)/z1          z < z1
              = 0.5                            z1 <= z <= z2
              = 0.5 (1 - z)/(1 - z2)           z > z2
        T     = BL(z) + 0.1 cos(pi x) sin(pi z)
        z1 = 0.102367, z2 = 0.897633

Both are mapped onto the model's periodic Lx = 2 grid with the mirror
continuation x -> 2 - x, which is exactly the free-slip / insulating 1x1 box.
"""
import os
import numpy as np

__all__ = ["ASPECT_IC_NAME", "find_aspect_ic", "aspect_theta",
           "analytic_bl_theta", "conductive_theta", "ic_theta", "IC_KINDS",
           "aspect_theta_interp", "aspect_field"]

ASPECT_IC_NAME = 'initial_temperature_case2a.txt'

#: search order for the ascii data file
_CANDIDATES = (
    os.environ.get('SPECMC_ASPECT_IC'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ASPECT_IC_NAME),
    # phase-1: the ascii data file still lives at the run-tree root, one level
    # above this package  (specmc/core/bench_ic.py -> specmc -> root)
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), ASPECT_IC_NAME),
)
IC_KINDS = ('aspect_file', 'analytic_bl', 'conductive')


def find_aspect_ic(path=None):
    """Return the first existing path of the ASPECT ascii initial field."""
    cands = ([path] if path else []) + [c for c in _CANDIDATES if c]
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError('could not find %s (tried %s)'
                            % (ASPECT_IC_NAME, cands))


def read_aspect_ic(path=None):
    """-> (x, y, T) 16x16 arrays from the ASPECT ascii data file."""
    path = find_aspect_ic(path)
    raw = open(path).read().splitlines()
    body = np.array([[float(v) for v in l.split()] for l in raw
                     if l.strip() and not l.startswith('#')])
    nx, ny = [int(v) for v in
              [l for l in raw if l.startswith('# POINTS')][0]
              .split(':')[1].split()]
    x = body[:, 0].reshape(ny, nx)
    y = body[:, 1].reshape(ny, nx)
    T = body[:, 2].reshape(ny, nx)
    if x.shape != (ny, nx) or y.shape != (ny, nx):
        raise ValueError('unexpected ascii layout %s' % (body.shape,))
    if not np.allclose(x, x[0]):
        raise ValueError('x is not the fastest index in %s' % path)
    if not np.allclose(y, y[:, :1]):
        raise ValueError('y is not constant along a column in %s' % path)
    return path, x[0], y[:, 0], T


def _cosine_coeffs(F, n):
    """Exact cosine-series coefficients of samples at x_j=(j+1/2)/n.

    Solves  sum_{m<n} a_m cos(m pi x_j) = F[j]  for a_m (the 16x16 system is
    invertible), so the returned series is the unique mirror-symmetric
    (cosine) interpolant with m <= n-1.  F has the n samples along axis 0 and
    may carry trailing dimensions.
    """
    j = np.arange(n)
    xj = (j + 0.5) / n
    M = np.cos(np.pi * np.outer(xj, np.arange(n)))       # (j, m)
    return np.linalg.solve(M, np.asarray(F, float))


def aspect_field(model, path=None, zinterp='pchip'):
    """ASPECT ascii field evaluated on the model grid -> theta(x,z) grid array.

    * x: exact cosine-series interpolation of the 16 samples (the field is
      mirror symmetric, so it is a pure cosine series); the series is then
      evaluated on the periodic Lx=2 grid, which supplies the mirror copy.
    * z: interpolation of the 16 uniform samples onto the model's CGL nodes.
      The isothermal walls (theta = 0 at z = 0, 1) are added as EXPLICIT
      knots so the interpolant is never evaluated outside its data range.
      This matters: with the 16 interior nodes alone a natural cubic spline
      must extra-polate over [0, 1/32] and [31/32, 1], and the CGL nodes sit
      as close as 1e-3 to the wall, the extrapolated cubic then yields a
      wall slope of O(1e2) instead of O(1) (measured: Nu_top = 115 at 32x48
      and 453 at 64x96, pure interpolation garbage).
      zinterp: 'pchip' (default, shape preserving), 'spline' (not-a-knot
      cubic through the augmented knots) or 'poly' (degree-17 fit).
    """
    path, ax, ay, T = read_aspect_ic(path)
    th = T - (1.0 - ay)[:, None]                  # deviation from conductive
    n = len(ax)
    A = _cosine_coeffs(th.T, n)                   # (m, ny)  coeffs per y node

    # --- z: knots = both walls + the 16 data nodes, value 0 at both walls
    zk = np.concatenate(([0.0], ay, [1.0]))
    Ak = np.concatenate((np.zeros((A.shape[0], 1)), A,
                         np.zeros((A.shape[0], 1))), axis=1)     # (m, 18)
    z = np.asarray(model.z, float)
    if zinterp == 'poly':
        C = np.polynomial.polynomial.polyfit(zk, Ak.T, len(zk) - 1)  # (18,m)
        vals = np.polynomial.polynomial.polyval(z, C.T).T            # (m, Nz)
    elif zinterp == 'spline':
        from scipy.interpolate import CubicSpline
        vals = CubicSpline(zk, Ak, axis=1, bc_type='not-a-knot')(z)
    else:
        from scipy.interpolate import PchipInterpolator
        vals = PchipInterpolator(zk, Ak, axis=1)(z)
    vals = np.asarray(vals, float)

    # --- x: evaluate the cosine series at the periodic grid points x_i
    xi = np.asarray(model.x, float)
    Cx = np.cos(np.pi * np.outer(xi, np.arange(n)))       # (Nx, m)
    g = Cx @ vals                                         # (Nx, Nz)
    g[:, 0] = 0.0                                         # isothermal walls
    g[:, -1] = 0.0
    return g


def aspect_theta_interp(model, path=None, zinterp='pchip'):
    """theta COEFFICIENTS of the ASPECT ascii initial field."""
    return model.coeffs(aspect_field(model, path, zinterp))


def aspect_theta(model, path=None, zinterp='pchip'):
    return aspect_theta_interp(model, path, zinterp)


def analytic_bl_field(model, z1=0.102367, z2=0.897633, amp=0.1, m=1, n=1):
    """Analytic boundary-layer + perturbation (base_case2a.prm `Function`)."""
    X, Z = np.meshgrid(np.asarray(model.x, float), np.asarray(model.z, float),
                       indexing='ij')
    T = np.where(Z < z1, 0.5 + 0.5 * (z1 - Z) / z1,
                 np.where(Z > z2, 0.5 * (1.0 - Z) / (1.0 - z2), 0.5))
    T = T + amp * np.cos(m * np.pi * X) * np.sin(n * np.pi * Z)
    g = T - (1.0 - Z)
    g[:, 0] = 0.0
    g[:, -1] = 0.0
    return g


def analytic_bl_theta(model, **kw):
    return model.coeffs(analytic_bl_field(model, **kw))


def conductive_theta(model, amp=1e-3, m=1, n=1):
    """The IC used by every previous run: pure conductive profile + tiny seed."""
    return model.seed(amp=amp, m=m, n=n)


def ic_theta(model, kind, path=None, amp=1e-3, zinterp='pchip'):
    kind = str(kind).lower()
    if kind in ('aspect_file', 'aspect', 'file'):
        return aspect_theta(model, path=path, zinterp=zinterp)
    if kind in ('analytic_bl', 'analytic', 'bl'):
        return analytic_bl_theta(model)
    if kind in ('conductive', 'cond', 'old'):
        return conductive_theta(model, amp=amp)
    raise ValueError('unknown initial condition %r (use one of %s)'
                     % (kind, IC_KINDS))


# ------------------------------------------------------------------ self test
if __name__ == '__main__':
    from ..models.coupled_model import RBCCoupled

    path, ax, ay, T = read_aspect_ic()
    print('ASPECT ascii initial condition: %s' % path)
    print('  grid %dx%d  x in [%.5f, %.5f]  y in [%.5f, %.5f]'
          % (T.shape[0], T.shape[1], ax[0], ax[-1], ay[0], ay[-1]))
    print('  T in [%.5f, %.5f]   node mean T = %.5f'
          % (T.min(), T.max(), T.mean()))

    for Nx, Nz in ((32, 48), (64, 96)):
        m = RBCCoupled(Ra=1e4, Nx=Nx, Nz=Nz, deta_T=1000.0, symmetrize=True,
                       zform='reduced', theta_scheme='implicit_bdf2',
                       theta_extrapolate=True, implicit_buoy=False,
                       stokes_mode='defect', lag=16, K=3, stokes_rtol=1e-8,
                       defect_tol=1e-8, max_K=8, eta_drift_tol=0.1)
        print('')
        print('  --- %dx%d ---' % (Nx, Nz))
        for kind in IC_KINDS:
            that = ic_theta(m, kind, path=path)
            g = np.real(m.grid(that))
            d = m.diagnostics(that)
            # interior mean temperature z in [0.25, 0.75], volume weighted
            Tf = 1.0 - m._Z + g
            w = m.wq * ((m.z >= 0.25) & (m.z <= 0.75))
            w = w / w.sum()
            Ti = float(np.sum(w * Tf.mean(axis=0)))
            print('    %-12s amp=%8.4f  Nu_top=%8.4f  Nu_bot=%8.4f  '
                  'Nu_E=%8.4f  Vrms=%9.3f  umax=%9.1f  <T>=%.5f  Tbar=%.5f'
                  % (kind, d['amp'], d['Nu_top'], d['Nu_bot'], d['Nu_energy'],
                     d['urms'], d['umax'], m.vol_mean(Tf), Ti))
            # mirror error / wall values
            print('                 mirror_err=%.2e  theta(wall)=%.2e'
                  % (m.mirror_error(that), max(abs(g[:, 0]).max(),
                                               abs(g[:, -1]).max())))
