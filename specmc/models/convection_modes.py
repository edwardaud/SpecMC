"""Single entry point for the 2D infinite-Prandtl convection model, with the
multi-physics switch (`viscosity_law`, `deta_T`).

Design rules
----------------------------------------------
* ONE code base, ONE time-stepping / transform / diagnostic framework: the
  variable-viscosity model is a *subclass* of `rbc_galerkin.RBC` that overrides
  only `stokes()`.  Everything else, FFT/Chebyshev transforms, 3/2 dealiasing,
  Clenshaw-Curtis volume weights, the BDF1 implicit diffusion step, the CFL
  control, the Nusselt/velocity diagnostics, is inherited unchanged.
* DEFAULT IS THE FROZEN BASELINE.  `viscosity_law='isoviscous'` (the default)
  returns a plain `rbc_galerkin.RBC` instance, so the behaviour is not only
  numerically equal but the same code path as the validated
  solver (`Nu = 4.885140`, `Vrms = 42.872720` at Ra=1e4, 64x48, t=0.25).
* Every extension is behind a parameter; nothing else changes.
* the isoviscous baseline is re-checked after each new physics module is added.

Viscosity modes
    'isoviscous'    eta = 1                               (default; frozen baseline)
    'exp_T_depth'   eta(z) = exp(-ln(deta_T) * (1-z))     (depth only)

The depth-only mode keeps the Stokes operator time independent, so the
per-wavenumber LU cache of the isoviscous solver is preserved.  The full
eta = eta(T(x,z,t)) mode (coupled) breaks the wavenumber decoupling and needs a
different solver; it is NOT implemented here.

Production defaults for the variable-viscosity mode
    zform='reduced'   block-eliminated 2nd-order Stokes operator
                      (u = (D^2-k^2) psi = -omega; conditioning ~N^4 instead of
                      the 4th-order tau's ~N^8).  'tau4' selects the 4th-order
                      reference operator.
    symmetrize=True   exact box (mirror) projection each step.  At deta_T = 1
                      this is the identity to roundoff, so the frozen
                      isoviscous baseline is unaffected.
"""
import numpy as np

from ..core.rbc_galerkin import RBC, cgl, cheb_V
from ..stokes.vv_stokes import VVStokes, VVStokesReduced
from ..physics import heating as HEAT
from ..physics import eta_bounds as EB
from ..physics.viscosity import eta_from_law, BASELINE_REF, REFERENCE

__all__ = ["make_convection", "RBCVariableViscosity", "list_modes",
           "BASELINE_REF", "REFERENCE"]

ISOVISCOUS = ("isoviscous", "const", "constant")
VV_LAWS = ("exp_t_depth", "exp_T_depth", "depth_exp")
ZFORMS = ("reduced", "tau4")


def _cheb_coeffs_of_eta(model, eta_vals):
    """Grid values of eta(z) -> Chebyshev coefficients (reuses RBC.Vi)."""
    return model.Vi @ np.asarray(eta_vals, dtype=float)


class RBCVariableViscosity(RBC):
    """`rbc_galerkin.RBC` with a frozen eta(z) Stokes operator.

    Only `stokes()` differs from the isoviscous model, so every diagnostic and
    every time-stepping detail is bit-for-bit the validated machinery.
    """

    def __init__(self, Ra=1e4, Nx=64, Nz=48, Lx=2.0, zsolver='tau',
                 dealias=True, viscosity_law='exp_T_depth', deta_T=1000.0,
                 use_lu=True, dealias_z=False, symmetrize=True,
                 zform='reduced', internal_heating=0.0,
                 dissipation_number=0.0, compressibility='boussinesq',
                 eta_min=None, eta_max=None, eta_smooth=None):
        super().__init__(Ra=Ra, Nx=Nx, Nz=Nz, Lx=Lx, zsolver=zsolver,
                         dealias=dealias, internal_heating=internal_heating,
                         dissipation_number=dissipation_number,
                         compressibility=compressibility)
        law = str(viscosity_law).lower()
        if law not in VV_LAWS:
            raise ValueError('unknown viscosity law %r' % (viscosity_law,))
        self.viscosity_law = law
        self.deta_T = float(deta_T)
        # the explicit viscosity bounds of `specmc.physics.eta_bounds`
        # (`ConvectionConfig.eta_min` / `eta_max`).  They are applied ONCE here,
        # to the frozen eta(z) field, before its Chebyshev coefficients are
        # taken, so the depth-only operator sees exactly the bounded field.
        # `None` means the documented default window.  The naming: the
        # bounds are `eta_clip_lo/hi`; `eta_min/eta_max` below keep their reference
        # meaning (the realised min/max of the eta field).
        self.eta_clip_lo, self.eta_clip_hi = (
            EB.bounds_default() if eta_min is None or eta_max is None
            else EB.validate_bounds(eta_min, eta_max))
        # the smooth clip: the C^2 smooth transition (`None` = 0.0 = the
        # hard clip, bit for bit).
        self.eta_smooth = EB.validate_smooth(
            EB.SMOOTH_DEFAULT if eta_smooth is None else eta_smooth,
            self.eta_clip_lo, self.eta_clip_hi)
        self.n_clip_eta = 0
        self.n_smooth_eta = 0
        self.eta_grid = eta_from_law(law, self.deta_T, self.z)
        if self.eta_smooth > 0.0:
            # keep the hard-clip counter meaningful next to the smooth one
            self.n_clip_eta = EB.n_clipped(
                self.eta_grid, self.eta_clip_lo, self.eta_clip_hi)
            self.eta_grid, self.n_smooth_eta = EB.smooth_clip_eta(
                self.eta_grid, self.eta_clip_lo, self.eta_clip_hi,
                self.eta_smooth)
        else:
            self.eta_grid, self.n_clip_eta = EB.clip_eta(
                self.eta_grid, self.eta_clip_lo, self.eta_clip_hi)
        self.eta_c = _cheb_coeffs_of_eta(self, self.eta_grid)
        # report, not silently clamp: negative eta would be unphysical
        self.eta_min = float(self.eta_grid.min())
        self.eta_max = float(self.eta_grid.max())
        # 'reduced' = block-eliminated 2nd-order operator (conditioning ~N^4,
        #             1.4e6x better conditioned than the 4th-order tau at
        #             Nz=96 and identical to it on resolved fields);
        # 'tau4'    = the 4th-order tau operator, kept for reference.
        self.zform = str(zform).lower()
        if self.zform not in ZFORMS:
            raise ValueError('unknown zform %r (use one of %s)'
                             % (zform, ZFORMS))
        Solver = VVStokesReduced if self.zform == 'reduced' else VVStokes
        self.vv = Solver(self.Z, self.kx, self.Ra, self.eta_c, use_lu=use_lu)
        self.n_stokes = 0
        self.dealias_z = bool(dealias_z)
        # DEFAULT ON for the variable-viscosity mode: the
        # doubled periodic grid also carries the box-forbidden antisymmetric
        # modes, which become unstable at high deta_T.  At deta_T = 1 the
        # projection is the identity to roundoff (verified as an internal
        # cross-check), so the isoviscous baseline is untouched.
        self.symmetrize = bool(symmetrize)
        self.sym_removed = 0.0
        if self.dealias_z:
            self._init_zpad()

    # ------------------------------------------- box (mirror) symmetry control
    def project_mirror(self, that):
        """Remove the x-antisymmetric part of theta.

        The 1x1 free-slip / insulating box is carried on a periodic grid of
        length Lx = 2 (see rbc_galerkin.py).  That doubled domain also admits
        the x-antisymmetric Fourier modes, which do NOT satisfy the box side
        conditions (they have theta_x != 0 at x = 0, 1 and psi != 0 at the
        side walls).  The mirror-symmetric subspace is invariant under the
        continuous equations, so it is exact, but if a *spurious or physical*
        antisymmetric mode is unstable, the doubled-domain integration drifts
        out of the box.  Projecting back is therefore not a filter: it removes
        only modes that are not solutions of the intended problem.

        A field is mirror symmetric iff its rfft coefficients are real (the
        rfft coefficient of mode i is X_i = V @ that_i).
        """
        X = (self.V @ np.asarray(that).T).T
        anti = float(np.abs(X.imag).max() / max(np.abs(X).max(), 1e-300))
        self.sym_removed = anti
        return (self.Vi @ X.real.T).T

    # ------------------------------------------- optional z-direction dealias
    def _init_zpad(self, Npad=None):
        """Zero-padded Chebyshev grid for an aliasing-free z product.

        `rbc_galerkin` dealiases only the *x* direction (2/3 rule, Np = 2 Nx).
        In z the product is evaluated on the same Nz CGL nodes, so the
        quadratic nonlinearity aliases the top Nz-1 Chebyshev modes back onto
        the resolved ones.  With Npad >= 2 Nz - 1 nodes the product of two
        degree-(Nz-1) polynomials is represented exactly and only *truncated*,
        which is the Galerkin projection instead of an aliasing error.
        """
        N = self.Nz
        self.Npad = int(Npad) if Npad else (2 * N - 1)
        zn = cgl(self.Npad)
        self.Vp = cheb_V(zn, N)                                  # (Npad, N)
        self.Tp = np.linalg.inv(cheb_V(zn, self.Npad))[:N, :]    # (N, Npad)

    def _Gz(self, cf):
        """Chebyshev coefficients (Nk, Nz) -> values on the (Np x Npad) grid."""
        a = (self.Vp @ np.asarray(cf).T).T                       # (Nk, Npad)
        sp = np.zeros((self.Np // 2 + 1, a.shape[1]), dtype=complex)
        sp[:self.Nk] = a
        return np.fft.irfft(sp * self.Np, n=self.Np, axis=0)     # (Np, Npad)

    def _nonlinear_coeffs_z(self, that, psi):
        th = self._Gz(that)
        thx = self._Gz(1j * self.kx[:, None] * that)
        thz = self._Gz((self.D @ np.asarray(that).T).T)
        ux = self._Gz((self.D @ np.asarray(psi).T).T)
        uz = self._Gz(-1j * self.kx[:, None] * psi)
        N = -(ux * thx + uz * thz) + uz
        if self.internal_heating != 0.0:
            N = N + self.internal_heating          # internal heating source
        if self.dissipation_number != 0.0:
            # viscous dissipation: (Di/Ra) Phi on the SAME z-padded grid the nonlinearity is
            # assembled on (`dissipation_eta` = the frozen eta(z) field).
            N = N + (self.dissipation_number / float(self.Ra)) \
                * self.dissipation_phi(psi, synth=self._Gz)
        A = np.fft.rfft(N, axis=0) / self.Np                     # (Nk, Npad)
        return (self.Tp @ A.T).T                                 # (Nk, Nz)

    def step(self, that, dt):
        """RBC.step, optionally with z-dealiasing and mirror projection."""
        if self.symmetrize:
            that = self.project_mirror(that)
        if not self.dealias_z:
            out = super().step(that, dt)
        else:
            psi, _ = self.stokes(that)
            Nc = self._nonlinear_coeffs_z(that, psi)
            out = np.zeros_like(that, dtype=complex)
            for i, k in enumerate(self.kx):
                rhs = that[i] / dt + Nc[i]
                out[i] = self.Z.solve(k, -rhs, lam=1.0 / dt)
            self.n_steps += 1
        if self.symmetrize:
            out = self.project_mirror(out)
        return out

    # ------------------------------------------------------------- physics
    def dissipation_eta(self):
        """viscous dissipation: the eta the depth-only Stokes operator consumes (eta(z)).

        `RBCVariableViscosity` freezes eta to the bounded ``eta_grid`` profile,
        which is exactly the field its ``vv`` operator was built from, so the
        dissipation source must use the same one.
        """
        return self.eta_grid

    def stokes(self, that):
        """theta coefficients -> (psi, omega) coefficients, variable eta(z)."""
        psi = np.zeros_like(that, dtype=complex)
        om = np.zeros_like(that, dtype=complex)
        for i, k in enumerate(self.kx):
            if k == 0:
                continue                 # no horizontal buoyancy gradient
            psi[i], om[i] = self.vv.solve(k, that[i])
        self.n_stokes += 1
        return psi, om

    def describe(self):
        return dict(mode='variable-viscosity (depth only)',
                    viscosity_law=self.viscosity_law,
                    deta_T=self.deta_T,
                    eta_min=self.eta_min, eta_max=self.eta_max,
                    eta_clip_lo=self.eta_clip_lo,
                    eta_clip_hi=self.eta_clip_hi,
                    eta_smooth=self.eta_smooth,
                    n_clip_eta=self.n_clip_eta,
                    n_smooth_eta=self.n_smooth_eta,
                    internal_heating=self.internal_heating,
                    dissipation_number=self.dissipation_number,
                    dissipation_active=bool(self.dissipation_number != 0.0),
                    dealias_x=self.dealias, dealias_z=self.dealias_z,
                    Npad=getattr(self, 'Npad', None),
                    symmetrize=self.symmetrize,
                    zform=self.zform,
                    stokes=('direct (dense LU per wavenumber, cached; '
                            '%s z-operator)' % self.zform))


def make_convection(Ra=1e4, Nx=64, Nz=48, Lx=2.0, zsolver='tau',
                    dealias=True, viscosity_law='isoviscous', deta_T=1.0,
                    use_lu=True, dealias_z=False, symmetrize=True,
                    zform='reduced', theta_scheme='explicit',
                    implicit_buoy=True, theta_rtol=1e-10, verbose=False,
                    theta_extrapolate=True, internal_heating=0.0,
                    dissipation_number=0.0, compressibility='boussinesq',
                    eta_min=None, eta_max=None, eta_smooth=None):
    """Factory: build the model for the requested physics mode.

    viscosity_law='isoviscous' and deta_T=1 (the defaults) return exactly the
    validated `rbc_galerkin.RBC`, i.e. the frozen baseline.

    For the variable-viscosity laws the defaults are the production settings
    chosen for that mode:
      * `symmetrize=True`, exact box (mirror) projection, which is the
        identity to roundoff at deta_T = 1 (so no baseline drift) and is
        required at high deta_T;
      * `zform='reduced'`, the block-eliminated 2nd-order Stokes operator
        (conditioning ~N^4 instead of the 4th-order tau's ~N^8).  Use
        `zform='tau4'` for the 4th-order reference operator.
    """
    law = str(viscosity_law).lower()
    ths = str(theta_scheme).lower()
    if law in ISOVISCOUS:
        if verbose and float(deta_T) != 1.0:
            print('  [make_convection] law=%s ignores deta_T=%g' % (law, deta_T))
        if dealias_z:
            raise ValueError('dealias_z is only available in the '
                             'variable-viscosity mode')
        if not symmetrize:
            print('  [make_convection] isoviscous mode ignores symmetrize=False')
        if str(zform).lower() != 'reduced':
            print('  [make_convection] isoviscous mode ignores zform=%r' % zform)
        if ths in ('explicit', 'explicit_be', 'be'):
            return RBC(Ra=Ra, Nx=Nx, Nz=Nz, Lx=Lx, zsolver=zsolver,
                       dealias=dealias, internal_heating=internal_heating,
                       dissipation_number=dissipation_number,
                       compressibility=compressibility)
        raise ValueError("theta_scheme=%r needs a variable-viscosity law "
                         "(use viscosity_law='exp_T_depth')" % (theta_scheme,))
    if law in VV_LAWS:
        if ths in ('explicit', 'explicit_be', 'be'):
            m = RBCVariableViscosity(Ra=Ra, Nx=Nx, Nz=Nz, Lx=Lx,
                                     zsolver=zsolver, dealias=dealias,
                                     viscosity_law=law, deta_T=deta_T,
                                     use_lu=use_lu, dealias_z=dealias_z,
                                     symmetrize=symmetrize, zform=zform,
                                     internal_heating=internal_heating,
                                     dissipation_number=dissipation_number,
                                     compressibility=compressibility,
                                     eta_min=eta_min, eta_max=eta_max,
                                     eta_smooth=eta_smooth)
        elif ths in ('implicit', 'implicit_be', 'theta_implicit'):
            from ..core.implicit_theta import RBCImplicitTheta
            m = RBCImplicitTheta(Ra=Ra, Nx=Nx, Nz=Nz, Lx=Lx, zsolver=zsolver,
                                 dealias=dealias, viscosity_law=law,
                                 deta_T=deta_T, use_lu=use_lu,
                                 dealias_z=dealias_z, symmetrize=symmetrize,
                                 zform=zform, theta_scheme='implicit',
                                 implicit_buoy=implicit_buoy,
                                 theta_rtol=theta_rtol,
                                 internal_heating=internal_heating,
                                 dissipation_number=dissipation_number,
                                 compressibility=compressibility,
                                 eta_min=eta_min, eta_max=eta_max,
                                 eta_smooth=eta_smooth)
        elif ths in ('implicit_bdf2', 'bdf2', 'theta_bdf2'):
            from ..core.implicit_theta import RBCImplicitTheta
            m = RBCImplicitTheta(Ra=Ra, Nx=Nx, Nz=Nz, Lx=Lx, zsolver=zsolver,
                                 dealias=dealias, viscosity_law=law,
                                 deta_T=deta_T, use_lu=use_lu,
                                 dealias_z=dealias_z, symmetrize=symmetrize,
                                 zform=zform, theta_scheme='implicit_bdf2',
                                 implicit_buoy=implicit_buoy,
                                 theta_rtol=theta_rtol,
                                 theta_extrapolate=theta_extrapolate,
                                 internal_heating=internal_heating,
                                 dissipation_number=dissipation_number,
                                 compressibility=compressibility,
                                 eta_min=eta_min, eta_max=eta_max,
                                 eta_smooth=eta_smooth)
        else:
            raise ValueError('unknown theta_scheme %r' % (theta_scheme,))
        if verbose:
            d = m.describe()
            print('  [make_convection] %s  deta_T=%g  eta in [%.6g, %.6g]  '
                  'zform=%s  symmetrize=%s'
                  % (d['mode'], d['deta_T'], d['eta_min'], d['eta_max'],
                     d['zform'], d['symmetrize']))
        return m
    raise ValueError('unknown viscosity_law %r' % (viscosity_law,))


def list_modes():
    return dict(isoviscous=ISOVISCOUS, variable_viscosity=VV_LAWS,
                zforms=ZFORMS, theta_schemes=("explicit", "implicit",
                                              "implicit_bdf2"))


if __name__ == '__main__':
    for law in ('isoviscous', 'exp_T_depth'):
        m = make_convection(Nx=8, Nz=12, viscosity_law=law, deta_T=100.0,
                            verbose=True)
        print('   ', type(m).__name__, m.kx.shape)
