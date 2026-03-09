# Spherical Skyrme HF (minimal) — Week 1/2/3 notes

This repository contains a minimal spherical Hartree–Fock (HF) solver for nuclei on a radial mesh, built up in weekly milestones.

Current goals:
- numerically stable self-consistent HF iterations,
- correct particle-number normalization,
- reasonable density/radius for O-16 after pragmatic stabilization,
- correct **spin–orbit shell ordering** (e.g. `1p_{3/2}` below `1p_{1/2}`),
- produce saved arrays + plots for validation.

---

## Week 1: mesh + densities + basic radial operators (foundation)

### 1) Radial mesh
We work on a uniform spherical radial mesh:
- `r in [0, Rmax]`
- spacing `dr`
- utilities for 3D integrals of spherical densities:
  \[
  \int \rho(\mathbf{r})\,d^3r = 4\pi\int_0^\infty \rho(r)\,r^2\,dr.
  \]

(Implemented in `src/mesh.py`, e.g. `RadialMesh.uniform(...)` and `integrate_3d(...)`.)

### 2) Initial “Fermi-ish” densities
HF iterations are initialized with a smooth Woods–Saxon / Fermi-like profile:
\[
\rho(r)\propto \frac{1}{1+\exp((r-r_0)/a)}
\]
then rescaled to exactly match target particle numbers:
- `N = ∫ rho_n d^3r`
- `Z = ∫ rho_p d^3r`

This provides a stable starting point for self-consistency.

### 3) Radial operators used later
We implement spherical radial differential operators required by later stages:
- spherical Laplacian acting on a spherical function:
  \[
  \nabla^2 f(r)=\frac{1}{r^2}\frac{d}{dr}\left(r^2\frac{df}{dr}\right)
  \]
Used for the pragmatic surface stabilization term in Week 3.

---

## Week 2: minimal HF (t0+t3) + orbital filling + densities

### 1) Radial bound-state solver in a spherical box
We solve the reduced radial equation for each `l` channel:
\[
\left[
-\frac{\hbar^2}{2m}\frac{d^2}{dr^2}
+ \frac{\hbar^2}{2m}\frac{l(l+1)}{r^2}
+ U(r)
\right] u(r) = E\,u(r)
\]
with box boundary conditions:
- `u(0) = 0`
- `u(Rmax) = 0`

Implementation: `solve_radial_bound_states_box(...)` in `src/hf/radial_solve.py`.

### 2) Occupation filling (no spin–orbit)
Orbitals are filled for each species in a spherical, no-spin-orbit model.

Degeneracy per `l` shell:
\[
g_l = 2(2l+1)
\]
(spin included, but no `j` splitting)

File: `src/hf/fill.py`
- `fill_spherical_no_spin_orbit(...)`

### 3) Density from orbitals
Spherical density from filled orbitals:
\[
\rho(r) = \sum_a \frac{\text{occ}_a}{4\pi}\left|\frac{u_a(r)}{r}\right|^2
\]
Implemented in:
- `density_from_orbitals(mesh, orbitals)`

### 4) Minimal Skyrme-like mean field (t0+t3 only)
Week 2 HF uses a simplified Skyrme central mean field containing only `t0+t3` density dependence.
This is enough to build a self-consistent loop, but it is *not yet* a complete Skyrme HF EDF.

(Implemented in `src/hf/skyrme_meanfield_min.py` as `central_mean_field_t0_t3(...)`.)

---

## Week 3: performance + Coulomb + stabilization + spin–orbit validation

### 1) Performance improvement: tridiagonal eigen-solve
Originally the Hamiltonian was assembled as a dense matrix and solved with `eigh(H)`.

It was replaced with `scipy.linalg.eigh_tridiagonal(...)` using:
- main diagonal `diag`
- off-diagonal `off`

and selecting only the lowest `n_states` eigenpairs.

**Impact:** runtime dropped dramatically (dense diagonalization bottleneck removed), enabling fast HF iteration and parameter scans.

---

### 2) Coulomb (optional, protons only)
Proton mean field can include:
- direct Coulomb potential
- optional Slater exchange term

Flags:
- `use_coulomb=True/False`
- `coulomb_exchange=True/False`

Implementation:
- `src/hf/coulomb.py`

---

### 3) Pragmatic surface stabilization (placeholder for missing t1/t2 physics)
To prevent unphysical collapse with the minimal `t0+t3` EDF, we add:
\[
U_{\text{surf}}(r) = -C_{\text{surf}}\,\nabla^2 \rho(r)
\]
applied equally to neutrons and protons.

This mimics missing gradient/effective-mass physics (normally coming from the full Skyrme `t1/t2` + kinetic-density sector).

Controlled by:
- `C_surf` argument in `run_hf_minimal_t0t3(...)`

---

### 4) Spin–orbit (spherical, jj-coupling)
To validate shell ordering we add an optional spin–orbit term.

We compute a radial spin–orbit strength from density gradients:
\[
U_{ls,q}(r)=\frac{W_0}{2}\frac{1}{r}\frac{d}{dr}\big(\rho(r)+\rho_q(r)\big)
\]
where `q = n,p` and `W0` is a tunable constant (Skyrme-like units).

For each `(l,j)` channel we solve with:
\[
U_{\text{eff}}(r) = U_q(r) + \langle \mathbf{l}\cdot\mathbf{s}\rangle\,U_{ls,q}(r)
\]
and
\[
\langle \mathbf{l}\cdot\mathbf{s}\rangle
= \frac12\left[j(j+1)-l(l+1)-\frac34\right].
\]

#### jj channel handling
- `l=0`: only `j=1/2`
- `l>0`: `j=l-1/2` and `j=l+1/2` (stored as `j2 = 2j` integers)

#### jj-filling
Each `(l,j)` orbital has degeneracy:
\[
g_j = 2j+1
\]
implemented as `g = j2 + 1`.

File: `src/hf/fill.py`
- `fill_spherical_jj(...)`

Gradient helper:
- `src/hf/radial_derivatives.py` (`d_dr(...)`)

---

## Typical O-16 validation run (Week 3)
A representative converged run (with tuned surface term + Coulomb + spin–orbit) produced:

- `N check = 8`, `Z check = 8`
- central density near saturation (~0.167 fm^-3)
- RMS radii around ~2.78 fm (ballpark reasonable for O-16 in this simplified model)
- correct p-shell ordering:
  - `1p_{3/2}` more bound than `1p_{1/2}`
  - occupations `4` and `2` consistent with degeneracies

---

## How to run

### HF validation script (O-16)
Run the Week 3 validation driver:

```bash
python3 -m src.run_week3_hf_validate_o16
```

This should save arrays in:
- `out/week3_o16_hf_coulomb_validate/`
  - `r_fm.npy`
  - `rho_n_fm-3.npy`
  - `rho_p_fm-3.npy`
  - `U_n_MeV.npy`, `U_p_MeV.npy` (if saved)
  - `Uls_n_MeV.npy`, `Uls_p_MeV.npy` (if spin–orbit enabled and saved)

---

## Plotting outputs
Plot density and potentials:

```bash
python3 -m src.plot_week3_density_potentials
```

Expected outputs in the same `out/...` folder:
- `density.png`
- `potentials.png`
- `spin_orbit_strength.png` (if `Uls_*.npy` exist)

---

## Known limitations / next steps
1) **EDF is incomplete:** This stage uses `t0+t3` plus an ad-hoc surface term.
   - For more physical radii/surface properties without tuning `C_surf`, implement Skyrme `t1/t2` + kinetic-density `τ` terms.
2) **Spin–orbit strength:** `W0` is currently a tuneable parameter; in a full Skyrme implementation it should come from the chosen Skyrme set (e.g. SLy4).
3) **Quantitative calibration:** for true comparison to experiment (charge radii, binding energies), additional physics and consistent conventions are required.

---

## File map (key)
- `src/mesh.py` — radial mesh + 3D integrals
- `src/hf/radial_ops.py` — spherical Laplacian
- `src/hf/radial_solve.py` — radial bound-state solver (tridiagonal eigensolve)
- `src/hf/skyrme_meanfield_min.py` — minimal `t0+t3` mean field
- `src/hf/fill.py` — orbital filling (no-SO and jj) + density builder
- `src/hf/coulomb.py` — Coulomb direct + Slater exchange
- `src/hf/radial_derivatives.py` — `d/dr` helper for spin–orbit gradients
- `src/hf/hf_minimal.py` — self-consistent HF loop (t0/t3 + surface + Coulomb + optional spin–orbit)
- `src/run_week3_hf_validate_o16.py` — O-16 validation driver
- `src/plot_week3_density_potentials.py` — plotting script