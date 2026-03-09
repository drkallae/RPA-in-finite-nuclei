# RPA in Finite Nuclei

This repository is a learning + research codebase to build a **spherical Skyrme Hartree–Fock (HF)** baseline in coordinate space and then construct a **(S)RPA / RPA** model on top of that baseline (with an emphasis on *separable* residual interactions so the RPA problem is tractable and transparent).

## Goals

1. **HF baseline (spherical, Skyrme-like EDF)**
   - Radial mesh / box boundary conditions
   - Self-consistent densities and single-particle spectrum
   - Coulomb direct (+ optional Slater exchange)
   - Spin–orbit term
   - Validation on a small set of benchmark nuclei (O-16, Ca-40, Ca-48)

2. **Residual interaction and RPA**
   - Start with **separable forces** (and/or separable approximation to the Skyrme residual)
   - Build the RPA matrices (or equivalent response-function formulation)
   - Compute low-lying multipole strength distributions (e.g. E1/E2/E0) and compare trends

3. **Validation + reproducibility**
   - Keep a documented set of reference runs
   - Track physics conventions (units, sign conventions, normalization)
   - Provide scripts to reproduce the benchmark outputs

## Current status (as of 2026-03-09)

- A working spherical HF loop exists and converges for **O-16, Ca-40, Ca-48** on a radial mesh.
- The HF baseline is still under active refinement (especially gradient/t1–t2 sector consistency and radius systematics).
- RPA/SRPA layer is planned next, once the HF baseline is stable and documented.

## Roadmap

### Milestone A — HF baseline stability (in progress)
- [ ] Ensure one consistent Skyrme-EDF convention for:
  - central terms (t0, t3)
  - gradient terms (t1, t2) including correct functional derivatives
  - effective-mass operator implementation
- [ ] Validate convergence behavior (mixing, iteration counts, stability)
- [ ] Produce a benchmark table for O-16, Ca-40, Ca-48:
  - central densities, rms radii, selected s.p. energies, and optionally total energy

### Milestone B — Minimal RPA with separable interaction
- [ ] Define multipole operators and matrix elements in spherical basis
- [ ] Implement separable residual interaction
- [ ] Solve RPA eigenvalue problem (or compute response)
- [ ] Validate against known analytic/benchmark cases

### Milestone C — Physics extensions / systematic studies
- [ ] Additional nuclei
- [ ] Strength distributions (sum rules, centroid energies)
- [ ] Sensitivity to box size / mesh spacing / Skyrme parametrization
- [ ] Optional pairing (HF-BCS / HFB) if needed later

## How to run (typical workflow)

### 1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies
If you have a requirements file:
```bash
pip install -r requirements.txt
```

If not, typical baseline requirements are:
- numpy
- (optional) scipy, matplotlib

### 3) Run example HF scripts
Examples (exact module names may differ depending on what’s in `src/`):
```bash
python3 -m src.run_hf_o16
python3 -m src.run_hf_ca40
python3 -m src.run_hf_ca48
```

The scripts typically print:
- convergence info
- particle-number checks
- density and rms-radius quick checks
- occupied single-particle levels

and save arrays to an `out/` directory.

## Output conventions (important)

This code typically works with:
- radial coordinate `r` in **fm**
- densities `rho(r)` in **fm^-3**
- energies/potentials in **MeV**

Rms radius computed from a 3D density is:
\[
r_\mathrm{rms} = \sqrt{\frac{\int 4\pi r^4 \rho(r)\,dr}{\int 4\pi r^2 \rho(r)\,dr}}
\]

When comparing to experiment:
- experimental values are usually **charge radii** \(r_\mathrm{ch}\)
- code typically computes **point-proton radii** \(r_p\) from \(\rho_p(r)\) unless finite-size corrections are added

## Repository structure (expected)

The project is organized roughly as:

- `src/`
  - HF / mesh / Skyrme EDF code
  - run scripts (e.g. `run_hf_o16.py`, etc.)
- `out/`
  - generated outputs (should be gitignored)
- `docs/` (optional)
  - references, derivations, notes, validation tables

If you add new scripts, prefer putting them under `src/` so they can be run with `python -m ...`.

## Development notes / guidelines

- Prefer small, testable increments: change one physics piece at a time and record its effect on benchmark nuclei.
- Keep “debug knobs” (mixing, gradient scaling, etc.) explicit and printed at runtime for reproducibility.
- When changing physics conventions (e.g. gradient sector sign/factor), document it in `docs/` and record before/after benchmark outputs.

## References (to be expanded)

Suggested references to cite/align conventions with (add to `docs/` as you go):
- Standard Skyrme EDF derivations (time-even sector)
- Coordinate-space HF implementations in spherical symmetry
- RPA formalism in nuclear physics (including separable interactions)

---

If you want, I can tailor this README to the *exact* directory layout in `drkallae/RPA-in-finite-nuclei` (by reading the repo contents) and add the real script names and commands you currently have.
