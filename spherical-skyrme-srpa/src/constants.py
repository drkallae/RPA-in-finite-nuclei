from __future__ import annotations

# Units:
# r in fm
# energies in MeV
# hbar*c in MeV*fm
HBARC_MEV_FM = 197.3269804

# Nucleon masses (MeV) (you can refine later if needed)
M_N_MEV = 939.56542052  # neutron mass-energy
M_P_MEV = 938.27208816  # proton mass-energy

# Handy: hbar^2 / (2m) in MeV fm^2
def hbar2_over_2m(MeV_mass: float) -> float:
    return (HBARC_MEV_FM**2) / (2.0 * MeV_mass)


HBAR2_OVER_2MN = hbar2_over_2m(M_N_MEV)
HBAR2_OVER_2MP = hbar2_over_2m(M_P_MEV)