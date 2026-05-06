"""Implementing the Jaynes-Cummings-Hubbard model."""
# Copyright 2018 TeNPy Developers
# Copyright 2019 Guillermo Romero, Universidad de Santiago de Chile
# guillermo.romero@usach.cl
import numpy as np
import scipy




class TFIModel:
    """Transverse-Field Ising Model with linear ramp of the field.

    Hamiltonian:
        H = -J ∑ σ^x_i σ^x_{i+1} + ∑ B(t) σ^z_i

    where the transverse field ramps linearly in time:
        B(t) = Bi + (Bf - Bi) * (time / tf)
    """

    def __init__(self, L, J, Bi, Bf, time, tf, bc='finite'):
        assert bc in ['finite', 'infinite']
        self.L, self.d, self.bc = L, 2, bc
        self.J, self.Bi, self.Bf, self.time, self.tf = J, Bi, Bf, time, tf

        # Pauli matrices
        self.sz = np.array([[1., 0.], [0., -1.]])
        self.sx = np.array([[0., 1.], [1., 0.]])
        self.Ip = np.eye(2)

        # Linear interpolation of the field
        #B_t = self.Bi + (self.Bf - self.Bi) * (time / tf)
        
        # Fixed transverse field
        B_t = Bi
        
        # On-site Hamiltonian term
        self.hIM = B_t * self.sz
        
        # H0 Hamiltonian
        self.h0 = J * np.kron(self.sx,self.sx)

        # Initialize 2-site bond Hamiltonians
        self.init_H_bonds()

    def init_H_bonds(self):
        """Initialize `H_bonds` Hamiltonian. Called by __init__()."""
        hIM, sx = self.hIM, self.sx
        d = self.d
        nbonds = self.L - 1 if self.bc == 'finite' else self.L
        H_list = []

        for i in range(nbonds):
            hIML = hIMR = 0.5 * hIM
            if self.bc == 'finite':
                if i == 0:
                    hIML = hIM
                if i + 1 == self.L - 1:
                    hIMR = hIM

            # Two-site Hamiltonian piece
            H_bond = self.J * np.kron(sx, sx) + np.kron(hIML, self.Ip) + np.kron(self.Ip, hIMR)
            H_list.append(np.reshape(H_bond, [d, d, d, d]))

        self.H_bonds = H_list
