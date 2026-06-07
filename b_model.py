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


class TFIModelOpen:
    """Open TFIM: H = -J Σ σ^x_i σ^x_{i+1} + B Σ σ^z_i with Lindblad jump operators.

    Builds the Lindblad generator K_bonds for the vectorized density matrix MPS.
    Each K_bond acts on super-sites (i, i+1) with super-index α = m*d + n (ds = d²).

    The Lindblad equation in vectorized form: d|ρ⟩⟩/dt = K |ρ⟩⟩
    where K = K_coh + K_diss:
      K_coh = -i(H ⊗ I - I ⊗ H*)    (coherent part)
      K_diss = Σ_k γ_k D[L_k]         (dissipative part)
    with D[L]|ρ⟩⟩ = (L ⊗ L* - ½ L†L ⊗ I - ½ I ⊗ (L†L)^T)|ρ⟩⟩
    """

    def __init__(self, L, J, B, jump_ops, gammas, bc='finite'):
        assert bc in ['finite', 'infinite']
        self.L, self.d, self.bc = L, 2, bc
        self.J, self.B = J, B
        self.jump_ops = jump_ops   # list of (d, d) arrays
        self.gammas = gammas       # list of floats
        self._init_K_bonds()

    def _init_K_bonds(self):
        d = self.d
        ds = d * d
        sx = np.array([[0., 1.], [1., 0.]])
        sz = np.array([[1., 0.], [0., -1.]])
        Id = np.eye(d)
        I_ds = np.eye(ds)

        nbonds = self.L - 1 if self.bc == 'finite' else self.L
        K_bonds = []

        for i in range(nbonds):
            # Weight for sharing single-site terms across bonds (finite bc)
            if self.bc == 'finite':
                wL = 1.0 if i == 0 else 0.5
                wR = 1.0 if (i + 1 == self.L - 1) else 0.5
            else:
                wL = wR = 0.5

            # Two-site Hamiltonian piece (shape d² × d²)
            H_bond = (-self.J * np.kron(sx, sx)
                      + wL * self.B * np.kron(sz, Id)
                      + wR * self.B * np.kron(Id, sz))

            # Coherent Lindblad superoperator in (ket, bra) flat basis:
            #   K_coh = -i(H ⊗ I_{d²} - I_{d²} ⊗ H*)
            K_KB = -1j * (np.kron(H_bond, I_ds) - np.kron(I_ds, H_bond.conj()))

            # Dissipative superoperator in (ket, bra) flat basis
            for L_op, gamma in zip(self.jump_ops, self.gammas):
                Lc = L_op.conj()
                LdagL = L_op.conj().T @ L_op

                # Jump and decay matrices extended to the two-site ket space
                L_left = np.kron(L_op, Id)       # site i
                Lc_left = np.kron(Lc, Id)
                LdagL_left = np.kron(LdagL, Id)

                L_right = np.kron(Id, L_op)      # site i+1
                Lc_right = np.kron(Id, Lc)
                LdagL_right = np.kron(Id, LdagL)

                D_left = (np.kron(L_left, Lc_left)
                          - 0.5 * np.kron(LdagL_left, I_ds)
                          - 0.5 * np.kron(I_ds, LdagL_left.T))

                D_right = (np.kron(L_right, Lc_right)
                           - 0.5 * np.kron(LdagL_right, I_ds)
                           - 0.5 * np.kron(I_ds, LdagL_right.T))

                K_KB += gamma * (wL * D_left + wR * D_right)

            # Convert K from (ket, bra) ordering to super-site (interleaved) ordering.
            # KB ordering:  flat = k * d² + b  where k=(m_i, m_{i+1}), b=(n_i, n_{i+1})
            # SS ordering:  flat = α_i * ds + α_{i+1}  where α_k = m_k*d + n_k
            #
            # Reshape to 8D axes: (m_i', m_{i+1}', n_i', n_{i+1}', m_i, m_{i+1}, n_i, n_{i+1})
            # Transpose to SS:    (m_i', n_i', m_{i+1}', n_{i+1}', m_i, n_i, m_{i+1}, n_{i+1})
            K_8d = K_KB.reshape(d, d, d, d, d, d, d, d).transpose(0, 2, 1, 3, 4, 6, 5, 7)
            K_bonds.append(K_8d.reshape(ds, ds, ds, ds))

        self.K_bonds = K_bonds
