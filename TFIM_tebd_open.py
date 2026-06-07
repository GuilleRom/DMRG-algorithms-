"""TEBD for open quantum systems — vectorized density matrix (purification) strategy.

The density matrix rho is encoded as a vectorized MPS:

    |rho>> = sum_{mn} rho_{mn} |m>_phys |n>_anc

with super-site index alpha = m*d + n, local dimension ds = d^2.

The Lindblad master equation

    drho/dt = -i[H,rho] + sum_k gamma_k (L_k rho L_k^dag - 1/2 {L_k^dag L_k, rho})

becomes a linear equation for |rho>>, evolved with 2nd-order Suzuki-Trotter TEBD.

Physical observables are recovered via

    Tr(O rho) = <<I| O x I |rho~>> / <<I|rho~>>

where |rho~>> is the Frobenius-normalised MPS and |I>> = x_k sum_n|n,n>_k.
"""

# Copyright 2025 Guillermo Romero, Universidad de Santiago de Chile

import numpy as np
from scipy.linalg import expm
from a_mps import split_truncate_theta


# ---------------------------------------------------------------------------
# TEBD machinery
# ---------------------------------------------------------------------------

def calc_U_bonds_open(K_bonds, dt):
    """Exponentiate Lindblad generators into 2nd-order Trotter TEBD gates.

    Even-index bonds get exp(dt/2 * K), odd-index bonds get exp(dt * K).
    run_TEBD applies [even, odd, even] per step -> 2nd-order accuracy.
    """
    U_bonds = {}
    for i, K in enumerate(K_bonds):
        ds = K.shape[0]
        K_mat = K.reshape(ds * ds, ds * ds)
        step = dt / 2 if i % 2 == 0 else dt
        U_bonds[i] = expm(step * K_mat).reshape(ds, ds, ds, ds)
    return U_bonds


def update_bond(psi, i, U_bond, chi_max, eps):
    """Apply U_bond on super-sites i and i+1 and truncate.

    Returns
    -------
    log_norm : float
        Log of the SVD norm factor divided out (for norm tracking).
    trunc_err : float
        Truncation error eps = 1 - ||Y_kept||^2 / ||Y_full||^2,
        i.e. the squared-norm fraction discarded by the truncation.
    """
    j = (i + 1) % psi.L
    theta = psi.get_theta2(i)
    Utheta = np.tensordot(U_bond, theta, axes=([2, 3], [1, 2]))
    Utheta = np.transpose(Utheta, [2, 0, 1, 3])
    chivL, dL, dR, chivR = Utheta.shape
    from scipy.linalg import svd as _svd
    _, Y, _ = _svd(Utheta.reshape(chivL * dL, dR * chivR), full_matrices=False)
    norm_old = np.sum(np.square(Y))
    Ai, Sj, Bj, _ = split_truncate_theta(Utheta, chi_max, eps)
    norm_new = np.sum(np.square(np.sort(Y)[::-1][:Sj.shape[0]]))
    log_norm = np.log(np.sqrt(norm_new))
    trunc_err = 1.0 - norm_new / norm_old
    Gi = np.tensordot(np.diag(psi.Ss[i] ** (-1)), Ai, axes=[1, 0])
    psi.Bs[i] = np.tensordot(Gi, np.diag(Sj), axes=[2, 0])
    psi.Ss[j] = Sj
    psi.Bs[j] = Bj
    return log_norm, trunc_err


def run_TEBD(psi, U_bonds, N_steps, chi_max, eps):
    """Evolve psi for N_steps with 2nd-order Trotter [even, odd, even]."""
    Nbonds = psi.L - 1 if psi.bc == 'finite' else psi.L
    for _ in range(N_steps):
        for k in [0, 1, 0]:
            for i_bond in range(k, Nbonds, 2):
                update_bond(psi, i_bond, U_bonds[i_bond], chi_max, eps)


def accum_norm(psi, U_bonds, chi_max, eps, current_log_norm=0.0):
    """Run one TEBD step and return the updated log-norm and max truncation error.

    Tracks the product of all SVD norm factors divided out during truncation.
    The physical Frobenius norm of rho is:

        ||rho(t)||_F = exp(log_norm) / <<I|rho~(t)>>

    and the trace is:

        Tr(rho(t)) = exp(log_norm) * <<I|rho~(t)>>

    Parameters
    ----------
    current_log_norm : float
        Log of the accumulated norm from all previous steps (start at 0.0).

    Returns
    -------
    new_log_norm : float
        Updated log-accumulated norm after this step.
    max_trunc_err : float
        Maximum truncation error over all bond updates in this step.
        eps_bond = 1 - ||Y_kept||^2 / ||Y_full||^2.
    """
    Nbonds = psi.L - 1 if psi.bc == 'finite' else psi.L
    log_norm = current_log_norm
    max_trunc_err = 0.0
    for k in [0, 1, 0]:
        for i_bond in range(k, Nbonds, 2):
            ln, te = update_bond(psi, i_bond, U_bonds[i_bond], chi_max, eps)
            log_norm += ln
            max_trunc_err = max(max_trunc_err, te)
    return log_norm, max_trunc_err


# ---------------------------------------------------------------------------
# Observable measurement — correct vectorized-density-matrix formula
# ---------------------------------------------------------------------------

def _identity_vec(d):
    """Local identity vector: I_k[alpha] = delta_{p,a} for alpha = p*d + a."""
    ds = d * d
    v = np.zeros(ds)
    for n in range(d):
        v[n * d + n] = 1.0
    return v


def _contract_identity(psi, I_vec, op_site=None):
    """Compute <<I| [op at site i] |rho~>> via left-to-right transfer matrices.

    op_site : None  -> compute <<I|rho~>>
              (i, M) -> insert M (ds x ds) at site i before contracting.
    """
    T = np.ones((1, 1), dtype=complex)
    for k in range(psi.L):
        B = psi.Bs[k]                      # (chi_L, ds, chi_R)
        if op_site is not None and k == op_site[0]:
            M = op_site[1]
            MB = np.tensordot(M, B, axes=([1], [1]))   # ds chi_L chi_R
            MB = MB.transpose(1, 0, 2)                 # chi_L ds chi_R
            T = np.einsum('ij,a,jal->il', T, I_vec, MB)
        else:
            T = np.einsum('ij,a,jal->il', T, I_vec, B)
    return T[0, 0]


def measure_phys(psi, op_phys, d):
    """Compute Tr(O rho) at every site using <<I|O x I|rho~>> / <<I|rho~>>.

    op_phys : (d, d) single-site physical operator.
    Returns np.ndarray of shape (L,).
    """
    I_vec = _identity_vec(d)
    O_super = np.kron(op_phys, np.eye(d))          # (ds, ds)
    denom = _contract_identity(psi, I_vec)
    return np.array([
        np.real(_contract_identity(psi, I_vec, op_site=(i, O_super)) / denom)
        for i in range(psi.L)
    ])


def frobenius_norm(psi, d):
    """Return ||rho||_F = 1 / <<I|rho~>>.

    TEBD normalises ||rho~||_F = 1, so the physical density matrix is rho = c*rho~
    with c = ||rho||_F.  For a pure state c = 1; as dephasing mixes the state
    c decreases toward 1/sqrt(d) (maximally mixed).  Tr(rho) = 1 is always preserved;
    the observable formula divides by <<I|rho~>> = 1/c to correct for this.
    """
    return 1.0 / np.real(_contract_identity(psi, _identity_vec(d)))


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def TFIM_TEBD_open(L, J, B, gamma, tmax, dt,
                   chi_max=50, eps=1e-8, init_state='+x'):
    """Simulate the open TFIM with dephasing noise using vectorized-MPS TEBD.

    Hamiltonian : H = -J sum sigma^x_i sigma^x_{i+1} + B sum sigma^z_i
    Jump operator: L_i = sigma^z_i  (local dephasing)
    Off-diagonal elements decay as  rho_{01}(t) = rho_{01}(0) * exp(-2*gamma*t).

    Parameters
    ----------
    L          : number of physical sites
    J, B       : Ising coupling and transverse field
    gamma      : dephasing rate
    tmax, dt   : total time and time step
    init_state : '+x'   -> rho = |+x><+x|^L  (matches closed-system default)
                 'up'   -> rho = |0><0|^L
                 'mixed' -> rho = (I/d)^L
    """
    import a_mps
    import b_model

    d = 2
    Nsteps = int(tmax / dt)
    tspan = np.linspace(0.0, tmax, Nsteps)

    sz = np.array([[1., 0.], [0., -1.]])
    sx = np.array([[0., 1.], [1., 0.]])

    M = b_model.TFIModelOpen(L=L, J=J, B=B,
                              jump_ops=[sz], gammas=[gamma],
                              bc='finite')

    psi = a_mps.init_purification_MPS(L, d, bc='finite', state=init_state)
    U_bonds = calc_U_bonds_open(M.K_bonds, dt)

    Sz_t, Sx_t, S_ent_t, frob_t, trace_t, trunc_t = [], [], [], [], [], []
    log_norm = 0.0

    for n in range(Nsteps):
        if n % max(1, Nsteps // 10) == 0:
            print(f"t = {tspan[n]:.2f},  bond dims = {psi.get_chi()}")

        log_norm, max_trunc = accum_norm(psi, U_bonds, chi_max, eps, log_norm)

        Sz_t.append(measure_phys(psi, sz, d))
        Sx_t.append(measure_phys(psi, sx, d))
        S_ent_t.append(psi.entanglement_entropy())
        frob_t.append(frobenius_norm(psi, d))
        bra_I = np.real(_contract_identity(psi, _identity_vec(d)))
        trace_t.append(np.exp(log_norm) * bra_I)
        trunc_t.append(max_trunc)

    Sz_t = np.array(Sz_t)
    Sx_t = np.array(Sx_t)
    S_ent_t = np.array(S_ent_t)
    frob_t = np.array(frob_t)
    trace_t = np.array(trace_t)
    trunc_t = np.array(trunc_t)

    _plot(tspan, Sz_t, Sx_t, S_ent_t, frob_t, trace_t, trunc_t, L, J, B, gamma, d)
    return tspan, Sz_t, Sx_t, S_ent_t, trunc_t


def _plot(tspan, Sz_t, Sx_t, S_ent_t, frob_t, trace_t, trunc_t, L, J, B, gamma, d):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    colors = ('#1090BF', '#DD5400', '#8500D1', '#3BB02F', 'k')

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(f'Open TFIM — vectorized-MPS TEBD  '
                 f'(L={L}, J={J}, B={B}, gamma={gamma})')

    ax = axes[0, 0]
    ax.plot(tspan, np.mean(Sz_t, axis=1), color=colors[0], label=r'$\langle\sigma^z\rangle$')
    ax.plot(tspan, np.mean(Sx_t, axis=1), color=colors[1], label=r'$\langle\sigma^x\rangle$')
    ax.set_xlabel('t'); ax.set_ylabel('avg magnetization'); ax.legend()

    ax = axes[0, 1]
    ax.plot(tspan, frob_t, color=colors[2], label=r'$\|\rho\|_F$')
    ax.plot(tspan, trace_t, color=colors[4], ls='--', lw=1, label=r'$\mathrm{Tr}(\rho)$')
    ax.set_xlabel('t'); ax.legend()
    ax.set_title(r'Purity $\|\rho\|_F$ and trace $\mathrm{Tr}(\rho)$ (should stay $\equiv 1$)')

    ax = axes[0, 2]
    ax.semilogy(tspan, np.maximum(trunc_t, 1e-16), color=colors[4], lw=0.9)
    ax.set_xlabel('t'); ax.set_ylabel(r'$\varepsilon_{\rm trunc}$')
    ax.set_title(r'Max truncation error per step'
                 '\n' r'$\varepsilon = 1 - \|Y_{\rm kept}\|^2/\|Y\|^2$')

    ax = axes[1, 0]
    ax.plot(tspan, S_ent_t, color=colors[3], lw=0.8)
    ax.set_xlabel('t'); ax.set_ylabel('bond entanglement entropy')
    ax.set_title('MPS entanglement entropy')

    ax = axes[1, 1]
    ax.plot(tspan, np.mean(Sx_t, axis=1), color=colors[1], label='TEBD')
    if gamma > 0:
        ax.plot(tspan, np.exp(-2 * gamma * tspan), color=colors[4],
                ls='--', lw=1, label=r'$e^{-2\gamma t}$')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\langle\sigma^x\rangle$')
    ax.set_title('Dephasing check'); ax.legend()

    ax = axes[1, 2]
    ax.semilogy(tspan, np.maximum(np.cumsum(trunc_t), 1e-16),
                color=colors[0], lw=0.9)
    ax.set_xlabel('t'); ax.set_ylabel(r'$\sum_t \varepsilon_{\rm trunc}$')
    ax.set_title('Cumulative truncation error')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    TFIM_TEBD_open(L=6, J=1.0, B=0.5, gamma=0.3, tmax=4.0, dt=0.01)
