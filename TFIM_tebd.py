"""Code implementing the time evolving block decimation (TEBD)."""
" We implement the quantum Ising model with time-dependent magnetic field"

# Copyright 2018 TeNPy Developers
# Copyright 2025 Guillermo Romero, Universidad de Santiago de Chile
# guillermo.romero@usach.cl

import numpy as np
from scipy.linalg import expm
from scipy import sparse
from a_mps import split_truncate_theta
import pickle
#from tenpy.networks.mps import MPS
#from tenpy.models.hubbard import BoseHubbardChain
#from tenpy.algorithms import tebd
#from tenpy.linalg import np_conserved as npc
#from scipy.fft import fft, ifft, fftshift
#from tenpy.tools.hdf5_io import Hdf5Exportable
#from scipy.linalg import svd
#from tenpy.networks.mps import MPS




def overlap(psiA, psiB, L):
    "Overlap between MPS states psiA and psiB via transfer matrix"
    A = psiA[0]
    B = psiB[0]
    chiA = A.shape[0]
    chiB = B.shape[0]
    L = len(psiB)    
    T = np.tensordot(A, np.conj(B), axes=[1, 1])  # vL [i] vR, vL* [i*] vR*
    T = np.transpose(T, [0, 2, 1, 3])  # vL vL* vR vR*
    for i in range(1, L):
        A = psiA[i]
        B = psiB[i]
        T = np.tensordot(T, A, axes=[2, 0])  # vL vL* [vR] vR*, [vL] i vR
        T = np.tensordot(T, np.conj(B), axes=[[2, 3], [0, 1]])
        # vL vL* [vR*] [i] vR, [vL*] [i*] vR*
    T = np.reshape(T, (chiA*chiB, chiA*chiB))
    # Compute the transfer matrix trace
    return np.trace(T)


def calc_U_bonds(H_bonds, dt, L):
    """Given the H_bonds, calculate ``U_bonds[i] = expm(-dt*H_bonds[i])``.

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
    Note that no imaginary 'i' is included, thus real `dt` means 'imaginary time' evolution!
    """
    "We use 2nd order Suzuki-Trotter expansion of the evolution operator"
    d = H_bonds[0].shape[0]
    U_bonds = {}
    
    #second-order Suzuki-Trotter ##
    for i in range(L-1):
        H = np.reshape(H_bonds[i], [d * d, d * d])
        if i % 2 == 0:
            U = expm(-1j * dt/2 * H)
            U_bonds[i] = np.reshape(U, [d, d, d, d])
        else: 
            U = expm(-1j*dt * H)
            U_bonds[i] = np.reshape(U, [d, d, d, d])
    return U_bonds

def run_TEBD(psi, U_bonds, N_steps, chi_max, eps):
    """Evolve for `N_steps` time steps with TEBD."""
    Nbonds = psi.L - 1 if psi.bc == 'finite' else psi.L
    assert len(U_bonds) == Nbonds
    for n in range(N_steps):
        for k in [0, 1, 0]:  # even, odd
            for i_bond in range(k, Nbonds, 2):
                update_bond(psi, i_bond, U_bonds[i_bond], chi_max, eps)
                
TruncE=[]
def update_bond(psi, i, U_bond, chi_max, eps):
    """Apply `U_bond` acting on i,j=(i+1) to `psi`."""
    j = (i + 1) % psi.L
    # construct theta matrix
    theta = psi.get_theta2(i)  # vL i j vR
    # apply U
    Utheta = np.tensordot(U_bond, theta, axes=([2, 3], [1, 2]))  # i j [i*] [j*], vL [i] [j] vR
    Utheta = np.transpose(Utheta, [2, 0, 1, 3])  # vL i j vR
    # split and truncate
    Ai, Sj, Bj, trunc = split_truncate_theta(Utheta, chi_max, eps)
    'Extract the accumulated error truncation'
    TruncE.append(trunc) 
    # put back into MPS
    Gi = np.tensordot(np.diag(psi.Ss[i]**(-1)), Ai, axes=[1, 0])  # vL [vL*], [vL] i vC
    psi.Bs[i] = np.tensordot(Gi, np.diag(Sj), axes=[2, 0])  # vL i [vC], [vC] vC
    psi.Ss[j] = Sj  # vC
    psi.Bs[j] = Bj  # vC j vR
    
    # Compute and store the two-site reduced density matrix
    rho_ij = psi.two_site_rdm(i)
    # Example: compute local entanglement entropy or store it
    eigvals = np.linalg.eigvalsh(rho_ij)
    S_ij = -np.sum(eigvals * np.log(eigvals + 1e-15))
    # You can store these for analysis later if desired:
    # psi.rdm_list.append((i, rho_ij, S_ij))

    


# Computing the ground state with DMRG
# One can use it to initialize the system for a quench dynamics
'''
def TEBD_gs_JCH_finite(L, J, g, omega, delta, mu, nmax):
    print("finite TEBD, imaginary time evolution, Jaynes-Cummings-Hubbard model")
    print("L={L:d}, J={J:.2f}, g={g:.2f}, omega={omega:.2f}, delta={delta:.2f}, nmax={nmax:.2f}".format(L=L, J=J, g=g, omega=omega, delta=delta, nmax=nmax))
    import a_mps
    import b_model
    M = b_model.JCHModel(L=L, J=J, g=g, omega=omega, delta=delta, mu=mu, nmax=nmax, bc='finite')
    #psi = a_mps.init_FM_MPS(M.L, M.d, M.bc)
    psi = a_mps.init_Gen_MPS(M.L, M.d, M.bc)
    dt = 1.e-1
    #for dt in [0.1, 0.01, 0.001, 1.e-4, 1.e-5]:
    U_bonds = calc_U_bonds(M.H_bonds, dt)
    run_TEBD(psi, U_bonds, N_steps=50, chi_max=20, eps=1.e-8)
    E = np.sum(psi.bond_expectation_value(M.H_bonds))
    print("dt = {dt:.5f}: E = {E:.13f}".format(dt=dt, E=E))
    print("final bond dimensions: ", psi.get_chi())
        
    return E, psi
'''


def TFIM_TEBD(L, J, Bi, Bf, tmax, dt):
    print("finite TEBD, real time evolution, Jaynes-Cummings lattice")
    print("L={L:d}, Bi={Bi:.2f}, Bf={Bf:.2f}, tmax={tmax:.2f}, dt={dt:.3f}".format(L=L, Bi=Bi, Bf=Bf, tmax=tmax, dt=dt))
 
    
    import a_mps
    import b_model
    Nsteps = int(tmax/dt)
    scale = 1/J
    print(dt*scale)
    #tspan = scale * np.arange(Nsteps*dt,tmax+Nsteps*dt,Nsteps*dt)
    tspan = scale * np.linspace(0.0, tmax, Nsteps)
    print(len(tspan))
    
    M = b_model.TFIModel(L=L, J=J, Bi=Bi, Bf=Bf, time = tspan[0], tf=tmax, bc='finite')
    """Return a ferromagnetic MPS (= product state with all spins up)"""
    psi = a_mps.init_FM_MPS(M.L, M.d, M.bc)
 

  
    
    S = []
    H0 = []
    Sz = []
    Sx = []
    TruncError = np.zeros(len(tspan))



    "init_prod_MPS"
    d = 2
    psiA = {}
    psiB = {}
    A = np.zeros([1,d,1])
    A[0,0,0] = 1./np.sqrt(2)
    A[0,1,0] = 1./np.sqrt(2)
    B = np.zeros([1,d,1])
    B[0,0,0] = 1./np.sqrt(2)
    B[0,1,0] = -1./np.sqrt(2)
    for i in range(L):
        psiA[i] =  A
        psiB[i] =  B
    
    PA = np.zeros(len(tspan))
    PB = np.zeros(len(tspan))
    Gamma = np.zeros(len(tspan))
    L1 = np.zeros(len(tspan))
    L2 = np.zeros(len(tspan))
    # Compute two-site entanglement entropy
    two_site_S = []
    Sxx = []
    
    for n in range(len(tspan)):
        if abs((n * dt + 0.1) % 0.2 - 0.1) < 1.e-10:
          print("t = {t:0.02f}, chi =".format(t=n * dt), psi.get_chi())
                  
        
        M = b_model.TFIModel(L=L, J=J, Bi=Bi, Bf=Bf, time = tspan[n],tf=tmax*scale, bc='finite')
        U_bonds = calc_U_bonds(M.H_bonds, dt *scale, L)  # (imaginary dt -> realtime evolution)
        
        # Two-site operator sigma_x ⊗ sigma_x with legs (i, j, i*, j*)
        op_xx = np.reshape(np.kron(M.sx, M.sx), (M.d, M.d, M.d, M.d))

        run_TEBD(psi, U_bonds, 1, chi_max=100, eps=1.e-8)

        S.append(psi.entanglement_entropy())
        Sxx.append(psi.two_site_expectation_value(op_xx))  # list of length L-1


        Sz.append(psi.site_expectation_value(M.sz))
        Sx.append(psi.site_expectation_value(M.sx))
        TruncError[n]=abs(1.-np.prod(TruncE))
        
        # Measure two-site reduced density matrix entropy on the central bond
        rho_mid = psi.two_site_rdm(L//2)
        #print(rho_mid.shape)
        eigs = np.linalg.eigvalsh(rho_mid)
        S_mid = -np.sum(eigs * np.log(eigs + 1e-15))
        two_site_S.append(S_mid)
        
        # Sanity check
        rho = psi.two_site_rdm(1)
        #print(np.allclose(np.trace(rho), 1))  # should be True
        #print(np.allclose(rho, rho.conj().T))  # Hermitian
        #print(np.min(np.linalg.eigvalsh(rho)))  # >= 0

        PA[n] = np.abs(overlap(psiA, psi.Bs, L))**2
        PB[n] = np.abs(overlap(psiB, psi.Bs, L))**2
        Gamma[n] = -1/L * np.log(PA[n]+PB[n])
        
    
        L1[n] = min(-1/L * np.log(PA[n]),-1/L * np.log(PB[n]))
   
       
      
    results = [tspan, S, Sz, Sx, TruncError, Gamma] 
    
 
    #with open("TFIM-TEBDL500.pickle",'wb') as handle:
     #    pickle.dump(results, handle) 

    import matplotlib.pyplot as plt
    S = np.reshape(S,(len(tspan),L-1))
    Sx = np.reshape(Sx,(len(tspan),L))
    Sz = np.reshape(Sz,(len(tspan),L))

    Site =np.arange(1,L+1,1) 
    p = len(tspan)-1
    
    import matplotlib as mpl
    import matplotlib
    mpl.rcParams.update(mpl.rcParamsDefault)
    colors = (
    '#1090BF',  # blue
    '#DD5400',  # orange
    '#ECD01F',  # yellow
    '#8500D1',  # purple
    '#3BB02F',  # green
    '#2EBEF0',  # cyan
    '#D10C8B',   # magenta
    'k', #black
    )

    linestyles = ('-', '--', ':', '-.')
    markers = ('o', '+', '*', '.', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h')


    font = {'family' : 'Times New Roman',
            'serif': 'Times', 
            'weight' : 'regular',
            'style' : 'normal',
            
            'size'   : 16}

    matplotlib.rc('font', **font)
    mpl.rcParams['text.usetex'] = False
    plt.figure()
    plt.plot(tspan/scale,np.sum(Sz,axis=1)/L,color=colors[0],linewidth=1.0)
    plt.plot(tspan/scale,np.sum(Sx,axis=1)/L,color=colors[1],linewidth=1.0)
    plt.show()

    plt.figure()
    #plt.plot(tspan/scale,S[:,L//2 -1],'b',linewidth=1.0)
    plt.plot(tspan/scale,S,linewidth=1.0)
    
    
    plt.figure()
    plt.plot(tspan/scale,np.sum(Sxx,axis=1)/L,linewidth=1.0,color=colors[0])
    
    plt.figure()
    plt.semilogy(tspan/scale,TruncError,color=colors[0])
    
    plt.figure()
    plt.plot(tspan/scale, two_site_S,color=colors[0],linewidth=1.0)
    plt.show()
    
    plt.figure()
    #plt.plot(tspan/scale, PA,color=colors[0],linewidth=1.0)
    #plt.plot(tspan/scale, PB,color=colors[1],linewidth=1.0)
    plt.plot(tspan/scale, Gamma,color=colors[0],linewidth=1.0)
    plt.plot(tspan/scale, L1,color=colors[1],linewidth=1.0)
    plt.show()
    
   
    
if __name__ == "__main__":

  
    J = 1.
    Bi = J/0.42
    Bf = 10 * J
    L = 10
    
    TFIM_TEBD(L=L, J = J, Bi=Bi, Bf=Bf,  tmax=3.5, dt=0.005)
    
    

