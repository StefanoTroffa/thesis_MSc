
import numpy as np, quimb as qu

def spin_at_square_site(i, j, n, m, component='z'):
    """
    Creates a spin operator for the specified component at site (i,j) in an n×m square lattice
    
    Parameters:
    -----------
    i, j : int
        Row and column indices for the site (0-based)
    n, m : int
        Dimensions of the lattice (n rows, m columns)
    component : str
        Spin component to measure: 'x', 'y', or 'z'
    
    Returns:
    --------
    operator
        The spin operator at the specified site
    """
    dims = [[2] * m] * n  # n×m grid of spin-1/2 particles
    
    S = qu.spin_operator(component, sparse=True)
    
    site = (i, j)  
    return qu.ikron(S, dims, inds=[site])
def calculate_site_magnetizations_square(state, n, m):
    """
    Calculates the z-component of magnetization at each site in  an n×m square lattice
    
    Parameters:
    -----------
    state : quantum state vector or density matrix
        The quantum state to measure
    n, m : int
        Dimensions of the lattice (n rows, m columns)
    
    Returns:
    --------
    numpy.ndarray, float
        2D array of site magnetizations and the total magnetization
    """
    magnetization = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            Sz = spin_at_square_site(i, j, n, m, 'z')
            magnetization[i, j] = qu.expec(Sz, state)
            
    total_mag = np.sum(magnetization)
    return magnetization, total_mag

def construct_total_Sz_square(n, m):
    """
    Constructs the total Sz operator directly for  an n×m square lattice
    
    Parameters:
    -----------
    n, m : int
        Dimensions of the lattice (n rows, m columns)
    
    Returns:
    --------
    operator
        The total Sz operator
    """
    dims = [[2] * m] * n  # n×m grid of spin-1/2 particles
    total_Sz = None
    
    for i in range(n):
        for j in range(m):
            site = (i, j)
            Sz_site = qu.ikron(qu.spin_operator('z', sparse=True), dims, inds=[site])
            
            if total_Sz is None:
                total_Sz = Sz_site
            else:
                total_Sz += Sz_site
    
    return total_Sz


def _site_op(op, i, N):
    """Return operator `op` acting on site i (0 … N-1)."""
    return qu.ikron(op, dims=[2]*N, inds=[i])

def build_site_ops(n, m):
    """Pre-build {Sx_i, Sy_i, Sz_i} for every site to avoid repetition."""
    N   = n*m
    Sxs = [_site_op(Sx, i, N) for i in range(N)]
    Sys = [_site_op(Sy, i, N) for i in range(N)]
    Szs = [_site_op(Sz, i, N) for i in range(N)]
    return Sxs, Sys, Szs

# ------------------------------------------------------------------
def spin_structure_factor(psi, n, m, kxs=None, kys=None):
    """
    Return S²(kx,ky) on the mesh (len(kxs), len(kys)).
    """
    N      = n*m
    Sxs, Sys, Szs = build_site_ops(n, m)          # cache single-site ops
    coords = np.array([(i//m, i % m) for i in range(N)])

    # two-site correlator C_{ij} = Σ_α <S_i^α S_j^α>
    C = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            C[i, j] = (
                qu.expec(Sxs[i] @ Sxs[j], psi)
              + qu.expec(Sys[i] @ Sys[j], psi)
              + qu.expec(Szs[i] @ Szs[j], psi)
            )

    # default k-grid covers the Brillouin zone
    if kxs is None:
        kxs = np.linspace(-np.pi, np.pi, n, endpoint=False)
    if kys is None:
        kys = np.linspace(-np.pi, np.pi, m, endpoint=False)

    S2 = np.empty((len(kxs), len(kys)), dtype=float)
    for ix, kx in enumerate(kxs):
        for iy, ky in enumerate(kys):
            phase = np.exp(1j * (kx*(coords[:,0,None]-coords[:,0])
                               + ky*(coords[:,1,None]-coords[:,1])))
            S2[ix, iy] = (C * phase).sum().real / N**2
    return kxs, kys, S2



def staggered_magnetization_quimb(n,m, gs_quimb):
    """
    Compute the staggered magnetization for a 2D Heisenberg model using Quimb.
    This only works for even numbered square lattices.
    """
    N_sites = n*m
    eps = np.array([1 if (x+y)%2==0 else -1
                    for x in range(n) for y in range(m)])  
    basis = ((np.arange(2**(N_sites))[:,None] >> np.arange(N_sites)) & 1)*2 - 1
    Ms_vals = (basis * eps[None,:]).sum(axis=1)    

    probs = np.abs(gs_quimb)**2      
    Ms= np.dot(probs.flatten(), Ms_vals)     
    Ms2= np.dot(probs.flatten(), Ms_vals**2)   
    m_rms_quimb = np.sqrt(Ms2) / N_sites
    m_abs_quimb = np.dot(probs.flatten(), np.abs(Ms_vals)) /(N_sites)
    Spp_quimb   = Ms2 / (N_sites)
    print("Quimb baseline:", m_rms_quimb, m_abs_quimb, Spp_quimb)
    return m_rms_quimb, m_abs_quimb, Spp_quimb

