import numpy as np
import quimb as qu
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
    # Define dimensions for the n×m grid
    dims = [[2] * m] * n  # n×m grid of spin-1/2 particles
    
    # Create the spin operator
    S = qu.spin_operator(component, sparse=True)
    
    # Place it at the specified site
    site = (i, j)  # Site coordinates in 2D grid
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