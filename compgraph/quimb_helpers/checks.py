import quimb as qu
import numpy as np
def create_uniform_state(n, m):
    """
    Creates a uniform superposition of all computational basis states
    for an n×m lattice with spin-1/2 particles
    
    Parameters:
    -----------
    n, m : int
        Dimensions of the lattice
    
    Returns:
    --------
    A quantum state representing equal superposition of all basis states
    """
    num_sites = n * m
    hilbert_dim = 2**num_sites
    
    # Create state with equal amplitude for all basis states
    uniform_amplitude = 1 / np.sqrt(hilbert_dim)
    state_vector = np.ones(hilbert_dim) * uniform_amplitude
    
    return qu.qu(state_vector)
def verify_constant_row_sum(H, uniform_state, n, m):
    """Verify that the Heisenberg Hamiltonian has constant row sums"""    
    # Convert to dense matrix to examine structure 
    # (only practical for small systems)
    if n*m <= 10:  # Limit to avoid memory issues
        H_dense = np.array(H)
        
        # Check row sums and column sums
        row_sums = np.sum(H_dense, axis=1)
        
        print(f"Row sums of Hamiltonian matrix:")
        print(f"Min: {np.min(row_sums):.8f}")
        print(f"Max: {np.max(row_sums):.8f}")
        print(f"Mean: {np.mean(row_sums):.8f}")
        print(f"Standard deviation: {np.std(row_sums):.8f}")
        
        # If all row sums are the same (within numerical precision)
        if np.std(row_sums) < 1e-10:
            print("ALL ROW SUMS ARE EQUAL - This explains why the uniform state remains uniform!")
        else:
            print("Row sums vary - The uniform state should change.")
    
    # Test by directly applying to uniform state
    H_psi = H @ uniform_state
    

    print("\nAfter applying H to uniform state:")
    print(f"Min amplitude magnitude: {np.min(np.abs(H_psi)):.10f}")
    print(f"Max amplitude magnitude: {np.max(np.abs(H_psi)):.10f}")
    print(f"Standard deviation: {np.std(np.abs(H_psi)):.10f}")
    
    return H_psi

def analyze_computational_basis_decomposition(gs, n, m,S_z_tot):
    """
    Analyze how the ground state decomposes into computational basis states
    and verify their magnetization properties
    """
    num_sites = n * m
    total_dim=2**num_sites
    gs_vec = gs.flatten()
    overlaps = np.abs(gs_vec)**2  # Probability amplitudes
    # Threshold for significant overlap
    threshold = 4 / total_dim**2
    # Track how much of the ground state is covered by significant states
    total_significant_prob = 0
    # For each computational basis state with significant overlap
    for idx in range(total_dim):
        if overlaps[idx] > threshold:
            # Get the bit string representation
            bit_string = format(idx, f'0{num_sites}b')
            
            # Create the computational basis state
            comp_state = qu.computational_state(bit_string)
            
            # Calculate Sz expectation using operator
            sz_operator = qu.expec(S_z_tot, comp_state)
            
            # Calculate Sz analytically
            num_ones = bit_string.count('1')
            sz_formula = -(num_ones - (num_sites - num_ones)) / 2  # (up - down)/2
            
            # Check if they match
            match = np.isclose(sz_operator, sz_formula, atol=1e-10)
            
            print(f"{idx:<8} {bit_string:<{num_sites+2}} {overlaps[idx]:<10.6f} {sz_operator:<15.6f} {sz_formula:<15.6f} {'✓' if match else '✗'}")
            
            total_significant_prob += overlaps[idx]

    print(f"\nTotal probability covered by significant basis states: {total_significant_prob:.6f}")
    # Group by Sz values to see distribution
    sz_groups = {}
    for idx in range(total_dim):
        if overlaps[idx] > 1e-10:  # Any non-zero contribution
            bit_string = format(idx, f'0{num_sites}b')
            num_ones = bit_string.count('1')
            sz_val = -(num_ones - (num_sites - num_ones)) / 2
            
            sz_groups.setdefault(sz_val, 0)
            sz_groups[sz_val] += overlaps[idx]

    print("\nDistribution of ground state by Sz values:")
    for sz_val in sorted(sz_groups.keys()):
        print(f"Sz = {sz_val:+.1f}: {sz_groups[sz_val]:.6f}")

    # Calculate the total magnetization of the ground state
    sz_gs = qu.expec(S_z_tot, gs)
    print(f"\nTotal Sz of ground state: {sz_gs:.6f}")
