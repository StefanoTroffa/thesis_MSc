from compgraph.useful import nd_to_index
import numpy as np

def apply_raising_operator(config, site):
    """
    We are assuming in the following that the configuration is given as an np.array
    Further the following structure is given, the configuration is a basis state representation, corresponding
    to site i we have the expectation value of S_z(i), with obvious notation (expectation valeu of the spin
    along the z direction for site i)

    """
    new_config = config.copy()
    if new_config[site] == -1:
        new_config[site] = 1  # Spin flip from down to up
        return new_config
    else:
        return None  # State is annihilated

def apply_lowering_operator(config, site):
    """
    We are assuming in the following that the configuration is given as an np.array
    Further the following structure is given, the configuration is a basis state representation, corresponding
    to site i we have the expectation value of S_z(i), with obvious notation (expectation value of the spin
    along the z direction for site i)

    """
    new_config = config.copy()
    if new_config[site] == 1:
        new_config[site] = -1  # Spin flip from up to down
        return new_config
    else:
        return None  # State is annihilated
def are_configs_identical(config1, config2):
    """
    We are assuming in the following that the configuration is given as an np.array
    Further the following structure is given, the configuration is a basis state representation, corresponding
    to site i we have the expectation value of S_z(i), with obvious notation (expectation valeu of the spin
    along the z direction for site i)

    """
    # Use numpy array equality for vectorized comparison
    return np.array_equal(config1, config2)

def configs_differ_by_two_sites(config1, config2):
    """
    We are assuming in the following that the configuration is given as an np.array of the form [ 1  1 ... -1 -1]
    Further the following structure is given, the configuration is a basis state representation, corresponding
    to site i we have the expectation value of S_z(i), with obvious notation (expectation valeu of the spin
    along the z direction for site i)

    """
    # Use numpy to count the differences in a vectorized manner
    return np.sum(config1 != config2) == 2
def apply_edge_contribution(config, i, j):
    # Apply the spin raising operator on site i and lowering on site j
    new_config_raised = apply_raising_operator(config, i)
    if new_config_raised is not None:
        new_config_lowered = apply_lowering_operator(new_config_raised, j)
    else:
        new_config_lowered = None

    # Apply the spin lowering operator on site i and raising on site j
    new_config_lowered_initial = apply_lowering_operator(config, i)
    if new_config_lowered_initial is not None:
        new_config_raised = apply_raising_operator(new_config_lowered_initial, j)
    else:
        new_config_raised = None

    # Combine the new configurations if they are not None
    new_configs = []
    if new_config_raised is not None:
        new_configs.append(new_config_raised)
    if new_config_lowered is not None:
        new_configs.append(new_config_lowered)

    return new_configs    



def Square_2DHam_exp(psi, graph, phi, J2, configs_psi, configs_phi):
    expectation_value = 0
    node_to_index=nd_to_index(graph)
    ###NEED TO FIX THE POSSIBILITY OF NOT allowing DIFF DIMENSION BETWEEN GRAPH AND CONFIGURATIONS GIVEN
    # Define the coupling constants, we only need J2 as J1 can be set WLOG to 1 
    J2 = 2. # 
    J1=1. #Is fixed, we can vary only j2 because it is equivalent up to a constant factor -> Does not influence the eigenvectors, only the eigenvalues 
    for (k,config_phi) in enumerate(configs_phi):
        for (l,config_psi) in enumerate(configs_psi):
            # print(l,k, "Config", config_psi, config_phi)
            # print(psi, phi, "those are the states ")
            if are_configs_identical(config_phi,config_psi):
                expectation_value += (J1+J2)*psi[l].conj()*phi[k]
            elif configs_differ_by_two_sites(config_phi,config_psi):
                for i, j in graph.edges:
                    # Map nodes to indices
                    i_index = node_to_index[i]
                    j_index = node_to_index[j]
                    # print(config_psi, i_index, j_index)
                    off_diag=apply_edge_contribution(config_psi,i_index, j_index)
                    if off_diag is not None:
                        # print(off_diag, config_phi, "HERE WE ARE")
                        for new_config in off_diag:
                            # Check if the new configuration is identical to config_phi
                            if are_configs_identical(new_config, config_phi):
                                expectation_value += 0.5*J1*psi[l].conj()*phi[k]
                            # to identify these pairs based on the geometry of your lattice.
                for i in graph.nodes:
                    i_index = node_to_index[i]

                    for j in graph.neighbors(i):
                        j_index = node_to_index[j]
                        off_diag=apply_edge_contribution(config_psi,i_index, j_index)
                        if off_diag is not None:
                                    # print(off_diag, config_phi, "HERE WE ARE")
                                    for new_config in off_diag:
                                        # Check if the new configuration is identical to config_phi
                                        if are_configs_identical(new_config, config_phi):
                                            expectation_value += 0.5*J2*psi[l].conj()*phi[k]

    print("end of square 2d function")
    return expectation_value
                        