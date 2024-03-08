import numpy as np
def apply_raising_operator(config, site):
    """
    We are assuming in the following that the configuration is given as an np.array
    Further the following structure is given, the configuration is a basis state representation, corresponding
    to site i we have the expectation value of S_z(i), with obvious notation (expectation valeu of the spin
    along the z direction for site i)

    """
    new_config = config.copy()
    if config[site] == -1:
        new_config[site] = 1  # Spin flip from down to up
        return new_config
    else:
        return None  # State is annihilated

def apply_lowering_operator(config, site):
    """
    We are assuming in the following that the configuration is given as an np.array
    Further the following structure is given, the configuration is a basis state representation, corresponding
    to site i we have the expectation value of S_z(i), with obvious notation (expectation valeu of the spin
    along the z direction for site i)

    """
    new_config = config.copy()
    if config[site] == 1:
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