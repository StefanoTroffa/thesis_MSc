
def explore_model(model):
    """
    Explore the variable structure of a Sonnet model.
    
    Args:
        model: A Sonnet model
    """

    
    print(f"Total variables in model: {len(model.variables)}")
    
    # Group variables by module
    modules = {}
    for var in model.variables:
        name = var.name
        parts = name.split('/')
        if len(parts) >= 2:
            module = parts[0]
            if module not in modules:
                modules[module] = []
            modules[module].append(var)
    
    # Print module hierarchy
    print("\nModule hierarchy:")
    for module, vars in modules.items():
        print(f"  {module} ({len(vars)} variables)")
        
        # Group by submodule
        submodules = {}
        for var in vars:
            name = var.name
            parts = name.split('/')
            if len(parts) >= 3:
                submodule = parts[1]
                if submodule not in submodules:
                    submodules[submodule] = []
                submodules[submodule].append(var)
        
        # Print submodules
        for submodule, subvars in submodules.items():
            print(f"    {submodule} ({len(subvars)} variables)")
            
            # Print a few example variables
            for i, var in enumerate(subvars):
                print(f"      {var.name}: {var.shape}")
          
