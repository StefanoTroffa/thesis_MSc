# Thesis Work of Stefano Troffa for the MSc in Theoretical Physics at Leiden University

This project aims to replicate the results presented in the paper [Learning ground states of quantum Hamiltonians with graph networks](https://arxiv.org/pdf/2110.06390) by [Dmitrii Kochkov et al], focusing on finding approximate ground states of Heisenberg Hamiltonians with a Graph Neural Network as a Neural Quantum state (NQS) using imaginary time evolution as an update rule for the target function.

## Setup:
To set up the environment, follow these steps:
### Create and activate conda environment from the provided yml file
conda env create -f environment_last.yml
conda activate quantum-ham-gnn

### If you prefer to use pip: 
Please find the relevant libraries under pip in environment_last.yml

### GPU setup: 
This was tested in Alice cluster from Leiden, as well as on my personal laptop. With cuda version $>=12.3$ it should work, tested on cuda $12.5$ and cuda $12.3$.

## Code Structure
The code is organized into several components:

- **compgraph**: Contains folders with helper functions and a Jupyter notebook that demonstrates the basic implementation of the project's core idea.
    - models.py -> This file contains the different models built on Sonnet.
    - monte_carlo.py -> This file contains the monte carlo sampler class. There are different implementation within the same class, plus numerous additional functions that were used in a previous stage of the codebase. Please refer mostly to the class and "compute_phi_terms" functions as those are used in the final simulation.
    - tensorflow_version directory: This contains numerous files allowing to log state information, cleanup the memory, load the saved logs with suitable parameters depending on the folder structure (which contains the simulation parameters).
    - quimb_helpers directory: This directory contains the minimal number of functions that were added on top of quimb functionalities to measure staggered magnetisation.


- **tests**: Contains folder with test functions. Some test functions are simply sanity checks on the consistency of data handling, for the bulk of the code, every quantity that we computed has been tested against Quimb library

- **notebooks**: This comprises notebooks to clarify the workflow and show the main milestones achieved during the project, along with an example usage of the project functions, the most updated version is 'FinalThesisgraphs.ipynb", the other notebook are preserved only as a testimony of previous efforts and of the path that lead to the ultimate version.

- **simulation**: This directory contains python files to run simulations -> tf_simulation_checkpoints, which runs a variatonal monte carlo (VMC) using a NQS. The data produced is saved in a directory which preserves 5 checkpoints, the number of checkpoints is tunable as well as the number of steps after which the checkpoint should be preserved. 


## Project Description

The purpose of this codebase is to compute approximate ground states using a computational graph. This method serves as a proxy for the ansatz that generates the wave function weights, an approach first introduced by Carleo in 2016. For more information, see the paper ["Solving the Quantum Many-Body Problem with Artificial Neural Networks"](https://arxiv.org/abs/1606.02318).

## Specific Neural Network Model

The neural network model explored in this project is built upon Sonnet modules, designed to recreate a pipeline similar to that described in the main reference paper. Detailed implementation and methodology can be found within the codebase.

## Final Disclaimer and work in progress
The repository and the functionalities are being refactored in a new polished version to further expand the capabilities of the codebase, and enhance its capacities. This project remains as it this to testify the actual work and the growth.
