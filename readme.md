# Thesis Work of Stefano Troffa for the MSc in Theoretical Physics at Leiden University

This project aims to replicate the results presented in the paper [Learning ground states of quantum Hamiltonians with graph networks
](https://arxiv.org/pdf/2110.06390) by [Dmitrii Kochkov et al], focusing on finding approximate ground states of Heisenberg Hamiltonians with a Graph Neural Network as a Neural Quantum state using imaginary time evolution as an update rule for the target function.

## Code Structure

The code is organized into several key components:

- **compgraph**: Contains folders with helper functions and a Jupyter notebook that demonstrates the basic implementation of the project's core idea.

- **tests**: Contains folder with test functions. Some test functions are simply sanity checks on the consistency of data handling, for the bulk of the code, every quantity that we computed has been tested against Quimb library

- **notebooks**: This comprises notebooks to clarify the workflow and show the main milestones achieved during the project, along with an example usage of the project functions. 


## Project Description

The purpose of this codebase is to compute approximate ground states using a computational graph. This method serves as a proxy for the ansatz that generates the wave function weights, an innovative approach first introduced by Carleo in 2016. For more information, see the paper ["Solving the Quantum Many-Body Problem with Artificial Neural Networks"](https://arxiv.org/abs/1606.02318).

## Specific Neural Network Model

The neural network model explored in this project is built upon Sonnet modules, designed to recreate a pipeline similar to that described in the main reference paper. Detailed implementation and methodology can be found within the codebase.