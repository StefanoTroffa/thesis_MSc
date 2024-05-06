# Thesis Work of Stefano Troffa for the MSc in Theoretical Physics at Leiden University

This project aims to replicate the results presented in the paper [Title of Paper](https://arxiv.org/pdf/2110.06390) by [Authors], focusing on [brief description of the main objective or findings of the paper].

## Code Structure

The code is organized into few distinct components:

- **compgraph**: Contains folders with helper functions and a Jupyter notebook that demonstrates the basic implementation of the project's core idea.
- **tests**: Contains folder with test functions. Some test functions are simply sanity checks on the consistency of data handling, for the bulk of the code, every quantity that we computed has been tested against Quimb library

- **notebooks**: This comprises notebooks to clarify the workflow and show the main milestones achieved during the project, along with an example usage of the project functions. 



## Project Description

The purpose of this codebase is to compute approximate ground states using a computational graph. This method serves as a proxy for the ansatz that generates the wave function weights, an innovative approach first introduced by Carleo in 2016. For more information, see the paper ["Title of Carleo's Paper"](https://arxiv.org/abs/1606.02318).

## Specific Neural Network Model

The neural network model explored in this project is built upon Sonnet modules, designed to recreate a pipeline similar to that described in the main reference paper. Detailed implementation and methodology can be found within the codebase.
