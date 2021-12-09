# P5 - An Agents Guide to the Galaxy

Project made by group d506, AAU Computer Science - 5th semester.

Authors: Magnus S. Andersen, Nikolai A. Bonderup, Jakob B. Hyldgaard, Christian B. Larsen, Sebastian Lassen & Simon Olsen.

Hand in date: 22/12-2021

## Repository structure

The repository is divided into the folders: `rebound_test`, `nn_trainer`, `testing_data_handling`, and `gekko_data_gen`.
Each of these folders contain a main and some util either as a subfolder or as a file.

- [rebound_test](./rebound_test)
  - Contains source code for running a simulation setup in the framework Rebound 
- [nn_trainer](./nn_trainer)
  - Contains source code for setup and training of neural networks using the framework Tensorflow.  
- [testing_data_handling](./testing_data_handling)
  - Contain source code for running programs related to handling and analysing data in this project.
  
    Note however that there is no data supplied as part of this repository, though it can be generated from the environment in [rebound_test](./rebound_test).

## Requirements

This project is written in python [version_num] and uses a number of different packages and frameworks. The required frameworks and the version used in development of this repository can be found in [requirements.txt](./requirements.txt).

The requirements can all be install using this command:

    pip install -r requirements.txt

## Venv

Can be used to run the program with all project dependencies.

### Guide

Create a venv

    python3 -m venv ./venv

Activate the venv

    source venv/bin/activate

