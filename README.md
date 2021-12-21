# P5 - An Agents Guide to the Galaxy

Project made by group d506, AAU Computer Science - 5th semester.

Authors: Magnus S. Andersen, Nikolai A. Bonderup, Jakob B. Hyldgaard, Christian B. Larsen, Sebastian Lassen & Simon B. Olsen.

Hand in date: 22/12-2021

## Repository structure

The repository is divided into the folders: `rebound_test`, `nn_trainer`, `testing_data_handling`, and `gekko_data_gen`.
Each of these folders contain a main and some util either as a subfolder or as a file.

- [rebound_test](./rebound_test)
  - Contains source code for running a simulation in the framework Rebound 
- [nn_trainer](./nn_trainer)
  - Contains source code for setup and training of neural networks using the framework Tensorflow.  
- [testing_data_handling](./testing_data_handling)
  - Contain source code for running programs related to handling and analysing data in this project.
  
    Note however that there is no data supplied as part of this repository, though it can be generated from the environment in [rebound_test](./rebound_test).

## General requirements 

This project is written in python 3 (either 3.8.10 or 3.9) and uses a number of different packages and frameworks all of which can be found in [requirements.txt](./requirements.txt). 
These can be installed globally or inside a venv. 
Installing and setting up the code in a venv, which is most likely the easiest and most convenient approach, is described below in [venv section](#venv) below. 

## Configuration
It is possible to give the program an invalid configuration which will make the program fail. If the program crashes, first check the project configuration. Each of the files containing a main function has to be called within the directory in which they are defined. 

## Data
It is required to have training data before running the program, othervise it will not work. 

## [Venv](https://docs.python.org/3/library/venv.html)

Can be used to conveniently run the program with all project dependencies.

### Create a venv

Assuming current working directory is the root of this repository:

    python3 -m venv <venv>

`<venv>` could just be `./venv` which will cause the above command to create a venv folder named `venv` in cwd.

### Activate the venv

|     Platform      |      Shell         |      Command to activate virtual environment       |
|:------------------|:-------------------|:---------------------------------------------------|
|     POSIX         |      bash/zsh      |      $ source <venv>/bin/activate                  |
|                   |      fish          |      $ source <venv>/bin/activate.fish             |
|                   |      csh           |      $ source <venv>/bin/activate.csh              |
|     Windows       |      cmd.exe       |      C:\> <venv>\Scripts\activate.bat              |
|                   |      PowerShell    |      PS C:\> <venv>\Scripts\Activate.ps1           |


After having activated the venv, you now have access to pip and can install packages into the venv.

### Installing requirements

You may need to install [wheel](https://pypi.org/project/wheel/) before proceeding to install the requirements.

    pip install wheel

Install dependencies:

    pip install -r requirements.txt


Need more information about virtual environments? Take a loook at [venv — Creation of virtual environments](https://docs.python.org/3/library/venv.html).
