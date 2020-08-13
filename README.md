# Improving Hamiltonian encodings with the Gray code

Obtain energies for deuteron Hamiltonian defined in [https://arxiv.org/abs/2008.05012](https://arxiv.org/abs/2008.05012) using a variational quantum eigensolver. Code was written jointly by Olivia Di Matteo ([@glassnotes](https://github.com/glassnotes/)), Anna McCoy ([@aemccoy](https://github.com/aemccoy/)), Peter Gysbers ([@pgysbers](https://github.com/pgysbers)) and Takayuki Miyagi ([@Takayuki-Miyagi](https://github.com/Takayuki-Miyagi)). 

## Installation

We have provided the specifications of an Anaconda environment in the file `environment.yml`. To reproduce the environment, install Anaconda and run
```
conda env create --file environment.yml
```

The results of the paper are obtained with qiskit version 0.19.2, this specified in the environment file.
To double-check, run `python`: 
```
Python 3.7.5 (default, Oct 25 2019, 15:51:11) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import qiskit
>>> qiskit.__version__
'0.14.1'
>>> qiskit.__qiskit_version__
{'qiskit-terra': '0.14.1', 'qiskit-aer': '0.5.1', 'qiskit-ignis': '0.3.0', 'qiskit-ibmq-provider': '0.7.1', 'qiskit-aqua': '0.7.1', 'qiskit': '0.19.2'}
```

## Instructions

To compute energies run the program in this director called `run_qiskit_experiments.py`.  This program takes a yaml input file which specifies the necessary parameters for the calculation.  An example of the input file is given by `params.yml`.  Set the parameters in the yaml file, and then run
```
python run_qiskit_experiments.py params.yml
```
By default, this will create a directory called `outputs` to store the output, unless you set `output_dir` in `params.yml`. If the directory doesn't exist, the directory will be created.

The program outputs three .npy files: 
```
<basename>-energies.npy : Ground state energy for each trial
<basename>-theta0.npy : Initial guess for deuteron wavefunction parameterization
<basename>-theta.npy : Wavefunction parameterization obtained using VQE. 
```
The `<basename>` is generated internally from input parameters.  

If optional boolean `save_to_textfile` at start of main is set to `True`, text versions of the above files will also be output. 

## Input file
```
    Mandatory parameters:
        N_states (int) -- number of deuteron basis states (N>=2).  
     
        Optional parameters:
        encoding (string) -- specifies encoding of deuteron Hamiltonian.  See paper [XREF]
            Supported encodings are:
               'gray_code'  (default)
               'jordan_wigner' 
     
        N_trials (int) -- number of independent trials to run.  (Default value : 1). 
     
        backend (string) -- name of qiskit backend.  Supported backend are 
            'statevector_simulator'     (default)
            'qasm_simulator'
     
        N_shots (int) -- number of repetitions of each circuit.  Default value : 10000). 
     
        device_name (string) -- Handle for noise model used in qasm simulations based on IBMQ machines.  
            If handle is 'None' or no handle is given then no noise model is used.  Files containing data 
            used to create noise models are found in the directory "devices".  
            Valid handles are:
                'None'              (default)
                'ibmq_16_melbourne'
                'ibmq_5_yorktown'
                'ibmq_burlington'
                'ibmq_essex'
                'ibmq_london'
                'ibmq_vigo'

        layout (list) -- Specification of which physical qubits on a device to use and the initial qubit ordering (the ordering may be changed by SWAPs during circuit execution, this corresponds to using the "initial_layout" argument for qiskit.execute()). The length of the list must correspond to 'N_qubits' (i.e. N_states with jordan_wigner encoding, ceil(log_2(N_states)) with gray_code)
            None (default)
            Examples with 3 qubits on a 5 qubit device:
            [0, 1, 2]
            [4, 3, 2]
     
        mitigate_meas_error (boolean) -- Apply readout error mitigation (Default value : False). 
     
        optimizer (string) -- Optimizer used for variation quantum eigensolver.
            'SPSA' (default)
            'Nelder-Mead' 
     
             For additional optimizers which can be used, see documentation for scipy.optimizer.minimize.
             Supported functions include ‘Powell’, ‘CG’, ‘BFGS’, 'L-BFGS-B’, ‘TNC’, ‘COBYLA’, ‘SLSQP’, ‘trust-constr’
             For iminuit optimizer use 'MIGRAD' option (this should be used only with statevector_simulator) and
                 N_cpus = 1
     
             Additional parameters only used with SPSA optimizer         
                spsa_a : scaling parameter for step size.  (Default value : 0.628)
                spsa_c : scaling parameter for evolution.  (Default value : 0.1)
     
        N_iter : Number of iterations after which to terminate optimizer algorithm. (Default value : 1000)
     
        N_cpus (int) -- Number of processes over which trials are distributed. (Default value : 1)
     
        output_dir (string) -- Directory for results files.  (Default value : "outputs"). 
                               Directory created at runtime if it doesn't exist.
```
