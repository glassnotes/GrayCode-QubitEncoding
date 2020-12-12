############################################################################################################
# Obtain energies for deuteron Hamiltonian defined in REF using a variational quantum eigensovler.
# This program takes a required yaml input file, e.g. parameters.yml, and outputs
#   <basename>-energies.npy : File containing deuteron ground state energy for each trial
#   <basename>-theta0.npy : File containing initial guess for deuteron wavefunction parameterization
#   <basename>-theta.npy : File containing wavefunction parameterization obtained using VQE.
#
#   <basename> generated based on input parameters.
#
#   If optional boolean save_to_textfile at start of main is set to True, text versions of the
#   above files will also be output.
#
# Parameters in input file:
#
#   Mandatory parameters:
#      N_states (int) -- number of deuteron basis states (N>=2).
#
#      Optional parameters:
#      encoding (string) -- specifies encoding of deuteron Hamiltonian.  See paper [XREF]
#          Supported encodings are:
#             'gray_code'  (default)
#             'one_hot'
#
#      N_trials (int) -- number of independent trials to run.  (Default value : 1).
#
#      backend (string) -- name of qiskit backend.  Supported backend are
#          'statevector_simulator'     (default)
#          'qasm_simulator'
#
#      N_shots (int) -- number of repetitions of each circuit.  Default value : 10000).
#
#      device_name (string) -- Handle for noise model used in qasm simulations based on IBMQ machines.
#          If handle is 'None' or no handle is given then no noise model is used.  Files containing data
#          used to create noise models are found in the directory "devices".
#          Valid handles are:
#              'None'              (default)
#              'ibmq_16_melbourne'
#              'ibmq_5_yorktown'
#              'ibmq_burlington'
#              'ibmq_essex'
#              'ibmq_london'
#              'ibmq_vigo'
#
#      mitigate_meas_error (boolean) : Apply readout error mitigation (Default value : False).
#
#      optimizer (string) -- Optimizer used for variation quantum eigensolver.
#          'SPSA' (default)
#          'Nelder-Mead'
#
#           For additional optimizers which can be used, see documentation for scipy.optimizer.minimize.
#           Supported functions include ‘Powell’, ‘CG’, ‘BFGS’, 'L-BFGS-B’, ‘TNC’, ‘COBYLA’, ‘SLSQP’, ‘trust-constr’
#           For iminuit optimizer use 'MIGRAD' option (this should be used only with statevector_simulator) and
#               N_cpus = 1
#
#           Additional parameters only used with SPSA optimizer
#              spsa_a : scaling parameter for step size.  (Default value : 0.628)
#              spsa_c : scaling parameter for evolution.  (Default value : 0.1)
#
#      N_iter : Number of iterations after which to terminate optimizer algorithm. (Default value : 1000)
#
#      N_cpus (int) -- Number of processes over which trials are distributed. (Default value : 1)
#
#      output_dir (string) -- Directory for results files.  (Default value : "outputs").
#                             Directory created at runtime if it doesn't exist.
############################################################################################################

import sys
import os
import yaml

import numpy as np
np.warnings.filterwarnings('ignore')
np.set_printoptions(precision=6, suppress=True)

# Load functions from src
sys.path.append("./src/")
from hamiltonian import *
from utils import *
from device import Device
from qiskit_circuits import *
from qiskit_experiment import *

if __name__ == "__main__":

    # Set to true if text output desired in addition to numpy outputfiles
    save_to_textfile=True

    # Get the input parameters filename
    if len(sys.argv) != 2:
        raise ValueError("Syntax: run_qiskit_experiment.py <input parameters>\n")

    parameter_file = sys.argv[1]
    parameters = {}

    # Read the parameters
    with open(parameter_file) as infile:
        try:
            parameters = yaml.safe_load(infile)
        except yaml.YAMLError as exc:
            print(exc)

    # If parameters not read in from file, set parameters to default.
    parameters=set_parameters(parameters)

    ###############################################################################################
    # Generate Hamiltonian representation in chosen basis (Gray code or one-hot)
    # and determine number of parameters for wavefunction for given encoding
    ham = None
    N_theta=0
    if parameters['encoding'] == 'gray_code':
        ham = DenseEncodingHamiltonian(N_states=parameters['N_states'])
        N_theta=2 ** ham.N_qubits - 1
    elif parameters['encoding'] == 'one_hot':
        ham = SparseEncodingHamiltonian(N_states=parameters['N_states'])
        N_theta=ham.N_qubits-1

    print("Hamiltonian Pauli rep")
    print(ham.pauli_rep,"\n")
    print("Pauli partitions")
    print(ham.pauli_partitions,"\n")

    ###############################################################################################
    # Zero initialize energy array
    results=[None]*parameters['N_trials']

    # Generate initial wavefuntion parameterization by thetas for each experiment
    initial_theta_list = []
    if( 'initial_thetas' in parameters ): initial_theta_list = list(parameters['initial_thetas'])
    if( len(initial_theta_list) == N_theta ):
        tmp = [ float(x) for i in range(parameters['N_trials']) for x in initial_theta_list ]
        initial_thetas = np.array( tmp ).reshape( (parameters['N_trials'], N_theta) )
    elif( len(initial_theta_list) == N_theta * parameters['N_trials'] ):
        initial_thetas = np.array( initial_theta_list ).reshape( (parameters['N_trials'], N_theta) )
    else:
        initial_thetas = np.random.uniform(low=-np.pi/2, high=np.pi/2,size=(parameters['N_trials'], N_theta))
    #print(initial_thetas)

    # Run experiment or experiment simulation for each set of wavefunction parameters to obtain energy
    # eigenvalues of Hamiltonian <ham>.  Runs are distributed over <N_cpus> processes using qiskit's
    # built-in parallel map function.
    results = parallel_map(do_experiment, initial_thetas, task_args=[ham,parameters], num_processes=parameters['N_cpus'])

    # Extracting energies and wavefunction parameterization from results
    output_energies,final_thetas=analyze_results(results,N_theta)

    ###############################################################################################
    # Write results to file
    output_filename = get_output_filename(parameters)
    print("output file basename:")
    print(output_filename,"\n")
    print(output_energies)

    # Save initial theta values for wavefunction parameterization for each trial,
    # final theta values, and eigenvalues of wavefunction for each trial
    np.save(output_filename + "-theta0.npy", initial_thetas)
    np.save(output_filename + "-theta.npy", final_thetas)
    np.save(output_filename + "-energies.npy", output_energies)

    if save_to_textfile==True:
        np.savetxt(output_filename + "-theta0.txt", initial_thetas,fmt='% 8.5f')
        np.savetxt(output_filename + "-theta.txt", final_thetas,fmt='% 8.5f')
        np.savetxt(output_filename + "-energies.txt", output_energies, fmt='% 8.5f')
