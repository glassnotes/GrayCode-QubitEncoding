import sys
import os
import datetime
from pprint import pprint

import numpy as np
np.warnings.filterwarnings('ignore')
np.set_printoptions(precision=6, suppress=True)

from qiskit import execute, Aer
from qiskit.tools import parallel_map

# For various optimizers
from noisyopt import minimizeSPSA
from scipy.optimize import minimize
from iminuit import minimize as iminimize

# Load up our stuff
sys.path.append("./src/")
from hamiltonian import *
from utils import *
from qiskit_circuits import *
from device import Device


def set_parameters(parameters):
    """
    Set any parameters not included in input parameter file to default values.

    Parameters:
        parameters (dictionary) : Input parameters for VQE.  

    Returns:
        parameters : Updated parameters to include default parameters not given by input file
    """

    default_parameters = {
                      'encoding' : 'gray_code',
                      'N_trials' : 1, 
                      'backend' : 'statevector_simulator',
                      'N_shots' : 10000,
                      'device_name' : None,
                      'mitigate_meas_error' : False,
                      'optimizer' : 'SPSA',
                      'spsa_a' : 0.628,
                      'spsa_c' : 0.1,
                      'N_iter' : 1000,
                      'N_cpus' : 1,
                      'output_dir': 'outputs',
                      'layout' : None
                      }

    for param in parameters:
        if (param == 'encoding') and (parameters['encoding']!='gray_code' and parameters['encoding']!='jordan_wigner'):
            raise ValueError("Encoding {} not supported.  Please select 'gray_code' or 'jordan_wigner'. ".format(parameters['encoding']) )
        if (param == 'backend') and (parameters['backend']!='statevector_simulator' and parameters['backend']!='qasm_simulator'):
            raise ValueError("Backend {} not supported.  Please select 'statevector_simulator' or 'qasm_simulator'. ")
    
    # TO DELETE
    parameters=convert_depricated_parameters(parameters)
    
    # Set default values for anything not provided
    for parameter in default_parameters.keys():
        # parameter 'N_states' must be set in input file 
        if 'N_states' not in parameters.keys():
            raise ValueError("Must provide number of states for simulation. (N_states parameter)")
        if parameter not in parameters.keys():
            # Setting parameter to default parameter 
            parameters[parameter] = default_parameters[parameter]
            print(f"No value for parameter {parameter} provided.")
            print(f"Setting {parameter} to default value {default_parameters[parameter]}.")

    # iminuit only works with 1 processor; make sure this gets set
    if parameters['optimizer'] == 'MIGRAD':
        if parameters['backend'] != 'statevector_simulator':
            raise ValueError("MIGRAD optimizer is for use with statevector_simulator only.")
        if parameters['N_cpus'] > 1:
            print("Using multiple CPUs is not currently supported with optimizer 'MIGRAD'.")
            print("Setting N_cpus to default value 1.")
            parameters['N_cpus'] = 1

    # accept 'None' as a string in params.yml
    if parameters['device_name'] == 'none' or parameters['device_name'] == 'None':
        parameters['device_name'] = None
    if parameters['layout'] == 'none' or parameters['layout'] == 'None':
        parameters['layout'] = None

    # Check compatibility between device, layout and measurement error mitigation
    if parameters['device_name'] is None:
        if parameters['layout'] is not None:
            raise ValueError("Layout cannot be specified without a device.")
        if parameters['mitigate_meas_error'] is not False:
            raise ValueError("Measurement mitigation is not possible if no device is specified")

    # Layout must None or a list of ints
    if parameters['layout'] is not None:
        assert type(parameters['layout']) is list, "Layout must be a list of integers."
        assert all([type(q) is int for q in parameters['layout']]), "Layout must be a list of integers."

    print("\nExperiment parameters")
    pprint(parameters)
    print()

    return parameters


###############################################################################################
def compute_energy(theta, backend, hamiltonian, device=None, shots=10000):
    """
    Computes the expectation value of the hamiltonian by running a quantum circuit for 
    wavefunction parameterized by theta.     
    
    Parameters
        theta (np.array) : Initial guesses for the ground state wavefunction parameterization.
        
        backend (BaseBackend): Qiskit backend to execute circuits on.
        
        hamiltonian : Deuteron Hamiltonian. See src/hamiltonian.py for class definitions.
    
        device (Device, optional) : Simulations of IBMQ devices.  If device=None, qiskit QASM simulator
            applied without any noise model. Default: None.

        shots (int, optional): Number of repetitions of each circuit, for sampling. Default: 10000.
        
    """
    # Cumulative energy
    energy = 0
    # Store the expectation values of all the operators
    complete_exp_vals = {}

    for measurement_idx in hamiltonian.pauli_partitions.keys():
        # Create the circuit depending on the Hamiltonian and measurement
        circuit = None
        if type(hamiltonian) == SparseEncodingHamiltonian:
            circuit = sparse_variational_circuit(theta, measurement_idx, backend.name())
        elif type(hamiltonian) == DenseEncodingHamiltonian:
            if measurement_idx == hamiltonian.N_qubits:
                circuit = dense_variational_circuit(theta, None, backend.name())
            else:
                circuit = dense_variational_circuit(theta, measurement_idx, backend.name())

        # Execute the circuit (on device if specified) and get the result
        # For QASM simulator, run the circuit for as many shots as desired.
        if backend.name() == 'qasm_simulator':
            if device is not None:
                job = execute(circuit,
                        backend=backend,
                        shots = shots,
                        coupling_map = device.coupling_map,
                        noise_model = device.noise_model,
                        basis_gates = device.noise_model.basis_gates,
                        initial_layout = device.layout)
            else:
                job = execute(circuit, backend, shots=shots)
    
            result = job.result().get_counts(circuit)
    
            # Perform measurement error mitigation if specified 
            if device is not None:
                if device.meas_filter:
                    result = device.meas_filter.apply(result)

            # Get the expectation value of each Pauli from the results
            for pauli in hamiltonian.pauli_partitions[measurement_idx].keys():
                complete_exp_vals[pauli] = pauli_expectation_value(pauli, result)
        
        # For statevector simulator, compute the explicit wavefunction at the end
        # and calculate the expectation values using the trace.
        elif backend.name() == 'statevector_simulator':
            result = execute(circuit, backend).result()
            psi = result.get_statevector(circuit)
            rho = np.einsum("i,j->ij", psi, np.conj(psi))

            # Paulis in the Hamiltonian are already in Qiskit ordering, which is the
            # same ordering as the psi that will come out.
            for pauli in hamiltonian.pauli_partitions[measurement_idx].keys():
                pauli_mat = get_pauli_matrix(pauli)
                complete_exp_vals[pauli] = np.real(np.trace(np.dot(rho, pauli_mat)))

    # Now that we have the complete set of expectation values, we can compute the energy
    for pauli, coeff in hamiltonian.pauli_coeffs.items():
        energy += coeff * complete_exp_vals[pauli]

    return energy

###############################################################################################
def do_experiment(theta,hamiltonian,parameters):
    """
    Use variational quantum eigensovler (VQE) to obtain ground state energy of the Hamiltonian. 
    Energies at each step evaluated by running a quatnum circuit.

    Parameters
        theta (np.array) : Initial guesses for the ground state wavefunction theta parameters.
        
        hamiltonian : Deuteron Hamiltonian. See src/hamiltonian.py for class definitions.

        parameters (dictionary) : Input parameters for VQE.  

    Returns
        res (scipy.optimize.OptimizeResult) : Results of VQE

    """
    # Set up backend  
    backend = Aer.get_backend(parameters['backend'])
    device = None
    if parameters['device_name'] is not None:
        if parameters['mitigate_meas_error']:
            device = Device(parameters['device_name'], True, hamiltonian.N_qubits, layout=parameters['layout'])
        else:
            device = Device(parameters['device_name'], False, hamiltonian.N_qubits, layout=parameters['layout'])
    
    if device is not None:
        print("Device specifications")
        pprint(vars(device))


    # Run the minimization routine 
    if parameters['optimizer'] != 'SPSA':
        if parameters['optimizer'] == 'MIGRAD': # iminuit option
            res = iminimize(compute_energy, theta, args=(backend, hamiltonian, device), method=parameters['optimizer'], options={'maxfev':parameters['N_iter']})
            # Display diagnostic information:            
            # res = iminimize(compute_energy, theta, args=(backend, hamiltonian, device), method=parameters['optimizer'], options={'maxfev':parameters['N_iter'],'disp':True})
        else:
            # "Regular" scipy optimizers
            res = minimize(compute_energy, theta, args=(backend, hamiltonian, device), method=parameters['optimizer'], options={'maxiter':parameters['N_iter']})
    else:
        res = minimizeSPSA(compute_energy,
                x0=theta,
                args=(backend, hamiltonian, device, parameters['N_shots']),
                niter=parameters['N_iter'],
                paired=False,
                a=parameters['spsa_a'],
                c=parameters['spsa_c'],
        )
    
    return res

###############################################################################################
def analyze_results(results,N_theta):
    """
    Analyze results of the different trials for solving the deuteron Hamiltonain via VQE

    Parameters
        results (list of scipy.optimize.OptimizeResult) : Set of results obtained from trials

        N_theta (int) : Number of values needed to parameterize the deuteron wavefunction

    Returns
        energies (np.array) : Array of energies of each trial

        thetas (np.array -- N_trials x N_theta) : Array of parameters for wavefunction for each trial 
    """
    # Zero initialize energy array
    N_trials=len(results)

    # Initialize results containers 
    energies = np.zeros((N_trials, ))
    thetas=np.ndarray(shape=(N_trials,N_theta), dtype=float)
    success_rate=0

    # Extract energies and wavefunction parameters from results of minimizer 
    for i in range(0,N_trials):
        res=results[i]
        energies[i]=res.fun
        success_rate+=res.success
        thetas[i]=res.x
    
    print("{} sucesses out of {} trials -- {:.1f}%\n".format(success_rate, N_trials, success_rate/N_trials*100))

    return energies,thetas

###############################################################################################
def get_output_filename(parameters):
    """
    Generate output filename from input parameters 

    Parameters:
        parameters (dictionary) : Input parameters for VQE

    Returns:
        output_filename (string) : Basename for output files
    """
    if parameters['output_dir'] not in os.listdir():
        os.mkdir(parameters['output_dir'])

    # Generate output filename from input parameters     
    date=datetime.datetime.now().strftime("%Y-%m-%d")

    # If measure mitigation used, add tag to filename
   # additionally if initial device layout is specified
    mit=""
    layout_str=""
    dev_name = parameters['device_name']
    if dev_name is None:
        dev_name = 'no_device'
    else:
        if parameters['mitigate_meas_error']:
            mit= "-mit_meas"
    if parameters['layout'] is not None:
        layout_str += "_layout-" + '-'.join(str(e) for e in parameters['layout'])

    if parameters['backend'] == 'qasm_simulator':
        output_filename = "{output_dir}/{date}_{encoding}-{N_states}_states-{backend}-{N_shots}_shots-{optimizer}-{device}{layout_str}{mit}".format(date=date, device=dev_name, layout_str=layout_str, mit=mit, **parameters)
    else:
        output_filename = "{output_dir}/{date}_{encoding}-{N_states}_states-{backend}-{optimizer}-{device}{layout_str}{mit}".format(date=date, device=dev_name, layout_str=layout_str, mit=mit, **parameters)

    return output_filename


