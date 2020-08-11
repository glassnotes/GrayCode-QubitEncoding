import sys
import os
import datetime
from pprint import pprint

import numpy as np
np.warnings.filterwarnings('ignore')
np.set_printoptions(precision=6, suppress=True)

from qiskit import execute, Aer
import qiskit.providers.aer.noise as noise
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
def compute_energy(theta, backend, hamiltonian, device=None, noise_model=None, shots=10000, num_cnot_pairs=0, num_folding=0, zero_noise_extrapolation=False):
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
    if( zero_noise_extrapolation and backend.name() == 'qasm_simulator' ):
        return compute_energy_extrap( theta, backend, hamiltonian, device, noise_model, shots, num_cnot_pairs, num_folding )
    # Cumulative energy
    energy = 0
    # Store the expectation values of all the operators
    complete_exp_vals = {}

    for measurement_idx in hamiltonian.pauli_partitions.keys():
        # Create the circuit depending on the Hamiltonian and measurement
        circuit = None
        if type(hamiltonian) == SparseEncodingHamiltonian:
            circuit = sparse_variational_circuit(theta, measurement_idx, backend.name(), num_cnot_pairs, num_folding)
        elif type(hamiltonian) == DenseEncodingHamiltonian:
            if measurement_idx == hamiltonian.N_qubits:
                circuit = dense_variational_circuit(theta, None, backend.name(), num_cnot_pairs, num_folding)
            else:
                circuit = dense_variational_circuit(theta, measurement_idx, backend.name(), num_cnot_pairs, num_folding)

        # Execute the circuit (on device if specified) and get the result
        # For QASM simulator, run the circuit for as many shots as desired.
        if backend.name() == 'qasm_simulator':
            if( noise_model != None ):
                job = execute(circuit,
                        backend=backend,
                        shots = shots,
                        noise_model = noise_model,
                        basis_gates = noise_model.basis_gates,
                        optimization_level=0)
            elif device is not None:
                job = execute(circuit,
                        backend=backend,
                        shots = shots,
                        coupling_map = device.coupling_map,
                        noise_model = device.noise_model,
                        basis_gates = device.noise_model.basis_gates,
                        initial_layout = device.layout,
                        optimization_level=0)
            else:
                job = execute(circuit, backend, shots=shots, optimization_level=0)

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
            result = execute(circuit, backend, shots=1).result()
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
def compute_energy_extrap(theta, backend, hamiltonian, device, noise_model, shots, num_cnot_pairs, num_folding):
    energies = []
    noise_parameters = []
    CNOTs = False
    Folding = False
    if( num_cnot_pairs > 0 ): CNOTs=True
    if( num_folding > 0 ): Folding=True
    if( CNOTs ): iter_max = num_cnot_pairs + 1
    if( Folding ): iter_max = num_folding + 1
    energies = []
    noise_parameters = []
    for num in range(iter_max+1):
        complete_exp_vals = {}
        for measurement_idx in hamiltonian.pauli_partitions.keys():
            # Create the circuit depending on the Hamiltonian and measurement
            circuit = None
            if type(hamiltonian) == SparseEncodingHamiltonian:
                if(CNOTs): circuit = sparse_variational_circuit(theta, measurement_idx, backend.name(), num, 0)
                if(Folding): circuit = sparse_variational_circuit(theta, measurement_idx, backend.name(), 0, num)
            elif type(hamiltonian) == DenseEncodingHamiltonian:
                if measurement_idx == hamiltonian.N_qubits:
                    if(CNOTs): circuit = dense_variational_circuit(theta, None, backend.name(), num, 0)
                    if(Folding): circuit = dense_variational_circuit(theta, None, backend.name(), 0, num)
                else:
                    if(CNOTs): circuit = dense_variational_circuit(theta, measurement_idx, backend.name(), num, 0)
                    if(Folding): circuit = dense_variational_circuit(theta, measurement_idx, backend.name(), 0, num)

            # Execute the circuit (on device if specified) and get the result
            # For QASM simulator, run the circuit for as many shots as desired.
            if( noise_model != None ):
                job = execute(circuit,
                        backend=backend,
                        shots = shots,
                        noise_model = noise_model,
                        basis_gates = noise_model.basis_gates,
                        optimization_level=0)
            elif device != None:
                job = execute(circuit,
                        backend=backend,
                        shots = shots,
                        coupling_map = device.coupling_map,
                        noise_model = device.noise_model,
                        basis_gates = device.noise_model.basis_gates,
                        initial_layout = device.layout,
                        optimization_level=0)
            else:
                job = execute(circuit, backend, shots=shots, optimization_level=0)

            result = job.result().get_counts(circuit)

            # Perform measurement error mitigation if specified
            if device != None:
                if device.meas_filter:
                    result = device.meas_filter.apply(result)

            # Get the expectation value of each Pauli from the results
            for pauli in hamiltonian.pauli_partitions[measurement_idx].keys():
                complete_exp_vals[pauli] = pauli_expectation_value(pauli, result)
        en = 0.0
        for pauli, coeff in hamiltonian.pauli_coeffs.items():
            en += coeff * complete_exp_vals[pauli]
        energies.append(en)
        noise_parameters.append( 2*num+1 )
    coef = np.polyfit(noise_parameters,energies,1)
    linear_fit = np.poly1d(coef)
    #print( "zero-noise extrapolation: n=0:{:8.4f}, n=1:{:8.4f}, n=3:{:8.4f}, n=5:{:8.4f}".format(linear_fit(0), linear_fit(1), linear_fit(3), linear_fit(5)) )
    return linear_fit(0)

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
    noise_model = None
    if parameters['device_name'] is not None:
        if ( parameters['device_name'][0:6] == "custom" ):
            noise_model = noise.NoiseModel()
            errors = parameters['device_name'].split("_")
            error1 = float(errors[1])
            error2 = float(errors[2])
            error_1 = noise.errors.depolarizing_error(error1, 1)
            error_2 = noise.errors.depolarizing_error(error2, 2)
            noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
            noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
        else:
            if parameters['mitigate_meas_error']:
                device = Device(parameters['device_name'], True, hamiltonian.N_qubits, layout=parameters['layout'])
            else:
                device = Device(parameters['device_name'], False, hamiltonian.N_qubits, layout=parameters['layout'])

    if device is not None:
        print("Device specifications")
        pprint(vars(device))

    number_cnot_pairs = 0
    if( "number_cnot_pairs" in parameters ): number_cnot_pairs = parameters['number_cnot_pairs']
    number_circuit_folding = 0
    if( "number_circuit_folding" in parameters ): number_cnot_pairs = parameters['number_circuit_folding']
    zero_noise_extrapolation=False
    if( "zero_noise_extrapolation" in parameters): zero_noise_extrapolation = parameters['zero_noise_extrapolation']

    # Run the minimization routine
    if parameters['optimizer'] != 'SPSA':
        if parameters['optimizer'] == 'MIGRAD': # iminuit option
            res = iminimize(compute_energy, theta,
                    args=(backend, hamiltonian,
                        device,
                        noise_model,
                        parameters['N_shots'],
                        number_cnot_pairs,
                        number_circuit_folding,
                        zero_noise_extrapolation),
                    method=parameters['optimizer'], options={'maxfev':parameters['N_iter']})
            # Display diagnostic information:
            # res = iminimize(compute_energy, theta, args=(backend, hamiltonian, device), method=parameters['optimizer'], options={'maxfev':parameters['N_iter'],'disp':True})
        else:
            # "Regular" scipy optimizers
            res = minimize(compute_energy, theta,
                    args=(backend, hamiltonian,
                        device,
                        noise_model,
                        parameters['N_shots'],
                        number_cnot_pairs,
                        number_circuit_folding,
                        zero_noise_extrapolation),
                    method=parameters['optimizer'], options={'maxiter':parameters['N_iter']})
    else:
        res = minimizeSPSA(compute_energy,
                x0=theta,
                args=(backend, hamiltonian,
                    device,
                    noise_model,
                    parameters['N_shots'],
                    number_cnot_pairs,
                    number_circuit_folding,
                    zero_noise_extrapolation),
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
        try:
            os.makedirs(parameters['output_dir'])
        except:
            pass

    # Generate output filename from input parameters
    date=datetime.datetime.now().strftime("%Y-%m-%d")
    dev_name = parameters['device_name']
    if dev_name == 'none':
        dev_name = 'no_device'

    # If measure mitigation used, add tag to filename
   # additionally if initial device layout is specified
    mit=""
    layout_str=""
    CNOT_pairs=""
    folding=""
    dev_name = parameters['device_name']
    if dev_name is None:
        dev_name = 'no_device'
    else:
        if parameters['mitigate_meas_error']:
            mit= "-mit_meas"
    if parameters['layout'] is not None:
        layout_str += "_layout-" + '-'.join(str(e) for e in parameters['layout'])

    if('number_cnot_pairs' in parameters ):
        CNOT_pairs = "-CNOTs"+str(parameters['number_cnot_pairs'])
        if('zero_noise_extrapolation' in parameters ):
            if( parameters['zero_noise_extrapolation'] ): CNOT_pairs += "-zero_noise"

    if('number_circuit_folding' in parameters ):
        folding = "-folding"+str(parameters['number_circuit_folding'])
        if('zero_noise_extrapolation' in parameters ):
            if( parameters['zero_noise_extrapolation'] ): folding += "-zero_noise"

    if parameters['backend'] == 'qasm_simulator':
        output_filename = "{output_dir}/{date}_{encoding}-{N_states}_states-{backend}-{N_shots}_shots-{optimizer}-{device}{layout_str}{mit}{CNOT_pairs}{folding}".format(date=date, device=dev_name, layout_str=layout_str, mit=mit, CNOT_pairs=CNOT_pairs, folding=folding, **parameters)
    else:
        output_filename = "{output_dir}/{date}_{encoding}-{N_states}_states-{backend}-{optimizer}-{device}{layout_str}{mit}{CNOT_pairs}{folding}".format(date=date, device=dev_name, layout_str=layout_str, mit=mit, CNOT_pairs=CNOT_pairs, folding=folding, **parameters)

    return output_filename


