from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit

import numpy as np

def sparse_variational_circuit(thetas, measured_idx, backend_name):
    """ Creates a variational ansatz for the Jordan-Wigner encoding.

    These circuits were defined in arXiv:1904.04338, and produce an ansatz state
    over the occupation subset of the computational basis (|1000>, |0100>, etc.)
    with generalized spherical coordinates as their amplitudes.

    The version here is the same circuit, but with the qubit order inverted. 
    This was found to lead to greater stability in the SPSA optimization procedure
    due to the relationships between of the variational parameters and the 
    "strength" of the basis states in the Hamiltonian. 

    Parameters:
        thetas (np.array) : Angles parameterizing ansatz wavefunction.  Number of 
            angles is one fewer than the number of qubits in the circuit. 

        measured_idx (int) : Either 0, 1, or 2 corresponding to X, Y, and Z bases. The Pauli operators 
            in these Hamiltonians are either all X, all Y, or all Z, and never a mix, so the measurement 
            basis is easily specified and applied to all qubits.
        
        backend_name (str) : name of the backend

    Returns:
        circuit (QuantumCircuit) : quantum circuit 
    
    """
    # The number of parameters tells us the number of states
    N_states = len(thetas) + 1
    N_qubits = N_states

    if measured_idx > 3 or measured_idx < 0:
        raise ValueError("Sparse Hamiltonians have only 3 sets of commuting operators.")
    
    q, c = QuantumRegister(N_qubits), ClassicalRegister(N_qubits)
    circuit = QuantumCircuit(q, c)
   
    circuit.x(q[N_qubits-1])
    circuit.ry(thetas[0], q[N_qubits-2])
    circuit.cx(q[N_qubits-2], q[N_qubits-1])
  
    # Recursive cascade 
    for control_idx in range(N_qubits - 2, 0, -1):
        target_idx = control_idx - 1
        
        circuit.cry(thetas[control_idx], q[control_idx], q[target_idx])
        circuit.cx(q[target_idx], q[control_idx])

    if backend_name == 'qasm_simulator':
        # Set the measurement basis - 0 is X, 1 is Y, 2 is Z
        if measured_idx < 2:
            if measured_idx == 1: # Y measurements
                for qubit_idx in range(N_qubits):
                    circuit.sdg(q[qubit_idx])

            for qubit_idx in range(N_qubits): # Y or X get Hadamards 
                circuit.h(q[qubit_idx])
    
        circuit.measure(q, c)
    
    return circuit


def dense_variational_circuit(thetas, measured_idx, backend_name):
    """ Creates a variational ansatz for the Gray code encoding.

    These circuits have a more familiar 'variational form'. Since in the dense case we
    are using all elements in the space, we can use something a bit more general. These
    circuits are layers of Y rotations followed by entangling gates on all the qubits.

    Parameters:

        thetas (np.array) : Angles parameterizing ansatz wavefunction.  Number of 
            angles is one fewer than the number of qubits in the circuit. 

        measured_idx (int): The Paulis in this Hamiltonian are partitioned into N + 1 commuting
            sets. For each set, only a single qubit will be measured in the X basis, and this is
            what is passed. The case of Z is handled by measured_idx = N_qubits.
        
        backend_name (str) : name of the backend

    Returns:
        circuit (QuantumCircuit) : quantum circuit 
        
    """
    N_states = len(thetas) + 1
    N_qubits = int(np.ceil(np.log2(N_states)))

    if measured_idx:
        if measured_idx < 0 or measured_idx >= N_qubits:
            raise ValueError("Invalid measurement index. Dense Hamiltonians have N_qubits + 1 commuting operators.")

    q, c = QuantumRegister(N_qubits), ClassicalRegister(N_qubits)
    
    circuit = QuantumCircuit(q, c)
    
    num_remaining_thetas = len(thetas)
    qubit_idx = 0
    last_layer = False

    while num_remaining_thetas > 0:
        # If it's the last layer, only do as many CNOTs as we need to
        if last_layer:
            for i in range(num_remaining_thetas):
                circuit.cx(q[i], q[(i+1)%N_qubits])
            for qubit in range(1, num_remaining_thetas+1):
                circuit.ry(thetas[num_remaining_thetas - 1], q[qubit])
                num_remaining_thetas -= 1
            break
        # If it's not the last layer, apply rotations and then increment counters
        else: 
            circuit.ry(thetas[num_remaining_thetas - 1], q[qubit_idx])
            qubit_idx += 1
            num_remaining_thetas -= 1

            if N_qubits >= 2:
                # If we have a full layer of parameters ahead, do the full CNOT cycle
                if qubit_idx == N_qubits and num_remaining_thetas > N_qubits:
                    for i in range(N_qubits):
                        circuit.cx(q[i], q[(i+1)%N_qubits])
                    qubit_idx = 0
                # Otherwise, we are entering the last layer
                elif qubit_idx == N_qubits and num_remaining_thetas <= N_qubits:
                    last_layer = True

    if backend_name == 'qasm_simulator': 
        # Rotate the basis of the qubit that is being measured
        if measured_idx is not None:
            circuit.h(q[measured_idx])
        
        circuit.measure(q, c)
                
    return circuit
