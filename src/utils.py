import numpy as np 
from openfermion.ops import QubitOperator
from itertools import product
from functools import reduce

mats = {"I" : np.eye(2),
        "X" : np.array(([[0, 1], [1, 0]])),
        "Y" : np.array(([[0, -1j], [1j, 0]])),
        "Z" : np.array(([[1, 0], [0, -1]])),
        "P0" : np.array(([[1, 0], [0, 0]])),
        "P1" : np.array(([[0, 0], [0, 1]]))}


def gray_code(N):
    """ Generate a Gray code for traversing the N qubit states. """
    if N <= 0 or type(N) is not int:
        raise ValueError("Input for gray code construction must be a positive integer.")
    if N == 1: # Base case
        return ["0", "1"]
    else:
        sub_code = gray_code(N-1)
        return ["0" + x for x in sub_code] + ["1" + x for x in sub_code[::-1]]


def find_flipped_bit(s1, s2):
    """ For two adjacent elements in a gray code, determine which bit is the 
    one that was flipped.
    """
    if len(s1) == 0 or len(s2) == 0:
        raise ValueError("Empty string inputted.")

    if len(s1) != len(s2):
        raise ValueError("Strings compared in gray code must have the same length.")
    
    if any([x != "0" and x != "1" for x in s1]) or any([x != "0" and x != "1" for x in s2]):
        raise ValueError(f"One of inputs {s1}, {s2} is not a valid binary string.")
    
    # Sum the strings elementwise modulo 2; the sum will be 1 only in the slot 
    # where we flipped a bit        
    string_sums = [(int(s1[i]) + int(s2[i])) % 2 for i in range(len(s1))]

    if string_sums.count(1) == 0:
        raise ValueError(f"Strings {s1} and {s2} are the same.")
    elif string_sums.count(1) > 1:
        raise ValueError(f"Strings {s1} and {s2} are not ordered in a gray code.")

    return string_sums.index(1)


def expand_projector_sequence(seq):
    # Take a list of projectors, e.g. ["P0", "P1", "X"] and expand it in terms of Paulis 
    # return an openfermion QubitOperator

    # Copy the sequence before making replacements
    substitution_seq = seq

    if len(seq) <= 0:
        raise ValueError(f"Cannot expand empty projector sequence.")

    if any([x not in mats.keys() for x in seq]):
        raise ValueError(f"Sequence {seq} contains elements that are not Paulis or P0/P1 projectors.")  

    prefactor = 1 / (2 ** (substitution_seq.count("P0") + substitution_seq.count("P1")))

    # First, replace P0 and P1 with 0.5 (1 +- Z)
    for item_idx in range(len(seq)):
        if seq[item_idx] == "P0":
            substitution_seq[item_idx] = ["I", "Z"]
        elif seq[item_idx] == "P1":
            substitution_seq[item_idx] = ["I", "mZ"]

    qubit_operators = []

    # Expand out the term into individual Paulis
    for pauli in product(*substitution_seq):
        pauli_string = "".join(pauli)

        # Extract the sign and remove the m indicators
        sign = (-1) ** pauli_string.count("m")
        pauli_string = pauli_string.replace("m", "")

        # Remove identities and label Paulis with their qubit indices
        qubit_operator_string = ""
        for qubit_idx in range(len(pauli_string)):
            if pauli_string[qubit_idx] != "I":
                qubit_operator_string += f"{pauli_string[qubit_idx]}{qubit_idx} "
        qubit_operators.append(QubitOperator(qubit_operator_string, sign*prefactor))

    full_operator = QubitOperator()
    for term in qubit_operators:
        full_operator += term

    return full_operator 


def pauli_generators(N, x_loc=None):
    """ Construct a list of strings of Pauli generators on N qubits. If x_loc is set
    to an integer, then we will construct the generators on N qubits where the x_loc qubit
    is set to X and the remaining qubits contain the generators of N - 1 qubits.

    For example, 
        pauli_generators(4) = ['ZIII', 'IZII', 'IIZI', 'IIIZ']
        pauli_generators(4, 2) = ['ZIXI', 'IZXI', IIXZ'] 
    """
    if N < 1:
        raise ValueError("Number of Paulis must be >= 1 to construct generators.") 

    if x_loc is None:
        return ["I" * idx + "Z" + "I" * (N - idx - 1) for idx in range(N)]
    else:
        if x_loc < 0 or x_loc > N:
            raise ValueError(f"Invalid placement ({x_loc}) X in {N}-qubit Pauli.")
        base_generators = [list("I" * idx + "Z" + "I" * (N - idx - 2)) for idx in range(N - 1)]
        
        # If we have two qubits, need to add I to the generator list
        if N == 2:
            base_generators.append(["I"])

        for idx in range(len(base_generators)):
            base_generators[idx].insert(x_loc, "X")
        return ["".join(gen) for gen in base_generators]


def get_pauli_matrix(pauli):
    """ Take a Pauli string and compute its matrix representation.

    Parameters:
        pauli (string): A string indicating the Pauli whose expectation value
            we want to compute, e.g. "ZZIZZ". Tensor products are computed
            from left to right here.
    """
    pauli_list = list(pauli)

    if any([op not in ['I', 'X', 'Y', 'Z'] for op in pauli_list]):
        raise ValueError("Pauli string must consist only of I, X, Y, or Z.")

    return reduce(np.kron, [mats[sigma_idx] for sigma_idx in pauli_list])


def pauli_expectation_value(pauli, meas_results):
    """ Compute and return the expectation value of a given Pauli
    based on the measurement outcomes observed in result.


    Parameters:
        pauli (string): A string indicating the Pauli whose expectation value
            we want to compute, e.g. "ZZIZZ"
        meas_results (Dict): A dictionary containing the results of an experiment run on qiskit.
            The key value pairs are computational basis states and number of times that
            state was observed, e.g. {'1001' : 24, '1000' : 36}, etc.

    Returns:
        The expectation value of pauli.        
    """
    pauli_list = list(pauli)
    n_qubits = len(pauli_list)
    n_shots = sum(meas_results.values())

    if any([op not in ['I', 'X', 'Y', 'Z'] for op in pauli_list]):
        raise ValueError("Pauli string must consist only of I, X, Y, or Z.")
    
    # Determine whether the computational basis states in meas_results are +1 or -1
    # eigenstates of the Pauli in question.
    eigenvalues = {}
    for basis_state in meas_results.keys():
        num_z_and_1 = [-1 if (basis_state[bit_idx] == '1' and pauli_list[bit_idx] != 'I') else 1 for bit_idx in range(n_qubits)]
        eigenvalues[basis_state] = reduce(lambda x, y: x*y, num_z_and_1)

    # Count number of +1 and -1 outcomes, i.e. 0 and 1
    num_0_outcomes = sum([meas_results[key] for key in eigenvalues.keys() if eigenvalues[key] == 1])    
    num_1_outcomes = sum([meas_results[key] for key in eigenvalues.keys() if eigenvalues[key] == -1])

    return (num_0_outcomes - num_1_outcomes) / n_shots