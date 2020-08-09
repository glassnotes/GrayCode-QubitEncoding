import numpy as np

from itertools import product, chain

from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner 
from openfermion.transforms import get_sparse_operator 
from openfermion.utils import get_ground_state 

from utils import * 

class EncodingHamiltonian():
    # Values of physical quantities, shared by every Hamiltonian.
    V0 = -5.68658111
    hw = 7    

    def __init__(self, N_qubits, N_states, qiskit_order=True):
        self.N_qubits = N_qubits
        self.N_states = N_states
        self.ferm_rep = self._generate_ferm_rep()
        self.qiskit_order = qiskit_order

    def _T(self, n, n_prime):
        # Kinetic energy 
        delta = int(n == n_prime)
        delta_p1 = int(n == n_prime + 1)
        delta_m1 = int(n == n_prime - 1)

        return (self.hw/2) * ((2*n+1.5)*delta - np.sqrt(n*(n+0.5))*delta_p1 - np.sqrt((n+1)*(n+1.5))*delta_m1)

    def _V(self, n, n_prime):
        # Potential energy
        return self.V0 * int((n == 0) and (n == n_prime))

    def _generate_ferm_rep(self):
        # Construct the Fermionic representation of this Hamiltonian
        # T and V for the 0th term are constant
        H = FermionOperator('1^ 1', self._V(0, 0) + self._T(0, 0))

        for n, n_prime in product(range(self.N_states), repeat=2):
            if n == 0 and n_prime == 0:
                continue

            H += FermionOperator(f"{n_prime+1}^ {n+1}", self._V(n, n_prime) + self._T(n, n_prime))

        return H

    def _separate_coeffs(self):
        """ Pulls out the coefficients of each Pauli and stores in a dictionary separate.
        Useful for computing the expectation value because we can look up coeffs easily.
        """
        all_paulis = {}
        for set_idx, measurement_setting in self.pauli_partitions.items():
            for pauli, coeff in measurement_setting.items():
                all_paulis[pauli] = coeff.real
        return all_paulis


class SparseEncodingHamiltonian(EncodingHamiltonian):
    def __init__(self, N_states, qiskit_order=True):    
        """ Class for the Jordan-Wigner encoded Hamiltonian.  Based on Hamiltonian 
            in original deuteron calculation paper arXiv:1801.03897.

        Parameters: 
            N_states (int): The number of harmonic oscillator states to consider. For sparse
                encoding, this is the same as the number of qubits. 

            qiskit_order (bool): Determines whether to order the qubits in qiskit order, i.e.
                in "reverse" as compared to the typical ordering. 
        """
        super(SparseEncodingHamiltonian, self).__init__(N_states, N_states, qiskit_order)

        self.pauli_rep = jordan_wigner(self.ferm_rep) # In terms of Paulis; for sparse Hamiltonian just JW
        self.relabel_qubits() # 0 index the qubits to match dense representation

        self.pauli_partitions = self._partition()
        self.n_partitions = len(self.pauli_partitions.keys())
        self.pauli_coeffs = self._separate_coeffs()
        self.matrix = self._to_matrix() # Numerical matrix

    def _to_matrix(self):
        mat = np.zeros((self.N_states, self.N_states))

        # For each term in the Hamiltonian, populate the relevant entry in the matrix
        for ferm_op in self.ferm_rep:
            dag_op, op = list(ferm_op.terms.keys())[0]
            dag_idx, op_idx = dag_op[0]-1, op[0]-1
            mat[dag_idx, op_idx] = ferm_op.terms[(dag_op, op)]

        return mat

    def relabel_qubits(self):
        # The Jordan-Wigner transform from OpenFermion gives us Paulis where 
        # the qubits are 1-indexed. Convert to 0 indexing, and if we are using
        # Qiskit order, we also have to reverse the order of the Paulis.
        new_pauli_rep = QubitOperator()

        for pauli, coeff in self.pauli_rep.terms.items():
            operator_string = ""
            for qubit in pauli:
                operator_string += (qubit[1] + str(qubit[0] - 1) + " ")
            new_pauli_rep += coeff.real * QubitOperator(operator_string)

        self.pauli_rep = new_pauli_rep

    def _partition(self):
        # Partition the Paulis into commuting sets; for a sparse Hamiltonian, there are 3 sets,
        # ones with only X, ones with only Y, and ones with only Z
        term_bins = {}
        pauli_keys = list(self.pauli_rep.terms.keys()) 
        
        # Now pull out all the terms that are Z only    
        z_only_terms = {}
        x_only_terms = {}
        y_only_terms = {}

        for pauli in pauli_keys:
            pauli_string = ["I"] * self.N_qubits
            for qubit_term in pauli:
                pauli_string[qubit_term[0]] = qubit_term[1]
            flat_list = list(chain(*[list(qubit_term) for qubit_term in pauli]))

            # If we are using qiskit, reverse the string so that the flipped Paulis
            # will get the correct coefficients
            pauli_string_ordered = pauli_string

            if self.qiskit_order:
                pauli_string_ordered = pauli_string_ordered[::-1]

            if 'X' in flat_list:
                x_only_terms["".join(pauli_string_ordered)] = self.pauli_rep.terms[pauli]
            elif 'Y' in flat_list:
                y_only_terms["".join(pauli_string_ordered)] = self.pauli_rep.terms[pauli]
            else: 
                z_only_terms["".join(pauli_string_ordered)] = self.pauli_rep.terms[pauli]
       
        term_bins[0] = x_only_terms 
        term_bins[1] = y_only_terms 
        term_bins[2] = z_only_terms 

        return term_bins


class DenseEncodingHamiltonian(EncodingHamiltonian):
    def __init__(self, N_states, qiskit_order=True, kill_bad_states=True):
        """ Class for Gray code encoding that uses N qubits to represent 2^N states.  [TODO:REF]

        Parameters:
            N_states (int) : The number of harmonic oscillator states to consider. For this
                encoding, the number of qubits will be Ceiling[log2[N_states]].

            qiskit_order (bool,optional) : Determines whether to order the qubits in qiskit order, i.e.
                in "reverse" as compared to the typical ordering. Default : True.

            kill_bad_states (bool,optional) : If N_states is not a power of 2, add additional 
                projector terms to "kill" those terms.
        """
        N_qubits = int(np.ceil(np.log2(N_states)))

        if N_states == 1:
            N_qubits = 1

        self.kill_in_between_terms = False
        if (N_states != 2 ** N_qubits) and kill_bad_states:
            self.kill_in_between_terms = True

        super(DenseEncodingHamiltonian, self).__init__(N_qubits, N_states, qiskit_order)

        # Get the order of the states in the gray code
        self.state_order = gray_code(self.N_qubits)
        self.permutation = [int("0b" + x, 2) for x in self.state_order] 

        # Pauli representation is not Jordan-Wigner anymore, it is the sequence of projectors 
        # that produce the gray code sequence. Outsource to another class method.
        self.pauli_rep = self._build_pauli_rep() 
        self.pauli_partitions = self._partition()
        self.n_partitions = len(self.pauli_partitions.keys())
        self.pauli_coeffs = self._separate_coeffs()
        self.matrix = self._to_matrix()
    
    def _build_pauli_rep(self):
        diagonal_terms = self._build_diagonal()
        off_diagonal_terms = self._build_off_diagonal()
        return diagonal_terms + off_diagonal_terms

    def _build_diagonal(self):
        # Diagonal terms represent the number operator, N|n> = n |n>
        # Each diagonal term will consist of the sequence of projectors corresponding to the
        # constituent values in the bit string, multiplied by the decimal value of that bit string.
        # i.e. |1> = |01> -> P0 P1, |2> = |11> -> P1 P1
        diagonal = QubitOperator() 

        for state_idx in range(self.N_states):
            coeff = (self._T(state_idx, state_idx) + self._V(state_idx, state_idx))
            projector_sequence = [f"P{x}" for x in self.state_order[state_idx]]
            diagonal += coeff * expand_projector_sequence(projector_sequence[::-1])
        # If number of states is not a power of 2, add additional projector terms that will 
        # "kill" those terms
        if self.kill_in_between_terms:
            for bad_state in self.state_order[self.N_states:]:
                projector_sequence = [f"P{x}" for x in bad_state]
                diagonal += 100 * expand_projector_sequence(projector_sequence[::-1])

        return diagonal

    def _build_off_diagonal(self):
        """ In this step, we look at the sequence of terms in the gray code and make projectors
        depending on which bit changes.
        """
        off_diagonal = QubitOperator() 

        for state_idx in range(self.N_states - 1):
            flip_location = find_flipped_bit(self.state_order[state_idx], self.state_order[state_idx+1])
            projector_sequence = [f"P{x}" for x in self.state_order[state_idx]]
            projector_sequence[flip_location] = "X"

            coeff = (self._T(state_idx, state_idx + 1) + self._V(state_idx, state_idx + 1))
            off_diagonal += coeff * expand_projector_sequence(projector_sequence[::-1])           

        return off_diagonal

    def _to_matrix(self):
        if self.qiskit_order:
            return reduce(lambda x, y: x + y, [p[1] * get_pauli_matrix(p[0][::-1]) for p in self.pauli_coeffs.items()])
        else:
            return reduce(lambda x, y: x + y, [p[1] * get_pauli_matrix(p[0]) for p in self.pauli_coeffs.items()])

    def _partition(self):
        # Partition the Paulis and their co-efficients here
        # I'm sorry this is so ugly
        term_bins = {} 
        pauli_keys = list(self.pauli_rep.terms.keys())

        # First, pull out all the terms that have X in a fixed location
        if self.N_qubits >= 1 and len(self.ferm_rep.terms) != 1:
            for qubit_idx in range(0, self.N_qubits):
                qubit_idx_terms = {}
                for pauli in pauli_keys:
                    pauli_string = ["I"] * self.N_qubits
                    for qubit_term in pauli:
                        pauli_string[qubit_term[0]] = qubit_term[1]
                    if pauli.count((qubit_idx, 'X')) == 1:
                        # If we are using qiskit, reverse the string so that the flipped Paulis
                        # will get the correct coefficients
                        qubit_idx_terms["".join(pauli_string[::-1])] = self.pauli_rep.terms[pauli]

                # Populate the dictionary, taking care to correctly indicate the qubit to measure
                term_bins[qubit_idx] = qubit_idx_terms

        # Now pull out all the terms that are Z only    
        z_only_terms = {}    
        for pauli in pauli_keys:
            pauli_string = ["I"] * self.N_qubits
            for qubit_term in pauli:
                pauli_string[qubit_term[0]] = qubit_term[1]
            flat_list = list(chain(*[list(qubit_term) for qubit_term in pauli]))

            if 'X' not in flat_list:
                z_only_terms["".join(pauli_string[::-1])] = self.pauli_rep.terms[pauli]

        term_bins[self.N_qubits] = z_only_terms 
        
        return term_bins