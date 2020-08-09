import numpy as np 
import pytest

from src.hamiltonian import *

# These are the "true" ground state energies of the deuteron Hamiltonian
# Our Hamiltonians should be able to reproduce all these values.
diagonalized_values = {
  1  : -0.436581,
  2  : -1.749160,
  3  : -2.045671,    
  4  : -2.143981, 
  5  : -2.183592, 
  6  : -2.201568, 
  7  : -2.210416, 
  8  : -2.215038,
  16 : -2.221059
}

class TestSparseHamiltonian:
    # Ensure the ground state energies are correct
    def test_energies(self):
        for N in diagonalized_values.keys():
            H = SparseEncodingHamiltonian(N_states=N)
            energy = np.min(np.linalg.eigh(H.matrix)[0])
            assert np.isclose(energy, diagonalized_values[N])

    def test_compare_to_richard(self):
        pauli_dict4 = {
            "IIII" : 28.657,
            "IIIZ" : 0.218,
            "IIZI" : -6.125,
            "IZII" : -9.625,
            "ZIII" : -13.125,
            "IIXX" : -2.143303,
            "IIYY" : -2.143303,
            "IXXI" : -3.91312,
            "IYYI" : -3.91312,
            "XXII" : -5.670648,
            "YYII" : -5.670648 
        }
             
        H = SparseEncodingHamiltonian(N_states=4, qiskit_order=True)
        for pauli in pauli_dict4.keys():
            assert np.isclose(pauli_dict4[pauli], H.pauli_coeffs[pauli], atol=1e-3)

    # Make sure we have the correct number of Pauli partitions
    # For the sparse case, this is always 3, things with X, Y, or Z
    def test_partition_length(self):
        for N in diagonalized_values.keys():
            H = SparseEncodingHamiltonian(N_states=N)
            assert H.n_partitions == 3    


class TestDenseHamiltonian:
    # Dense Hamiltonians are not implemented for N = 1 case at the moment
    # So ignore for now.
    def test_energies(self):
        for N in diagonalized_values.keys():
            H = DenseEncodingHamiltonian(N_states=N)
            energy = np.min(np.linalg.eigh(H.matrix)[0])
            assert np.isclose(energy, diagonalized_values[N])

    def test_energies_qiskit(self):
        for N in diagonalized_values.keys():
            H = DenseEncodingHamiltonian(N_states=N, qiskit_order=True)
            energy = np.min(np.linalg.eigh(H.matrix)[0])
            assert np.isclose(energy, diagonalized_values[N])

    def test_compare_to_richard_4state(self):
        pauli_dict2 =  {
            "II" : 14.3283547225, 
            "IZ" : -1.4216452775, 
            "ZI" : -8.4216452775, 
            "ZZ" : -4.9216452775, 
            "IX" : -7.8139515, 
            "ZX" : 3.5273445, 
            "XI" : -3.91312, 
            "XZ" : 3.91312
        }

        # Richard's Hamiltonian is in qiskit order and mine should match his
        H = DenseEncodingHamiltonian(N_states=4, qiskit_order=True)
        for pauli in pauli_dict2.keys():
            assert np.isclose(pauli_dict2[pauli], H.pauli_coeffs[pauli])


    def test_compare_to_richard_8state(self):
        pauli_dict3 = {
            "III" : 29.03917736125,
            "IIZ" : -0.71082263875,
            "IZI" : -0.71082263875,
            "ZII" : -14.7108226387,
            "IZZ" : -0.71082263875,
            "ZZI" : -7.71082263875,
            "ZIZ" : -0.71082263875,
            "ZZZ" : -4.21082263875,
            "IIX" : -14.8354918425,
            "IZX" : 0.0122336875,
            "ZIX" : 7.0215402325,
            "ZZX" : 3.5151109225,
            "IXI" : -7.4209327225, 
            "IXZ" : 7.4209327225, 
            "ZXI" : 3.5078137725, 
            "ZXZ" : -3.5078137725, 
            "XII" : -3.7123106,   
            "XIZ" : -3.7123106,   
            "XZI" : 3.7123106,    
            "XZZ" : 3.7123106
        }    
                      
        # Richard's Hamiltonian is in qiskit order and mine should match his
        H = DenseEncodingHamiltonian(N_states=8, qiskit_order=True)
        for pauli in pauli_dict3.keys():
            assert np.isclose(pauli_dict3[pauli], H.pauli_coeffs[pauli])


    def test_partition_length(self):
        for N in diagonalized_values.keys():
            H = DenseEncodingHamiltonian(N_states=N)
            assert H.n_partitions == np.ceil(np.log2(N)) + 1
