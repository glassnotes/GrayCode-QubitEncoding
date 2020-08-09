import numpy as np 
import pytest

from src.utils import *

from openfermion import QubitOperator

class TestGrayCode:
    def test_base_case(self):
        assert gray_code(1) == ['0', '1']

    def test_2_bit_code(self):
        assert gray_code(2) == ['00', '01', '11', '10']
        
    def test_3_bit_code(self):
        assert gray_code(3) == ['000', '001', '011', '010', '110', '111', '101', '100']
       
    def test_invalid_float_input(self): 
        with pytest.raises(ValueError):        
            gray_code(3.4)

    def test_invalid_zero_input(self): 
        with pytest.raises(ValueError):        
            gray_code(0)


class TestFindFlippedBit:
    def test_uneven_length(self):
        with pytest.raises(ValueError):        
            find_flipped_bit('0100', '11')
            find_flipped_bit('01', '011')

    def test_zero_length(self):
        with pytest.raises(ValueError):        
            find_flipped_bit('', '11')
            find_flipped_bit('100', '')
            find_flipped_bit('', '')

    def test_non_gray_code(self):
        with pytest.raises(ValueError):        
            find_flipped_bit('011', '110')
            find_flipped_bit('00011', '10010')

    def test_non_binary_input(self):
        with pytest.raises(ValueError):
            find_flipped_bit('b01', '1b0')

    def test_gray_code(self):
        assert find_flipped_bit('11', '10') == 1
        assert find_flipped_bit('11110', '11010') == 2
        assert find_flipped_bit('100110011', '000110011') == 0


class TestProjectorExpansion:
    def test_invalid_input(self):
        with pytest.raises(ValueError):
            expand_projector_sequence([])
            expand_projector_sequence(["Q"])
            expand_projector_sequence(["P1", "X", "X", "Z", "S"])
            
    def test_projector_input(self):
        assert str(expand_projector_sequence(["P0"])) == "0.5 [] +\n0.5 [Z0]"          
        assert str(expand_projector_sequence(["P1"])) == "0.5 [] +\n-0.5 [Z0]"  
        assert str(expand_projector_sequence(["P0", "P1"])) == "0.25 [] +\n0.25 [Z0] +\n-0.25 [Z0 Z1] +\n-0.25 [Z1]"  

    def test_single_qubit_input(self):
        assert str(expand_projector_sequence(["X"])) == "1.0 [X0]"        
        assert str(expand_projector_sequence(["Y"])) == "1.0 [Y0]"
        assert str(expand_projector_sequence(["Z"])) == "1.0 [Z0]"

    def test_multi_qubit_input(self):
        assert str(expand_projector_sequence(["X", "Z"])) == "1.0 [X0 Z1]"
        assert str(expand_projector_sequence(["I", "Z", "I", "Y"])) == "1.0 [Z1 Y3]"
        assert str(expand_projector_sequence(["P1", "X", "P0"])) == "-0.25 [Z0 X1] +\n-0.25 [Z0 X1 Z2] +\n0.25 [X1] +\n0.25 [X1 Z2]"


class TestPauliGenerators:
    def test_invalid_input(self):
        with pytest.raises(ValueError):
            pauli_generators(0)
            pauli_generators(2, 3)

    def test_z_only_inputs(self):
        assert pauli_generators(1) == ['Z']
        assert pauli_generators(2) == ['ZI', 'IZ']        
        assert pauli_generators(4) == ['ZIII', 'IZII', 'IIZI', 'IIIZ']

    def test_x_inputs(self):
        assert pauli_generators(2, 0)  == ['XZ', 'XI']        
        assert pauli_generators(2, 1)  == ['ZX', 'IX']
        assert pauli_generators(3, 0)  == ['XZI', 'XIZ']
        assert pauli_generators(4, 2)  == ['ZIXI', 'IZXI', 'IIXZ']


class TestGetPauliMatrix:
    def test_invalid_input(self):
        with pytest.raises(ValueError):
            get_pauli_matrix('XXXA')            
            get_pauli_matrix('IBZI')
            get_pauli_matrix('YYCD')

    def test_standard_order(self):
        assert np.allclose(get_pauli_matrix('X'), np.array([[0, 1], [1, 0]]))
        assert np.allclose(get_pauli_matrix('ZX'), 
            np.array([[0, 1, 0, 0], 
                      [1, 0, 0, 0],
                      [0, 0, 0, -1],
                      [0, 0, -1, 0]]))
        assert np.allclose(get_pauli_matrix('XIY'),
            np.array([[0, 0, 0, 0, 0, -1j, 0, 0],
                      [0, 0, 0, 0, 1j, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, -1j],
                      [0, 0, 0, 0, 0, 0, 1j, 0],
                      [0, -1j, 0, 0, 0, 0, 0, 0],
                      [1j, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, -1j, 0, 0, 0, 0],
                      [0, 0, 1j, 0, 0, 0, 0, 0]]))


class TestXanaduPauli:
    def test_invalid_input(self):
        with pytest.raises(ValueError):
            string_to_xanadu_pauli('')
            string_to_xanadu_pauli('A')
            string_to_xanadu_pauli('ZYAC')

    def test_string_to_pauli(self):
        # TODO - need to find a way to get matrix rep or something
        assert True