import os
import pickle 

import numpy as np

from qiskit import QuantumRegister
from qiskit import execute, Aer
from qiskit import IBMQ
from qiskit.providers.aer import noise
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,
                                                 CompleteMeasFitter, 
                                                 MeasurementFilter)

from hamiltonian import *

# List of currently supported devices
supported_devices = [
    None, 
    "ibmq_16_melbourne",
    "ibmq_5_yorktown",
    "ibmq_burlington",
    "ibmq_essex",
    "ibmq_london",
    "ibmq_vigo"
]

class Device():
    def __init__(self, device_name=None, mitigate_meas_error=False, N_qubits=0, layout=None):
        # Default set of parameters
        self.name = device_name
        self.mitigate_meas_error = mitigate_meas_error
        self.coupling_map = None
        self.noise_model = None
        self.meas_filter = None
        self.layout = layout

        if layout is not None:
            assert self.name is not None, \
               f"Layout must None or unspecified when running without a device."
            if len(layout)!=N_qubits:
               error_str = f"The length of the layout list ({len(layout)}) must match the number of qubits in use ({N_qubits})."
               raise ValueError(error_str)

        if self.name not in supported_devices: 
            error_str = f"Please given name of IBMQ machine. Options are: \n {supported_devices}"
            raise ValueError(error_str)

        # If we actually have a device to deal with, do everything else
        if self.name is not None:
            self.read_device()

            if self.mitigate_meas_error:
                if N_qubits <= 0:
                    raise ValueError("Please provide number of qubits for measurement error mitigation.")

                # Because there are issues with running stuff in parallel after calling Qiskit
                # for calibration, let's make things so that we can just save/load this data from a file. 
                if layout is not None:
                    calibration_file = f"device_{self.name}_calibration_{N_qubits}qubits_layout-{layout}.pkl" 
                else:
                    calibration_file = f"device_{self.name}_calibration_{N_qubits}qubits.pkl" 

                if calibration_file not in os.listdir('devices'):
                    print(f"Calibration file not found; creating calibration file at {calibration_file}")
                    self.meas_filter = self.initialize_meas_calibration(N_qubits, layout)
                    with open("devices/" + calibration_file, "wb") as out_file:
                        pickle.dump(self.meas_filter, out_file)
                else:
                    print(f"Calibration file found; reading calibration data from {calibration_file}")
                    with open("devices/" + calibration_file, "rb") as in_file:
                        self.meas_filter = pickle.load(in_file)
        


    def read_device(self):
        """ Reads in noise models for IBMQ device <device_name> stored in file 
            devices/device_<device_name>.pk and returns a tuple containing the 
            coupling map and corresponding noise model.
    
            If the file is not already populated, download and populate it.
        """

        filename = f"device_{self.name}.pk"

        # Check for device information directory, create if not there
        if "devices" not in os.listdir():        
            os.mkdir("devices")
           
        # Check for whether we have already downloaded device information 
        if filename not in os.listdir('devices'):
            # Log into IBMQ using stored account information 
            provider = IBMQ.load_account()
            provider.backends()

            # get device information 
            device = provider.get_backend(self.name)
            properties = device.properties()

            # Get coupling map and noise model 
            coupling_map = device.configuration().coupling_map
            noise_model = noise.NoiseModel.from_backend(properties)

            # Write tuple contianing coupling map and noise model (converted to dictionary) to file 
            device_write = (coupling_map, noise_model.to_dict())

            with open(f'devices/{filename}','wb') as out_file:
                doc = pickle.dump(device_write, out_file)

        # Read data from the file
        with open(f'devices/{filename}', 'rb') as in_file:
            coupling_map, model_dict = pickle.load(in_file)

            # Reconstruct noise model from dictionary
            noise_model = noise.noise_model.NoiseModel.from_dict(model_dict)

            # Now, we set the class variables
            self.coupling_map = coupling_map
            self.noise_model = noise_model


    def initialize_meas_calibration(self, N_qubits, layout):
        """ Set up the confusion matrix needed for measurement error mitigation.
            This is basically just boilerplate code from the Ignis Github
            https://github.com/Qiskit/qiskit-ignis
        """
        if layout is None:
            cal_q = QuantumRegister(N_qubits)
            meas_cals, state_labels = complete_meas_cal(qr=cal_q)
        else:
            meas_cals, state_labels = complete_meas_cal(qubit_list=layout)

        # Run the calibration circuits with the device noise model
        backend = Aer.get_backend('qasm_simulator')
        job = execute(meas_cals, backend=backend, shots=10000, noise_model=self.noise_model)
        cal_results = job.result()

        return CompleteMeasFitter(cal_results, state_labels).filter
