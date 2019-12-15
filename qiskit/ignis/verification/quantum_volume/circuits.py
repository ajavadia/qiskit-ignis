# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Generates quantum volume circuits
"""

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info.random import random_unitary


def model_circuit(width, depth=None, seed=None):
    """Create quantum volume model circuit of size width x depth
    (default depth is equal to width), with a random seed.

    The model circuits consist of layers of Haar random
    elements of SU(4) applied between corresponding pairs
    of qubits in a random bipartition.

    Args:
        width (int): number of active qubits in model circuit
        depth (int): layers of SU(4) operations in model circuit
        seed (int): randomization seed

    Returns:
        QuantumCircuit: a quantum volume model circuit
    """
    depth = depth or width

    qr = QuantumRegister(width, 'q')
    cr = ClassicalRegister(width, 'c')
    circuit = QuantumCircuit(qr, cr)

    for _ in range(depth):

        # Generate uniformly random permutation Pj of [0...n-1]
        rng = np.random.RandomState(seed)
        perm = rng.permutation(width)

        # For each pair p in Pj, generate Haar random SU(4)
        for k in range(int(np.floor(width/2))):
            U = random_unitary(4, seed=seed)
            pair = int(perm[2*k]), int(perm[2*k+1])
            physical_q_0 = qr[pair[0]]
            physical_q_1 = qr[pair[1]]
            circuit.append(U, [physical_q_0, physical_q_1])

    circuit.barrier(qr)
    circuit.measure(qr, cr)

    return circuit


def qv_circuits(qubit_lists=None, ntrials=1):
    """
    Return circuit sequences for a quantum volume experiment.

    On each list of qubits, sequences of random circuits
    of incresing depth are generated.

    Args:
        qubit_lists: list of list of qubits to apply qv circuits to. Assume
            the list is ordered in increasing number of qubits
        ntrials: number of random iterations

    Returns:
        list[list[QuantumCircuit]]: qv circuit sequences
            (separate list for each trial)
    """
    circuits = [[] for e in range(ntrials)]

    for trial in range(ntrials):
        for qubit_list in qubit_lists:
            width = len(qubit_list)
            max_qubit = np.max(qubit_list)

            qc = model_circuit(width=width, depth=width)

            # embed on physical qubits of interest
            qc = transpile(qc, coupling_map=CouplingMap.full(max_qubit+1),
                           initial_layout=qubit_list)

            qc.name = 'qv_width_%d_trial_%d' % (width, trial)

            circuits[trial].append(qc)

    return circuits
