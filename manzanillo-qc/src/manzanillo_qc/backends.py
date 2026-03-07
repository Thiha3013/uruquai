"""PennyLane device factory.

Supported backends
------------------
"default.qubit"    CPU state-vector simulator (always available)
"lightning.qubit"  Xanadu fast C++ simulator (pip install pennylane-lightning)
"qiskit.aer"       IBM Aer simulator          (pip install pennylane-qiskit)
"qiskit.ibmq"      Real IBM hardware          (requires IBMQ account + token)
"""
from __future__ import annotations

import pennylane as qml


def get_device(backend: str, n_qubits: int) -> qml.devices.Device:
    """Return a PennyLane device for the given backend string.

    Parameters
    ----------
    backend : str
        PennyLane device name, e.g. ``"default.qubit"``.
    n_qubits : int
        Number of qubits (wires).

    Returns
    -------
    qml.devices.Device
    """
    return qml.device(backend, wires=n_qubits)
