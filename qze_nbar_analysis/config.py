from pathlib import Path

# where the qubit_4roundN folders live
DATA_ROOT = Path("/Users/olivias/Library/CloudStorage/Dropbox/Bad qubit_Fat TLSs/Olivia/all_qubits")

# zero based, matches folder qubit_4roundN and h5 group Q5
QUBIT_INDEX = 4

# "auto" grabs every round folder present
ROUNDS = "auto"

CHI_MHZ = -0.137

QSPEC_SUBDIR = "QSpec_zeno"
T1_SUBDIR = "T1_ge_zeno"
T2R_SUBDIR = "T2_ge_zeno"
T2E_SUBDIR = "T2E_ge_zeno"

MIN_QSPEC_POINTS = 5
MIN_T1_POINTS = 4

OUT_DIR = Path(__file__).resolve().parent / "outputs"
