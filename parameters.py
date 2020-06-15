

import numpy as np

"""
This file includes all the parameters necessary for the autoscheduler. It includes two types of parameters.
One: problem dependent parameters. These are parameters specific to the tensor algebra problem. For example, 

"""

PROBLEM = "SPMM"
HARDWARE = "CPU"

PRECOMPUTE_REDUCTION = True if (PROBLEM == "SPMV" and HARDWARE == "GPU") else False

PRECOMPUTE_VAR = "j" if (PROBLEM == "MTTKRP" and HARDWARE == "CPU") else None
MANUAL_PRECOMPUTE = True if (PROBLEM == "MTTKRP" and HARDWARE == "CPU") else False
MANUAL_PRECOMPUTED_VARS = ["l","j"]

SPARSE_TENSOR_MAP = {}
SPARSE_TENSOR_MAP["SPMM"] = "A(i,j)"
SPARSE_TENSOR_MAP["SPMV"] = "A(i,j)"
SPARSE_TENSOR_MAP["MTTKRP"] = "B(i,k,l)"
SPARSE_TENSOR_MAP["TTV"] = "B(i,j,k)"
SPARSE_TENSOR_MAP["SDDMM"] = "B(i,j)"
# this is not going to work if you do pos(i). this is something
# we'll need to fix. you need to choose which sparse tensor's position
# space to be in.
SPARSE_TENSOR_MAP["SPMSPV"] = "A(i,j)"
SPARSE_TENSOR = SPARSE_TENSOR_MAP[PROBLEM]

PROBLEM_MATRIX_MAP = {}

PROBLEM_MATRIX_MAP["SPMSPV"] = np.array(
    [
        ["D","C"],
        ["C","Z"]
    ]
)

PROBLEM_MATRIX_MAP["SDDMM"] = np.array(
    [
        ["D", "C", "Z"],
        ["D", "Z", "D"],
        ["Z", "D", "D"]
    ]
)

PROBLEM_MATRIX_MAP["TTV"] = np.array(
    [
        ["D", "C", "C"],
        ["Z", "Z", "D"]
    ]
)

PROBLEM_MATRIX_MAP["SPMM"] = np.array(
    [
        ["D", "C", "Z"],
        ["Z", "D", "D"]
    ]
)

PROBLEM_MATRIX_MAP["SPMV"] = np.array(
    [
        ["D", "C"],
        ["Z", "D"]
    ]
)

PROBLEM_MATRIX_MAP["MTTKRP"] = np.array(
    [
        ["D", "C","C","Z"],
        ["Z", "D","Z","D"],
        ["Z", "Z","D","D"]
    ]
)

PROBLEM_MATRIX_0 = PROBLEM_MATRIX_MAP[PROBLEM]

KNOWN_AXES_MAP = {}
KNOWN_AXES_MAP["SPMM"] = {"k": "128"}
KNOWN_AXES_MAP["SPMV"] = {}
KNOWN_AXES_MAP["MTTKRP"] = {"j": "32"}
KNOWN_AXES_MAP["TTV"] = {}
KNOWN_AXES_MAP["SDDMM"] = {"j":"128"}
KNOWN_AXES_MAP["SPMSPV"] = {}

KNOWN_AXES = KNOWN_AXES_MAP[PROBLEM]

# PROBLEM_MATRIX_0 = SPMM_PROBLEM_MATRIX_2

# number of tensors
T = PROBLEM_MATRIX_0.shape[0]
# number of index variables
N = PROBLEM_MATRIX_0.shape[1]

"""

what does the DNF mean? It's a disjunctive normal form. It contains a list of conjunctions, which represent * between tensors
so DNF=[[0,1]] means there is only one conjuction, between tensor 0 and tensor 1, likely corresponding to a multiplication.

"""
DNF_MAP = {}
DNF_MAP["SPMSPV"] = [[0, 1]]
DNF_MAP["SPMV"] = [[0, 1]]
DNF_MAP["SPMM"] = [[0, 1]]
DNF_MAP["MTTKRP"] = [[0,1,2]]
DNF_MAP["TTV"] = [[0,1]]
DNF_MAP["SDDMM"] = [[0,1,2]]
DNF = DNF_MAP[PROBLEM]

INDEX_VARIABLES = {}
INDEX_VARIABLES["SPMM"] = np.array(["i", "j", "k"])
INDEX_VARIABLES["SPMV"] = np.array(["i", "j"])
INDEX_VARIABLES["MTTKRP"] = np.array(["i", "k", "l", "j"])
INDEX_VARIABLES["TTV"] = np.array(["i","j","k"])
INDEX_VARIABLES["SDDMM"] = np.array(["i","k","j"])
INDEX_VARIABLES["SPMSPV"] = np.array(["j","i"])
INDEX_VARIABLES_0 = INDEX_VARIABLES[PROBLEM]


REDUCTION_AXES_MAP = {}
REDUCTION_AXES_MAP["SPMM"] = ["j"]
REDUCTION_AXES_MAP["SPMV"] = ["j"]
REDUCTION_AXES_MAP["MTTKRP"] = ["k","l"]
REDUCTION_AXES_MAP["TTV"] = ["k"]
REDUCTION_AXES_MAP["SDDMM"] = ["j"]
REDUCTION_AXES_MAP["SPMSPV"] = ["j"]
REDUCTION_AXES = REDUCTION_AXES_MAP[PROBLEM]


PRECEDENCES_MAP = {}
if HARDWARE == "CPU":
    PRECEDENCES_MAP["MTTKRP"] = { "k": ["i"]}
else:
    PRECEDENCES_MAP["MTTKRP"] = {"l":["i","k"],"k":["i"]}
PRECEDENCES_MAP["SPMM"] = {"j":["i"]}
PRECEDENCES_MAP["SPMV"] = {"j":["i"]}
PRECEDENCES_MAP["TTV"] = {"k":["i","j"],"j":["i"]}
PRECEDENCES_MAP["SDDMM"] = {"k":["i"]}
PRECEDENCES_MAP["SPMSPV"] = {"i":["j"]}
PRECEDENCES = PRECEDENCES_MAP[PROBLEM]

PREFERENCES_SPMM_CPU = [("j","k"),("i","k")]
PREFERENCES_SPMM_GPU = [("j","k"),("k","i")]
PREFERENCES_SPMV = []
PREFERENCES_MTTKRP = [("k","j"),("l","j")]

if HARDWARE == "GPU" and PROBLEM == "SPMM":
    PREFERENCES = PREFERENCES_SPMM_GPU
elif HARDWARE == "CPU" and PROBLEM == "SPMM":
    PREFERENCES = PREFERENCES_SPMM_CPU
elif PROBLEM == "SPMV":
    PREFERENCES = PREFERENCES_SPMV
elif PROBLEM == "MTTKRP":
    PREFERENCES = PREFERENCES_MTTKRP
elif PROBLEM == "TTV":
    PREFERENCES = []
elif PROBLEM == "SDDMM":
    PREFERENCES = [("i","j"),("k","j")]
elif PROBLEM == "SPMSPV":
    PREFERENCES = []

GPU_FACTORS = [4, 16, 64]
CPU_FACTORS = [8, 16, 32]

if HARDWARE == "CPU":
    FACTORS = CPU_FACTORS
else:
    FACTORS = GPU_FACTORS


"""
The following never change.

"""


LA_FUSABILITY = {"Z": True, "C": False, "PS": False, "F": True, "FPS": False, "FS": False}


STRATEGIES_0 = {
    "C": ["Z", "PS"],
    "D": ["Z", "C"]
}

STRATEGIES_1 = {
    ("C", True): ["Z", "F", "FPS","PS"],
    ("C", False): ["Z", "PS"],
    ("D", True): ["Z", "C", "F"],
    # we are not allowing FS! if last axis is dense, then FS only makes sense when both dense small
    # if last axis is compressed, why don't you split there (strided)
    ("D", False): ["Z", "C"]
}

# if the last axis was a fuse, then
STRATEGIES_2 = {
    ("C"): ["F", "FPS"],
    ("D"): ["F"],  # we are not allowing FS! if last axis is dense, then FS only makes sense when both dense small
    # if last axis is compressed, why don't you split there (strided)
}