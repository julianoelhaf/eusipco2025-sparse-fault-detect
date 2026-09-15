# constants.py

# Constants representing the index positions of the data fields
FAULT = 0
FAULT_TARGET = 1
SC_LOCATION = 2
PHASE_SELECT = 3
SAMPLE_ID = 4

VOLTAGE_CHANNELS = [0, 1, 2]
CURRENT_CHANNELS = [3, 4, 5]

# FAULT LABEL VALUES
LABEL_NO_FAULT_CLASS = 0
LABEL_FAULT_CLASS = 1

# Dictionary to map the field names to their respective indices and data types
LABEL_MAPPINGS = {
    "FAULT": (FAULT, int),
    "FAULT_TARGET": (FAULT_TARGET, object),
    "SC_LOCATION": (SC_LOCATION, float),
    "PHASE_SELECT": (PHASE_SELECT, int),
    "SAMPLE_ID": (SAMPLE_ID, int),
}


EMT_LABELS_TO_UNIT = {
    ".0": "cur_L1_A",  # secondary current, phase L1
    ".1": "cur_L2_A",  # secondary current, phase L2
    ".2": "cur_L3_A",  # secondary current, phase L3
    ".3": "vol_L1_V",  # secondary current, phase L1
    ".4": "vol_L2_V",  # secondary current, phase L2
    ".5": "vol_L3_V",  # secondary current, phase L3
}

# Mapping of labels to units
RMS_LABELS_TO_UNIT = {
    ".0": "cur_L1_RE_A",  # secondary current, phase L1, real part, Ampere
    ".1": "cur_L1_IM_A",  # secondary current, phase L1, imaginary part, Ampere
    ".2": "cur_L2_RE_A",  # secondary current, phase L2, real part, Ampere
    ".3": "cur_L2_IM_A",  # secondary current, phase L2, imaginary part, Ampere
    ".4": "cur_L3_RE_A",  # secondary current, phase L3, real part, Ampere
    ".5": "cur_L3_IM_A",  # secondary current, phase L3, imaginary part, Ampere
    ".6": "vol_L1_RE_V",  # secondary voltage, phase L1, real part, Volt
    ".7": "vol_L1_IM_V",  # secondary voltage, phase L1, imaginary part, Volt
    ".8": "vol_L2_RE_V",  # secondary voltage, phase L2, real part, Volt
    ".9": "vol_L2_IM_V",  # secondary voltage, phase L2, imaginary part, Volt
    ".10": "vol_L3_RE_V",  # secondary voltage, phase L3, real part, Volt
    ".11": "vol_L3_IM_V",  # secondary voltage, phase L3, imaginary part, Volt
}


# CONSTANTS for labeling process
MIN_DELTA = 0.001  # Delta by which the t_evnt_start must be inside the window

BUS_TO_RELAY_MAPPING = {1: [0, 1], 2: [2, 3, 4, 5], 3: [6, 7]}
