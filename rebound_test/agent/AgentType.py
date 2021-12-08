from enum import Enum

class AgentType(Enum):
    ANALYTICAL = "analytical"
    GCPD = "gcpd"
    NN = "nn"
    NN_NOP = "nn_nop"
    NN_GRAV = "nn_grav"
    NN_TESTING = "nn_testing"
