from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 0
_C.SYSTEM.NUM_WORKERS = 2
_C.SYSTEM.HOSTNAMES = []

_C.DATASET = CN()
_C.DATASET.NAME = "rplan"
_C.DATASET.SUBSET = ""
_C.DATASET.BATCHSIZE = 8

_C.MODEL = CN()
_C.MODEL.GENERATOR = CN()
_C.MODEL.GENERATOR.DECAY_EPOCHS = 40
_C.MODEL.GENERATOR.LEARNING_RATE = 1e-4
_C.MODEL.RENDERER = CN()
_C.MODEL.RENDERER.RENDERING_SIZE = 64
_C.MODEL.DISCRIMINATOR = CN()
_C.MODEL.DISCRIMINATOR.NUM_RESBLOCKS = 3
_C.MODEL.DISCRIMINATOR.DECAY_EPOCHS = 20
_C.MODEL.DISCRIMINATOR.LEARNING_RATE = 5e-5

_C.TRAIN = CN()
_C.TRAIN.NUM_EPOCHS = 3000

_C.TENSORBOARD = CN()
_C.TENSORBOARD.SAVE_INTERVAL_EPOCHS = 1
_C.TENSORBOARD.ANNOTATION = ""

_C.PATH = CN()
_C.PATH.RPLAN = []
_C.PATH.Z_FILE = "fixed_z/fixed_xyaw_rplan_0320.pkl"
_C.PATH.LOG_DIR = "runs_rplan"

_C.MANUAL = CN()
_C.MANUAL.RPLAN_PATH = "/content/FloorplanGANForCollab/data_FloorplanGAN/pkls"


def get_cfg():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
