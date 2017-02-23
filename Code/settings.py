import os
ROOT_DIR = os.getcwd()+'/'
DATA_DIR = ROOT_DIR + 'Data/'
DATA_DIR_TRAIN = DATA_DIR + 'train/'
DATA_DIR_TEST = DATA_DIR + 'test/'
FEATURE_NAMES_FILE = 'featnames.csv'
SAMPLE_SUBMISSION_CSV = DATA_DIR + 'sampleSubmission.csv'
PREDICT_CSV = DATA_DIR + 'train.csv'
STATION_INFO_CSV = ROOT_DIR + 'Data/station_info.csv'
NORMCOEFFS_FILE = 'norm.pickle'
NUM_STATIONS = 98
OUTDIR = None  # Overwrite when run train_solar_predict.py from command line.
PLOTRAWDATA = True
# Machine learning parameters
TRAIN_FRAC = 0.80

# XGBoost settings 
BOOSTER = "gbtree"
OBJECTIVE = "reg:logistic"
EVAL_METRIC = "mae"
N_STEPS = 400
EARLY_STOP = 50
ETA = [0.1]
ALPHA = [1.0]
LAMBDA = [1.0]
GAMMA = [1.0]
COLSAMPLE_BYTREE = [0.8]
COLSAMPLE_BYLEVEL = [1.0]
MAX_DELTA_STEP = [0.0]
MIN_CHILD_WEIGHT = [2.0]
SCALE_POS_WEIGHT = [1.0]
SUBSAMPLE = [0.8]
MAX_DEPTH = [25]