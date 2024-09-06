from os.path import isfile

SEED = 1143  # Random seed for reproducibility
sampling_rate = 16000  # Sampling rate for audio
frames_per_second = 25  # Frames per second for video
max_frames = 75  # Maximum number of frames per video for training
max_audio_len = sampling_rate * 3  # Maximum number of audio samples per video for training

DATA_ROOT = "./data" # Path to the avsec dataset
assert not isfile(DATA_ROOT), "Please set DATA_ROOT in config.py to the correct path to the avsec dataset"
