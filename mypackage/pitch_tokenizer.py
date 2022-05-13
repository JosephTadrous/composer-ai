import numpy as np

min_pitch = 21
max_pitch = 108
pitch_scale_len = max_pitch - min_pitch


def tokenize(pitches):
    note = np.zeros(max_pitch-min_pitch, dtype=int)
    for pitch in pitches:
        note[pitch - min_pitch] = 1
    return note


def tokenize_all(pitches_list):
    return [tokenize(pitches) for pitches in pitches_list]


def detokenize(token):
    return np.argwhere(token == 1).reshape(-1) + min_pitch


def detokenize_all(tokens):
    return [detokenize(token) for token in tokens]
