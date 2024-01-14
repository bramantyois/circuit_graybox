
from circuit_graybox.graybox_model.lti_model.lp import Lowpass
from audiodsp.utils import generate_sine_sweep, estimate_welch
import numpy as np


def test_filter_onepole():
    sr = 44100

    x = generate_sine_sweep(duration=0.5, sample_rate=sr, noise_std=0.2)

    lp = Lowpass(sample_rate=sr)
    
    lp.reset()
    
    lp_y = lp.process_block(x)
    
    f, pxx = estimate_welch(x, sample_rate=sr, throw_dc=True)
    lp_f, lp_pxx = estimate_welch(lp_y, sample_rate=sr, throw_dc=True)
    
    # Check that the power of the filtered signal is less than the original
    assert lp_pxx[-10:].mean() < pxx[-10:].mean()
