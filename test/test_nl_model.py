import numpy as np

from circuit_graybox.graybox_model.NLModel.htan import HypTan


def test_nl_model():
    sr = 44100
    duration = 1
    
    # Generate sine sweep
    input = np.sin(2 * np.pi * np.arange(sr * duration) / sr)
    
    # Instantiate model
    model = HypTan()
    
    # Process block
    output = model.process_block(input)
    
    # compute power
    input_power = np.sum(input ** 2)
    output_power = np.sum(output ** 2)
    
    # compute power ratio
    power_ratio = output_power / input_power
    
    # assert power ratio is less than 1
    assert power_ratio != 0
    
    # correlate
    corr = np.correlate(input, output, mode="valid")
    
    # assert correlation is not 0
    assert corr != 0
