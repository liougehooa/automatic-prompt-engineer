import random


def subsample_data(data, subsample_size):
    """
    Subsample data. Data is in the form of a tuple of lists.
    """
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), subsample_size)
    inputs = [inputs[i] for i in indices]
    outputs = [outputs[i] for i in indices]
    return inputs, outputs

def subsample_single_data(inputs, subsample_size):
    """
    Subsample data. Data is in the form of a tuple of lists.
    """
    indices = random.sample(range(len(inputs)), subsample_size)
    inputs = [inputs[i] for i in indices]
    return inputs

