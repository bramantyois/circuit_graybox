from typing import Dict
import numpy as np

from ..data_model.parameters import Parameters


class GrayboxModel:
    def __init__(
        self,
        name: str = "GrayboxModelExample",
        module_name: str = "GrayboxModel",
        module_type: str = "non-linear",
        sample_rate: int = 44100,
    ) -> None:
        """
        Graybox model base class
        :param name: model name
        :param module_name: module name
        :param module_type: module type. Choices: 'lti' (linear time invariant) for filters, 'non-linear' for non-linear modelling
        """
        self.name = name
        self.module_name = module_name
        self.module_type = module_type
        self.sample_rate = sample_rate

        self.parameters = Parameters(
            name=self.name,
            module_name=self.module_name, 
            module_type=self.module_type, 
            sample_rate=self.sample_rate
        )
        self.init_param()

    def init_param(self):
        NotImplementedError

    def reset_buffer(self):
        NotImplementedError

    def get_param(self) -> Dict[str, float]:
        return self.parameters.values

    def get_param_array(self) -> np.ndarray:
        return np.array(list(self.parameters.values.values()))

    def process_sample(self, input: float) -> float:
        NotImplementedError

    def process_block(self, input: np.ndarray) -> np.ndarray:
        output = np.zeros_like(input)
        for i in range(len(input)):
            output[i] = self.process_sample(input[i])
        return output

    def process_block_with_param(
        self, input: np.ndarray, param: Parameters
    ) -> np.ndarray:
        self.parameters = param
        self.reset_buffer()
        
        return self.process_block(input)
    
