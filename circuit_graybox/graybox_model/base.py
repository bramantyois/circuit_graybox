from typing import Dict, Literal
import numpy as np

from ..data_model.parameters import Parameters


class GrayboxBaseModel:
    def __init__(
        self,
        name: str = "GrayboxModelExample",
        module_name: str = "GrayboxModel",
        module_type: Literal["lti", "non-linear"] = "non-linear",
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
            sample_rate=self.sample_rate,
        )
        self.init_param()

    def init_param(self):
        NotImplementedError

    def reset(self):
        NotImplementedError

    def set_param(self, param: Parameters) -> None: 
        self.parameters = param
        self.reset()
        
    def set_param_array(self, param_arr: np.ndarray) -> None:
        new_param_dict = {}
        for i, key in enumerate(self.parameters.values.keys()):
            new_param_dict[key] = param_arr[i]
        self.parameters.values = new_param_dict
        self.reset()

    def get_param(self) -> Parameters:
        return self.parameters

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
        self, input: np.ndarray, 
        param_arr: np.ndarray
    ) -> np.ndarray:
        self.set_param_array(param_arr)
        
        return self.process_block(input)


class LTIModel(GrayboxBaseModel):
    def __init__(
        self,
        name: str = "GrayboxModelExample",
        module_name: str = "GrayboxModel",
        sample_rate: int = 44100,
    ) -> None:
        """
        Graybox model base class
        :param name: model name
        :param module_name: module name
        :param module_type: module type. Choices: 'lti' (linear time invariant) for filters, 'non-linear' for non-linear modelling
        """
        super().__init__(name, module_name, module_type="lti", sample_rate=sample_rate)


class NLModel(GrayboxBaseModel):
    def __init__(
        self,
        name: str = "GrayboxModelExample",
        module_name: str = "GrayboxModel",
        sample_rate: int = 44100,
    ) -> None:
        """
        Graybox model base class
        :param name: model name
        :param module_name: module name
        :param module_type: module type. Choices: 'lti' (linear time invariant) for filters, 'non-linear' for non-linear modelling
        """
        super().__init__(name, module_name, module_type="non-linear", sample_rate=sample_rate)
