from typing import Dict
import numpy as np

from ..data_model.parameters import Parameters


class GrayboxModel:
    def __init__(
        self,
        name: str = "GrayboxModelExample",
        module_name: str = "GrayboxModel",
        module_type: str = "non-linear",
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

        self.parameters = Parameters(self.name, self.module_name, self.module_type)
        self.init_param()

    def init_param(self):
        NotImplementedError

    def generate_init_params(self) -> Dict[str, float]:
        NotImplementedError

    def process_sample(self, input: float) -> float:
        NotImplementedError

    def process_block(self, input: np.ndarray) -> np.ndarray:
        NotImplementedError

    def process_block_with_param(
        self, input: np.ndarray, param: Parameters
    ) -> np.ndarray:
        NotImplementedError
