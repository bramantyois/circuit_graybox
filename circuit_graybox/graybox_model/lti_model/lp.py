import math
import numpy as np

from audiodsp.filter.onepole import OnePoleLP

from ..base import LTIModel


class Lowpass(LTIModel):
    def __init__(
        self,
        name: str = "GrayboxModelExample",
        module_name: str = "GrayboxModel",
        module_type: str = "non-linear",
        sample_rate: int = 44100,
    ) -> None:
        super().__init__(name, module_name, module_type, sample_rate)

        self.init_param()
        self.reset()

    def init_param(self):
        params = np.zeros(2).astype(float)
        rand = np.random.random(2)

        params[0] = 0.5 + 0.5 * rand[0]
        params[1] =  (0.5 + rand[1]) * 10000

        param_dict = {
            "bias-gain": params[0],
            "cut-off": params[1],
        }

        self.parameters.values = param_dict

    def reset(self):
        self.low_pass = OnePoleLP(
            fc=self.parameters.values["cut-off"],
            sample_rate=self.parameters.sample_rate,
            name="LowPass" + self.name,
        )

    def process_sample(self, input: float) -> float:
        input = self.low_pass.process_sample(input)
        input = input * self.parameters.values["bias-gain"]
        
        return input
