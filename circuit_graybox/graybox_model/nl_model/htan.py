import math
import numpy as np

from audiodsp.filter.onepole import OnePoleLP
from ..base import NLModel


class HypTan(NLModel):
    def __init__(
        self,
        name: str = "HyperbolicTangentExample",
        module_name: str = "HyperbolicTangentExample",
        module_type: str = "non-linear",
        sample_rate: int = 44100,
    ) -> None:
        super().__init__(name, module_name, module_type, sample_rate)

        self.low_pass = OnePoleLP(
            fc=10, sample_rate=self.parameters.sample_rate, name="LowPass" + self.name
        )

    def init_param(self):
        params = np.array([1, 1, 1, 0]).astype(float)
        params += np.random.normal(0, 0.1, 4)
        param_dict = {
            "bias-gain": params[0],
            "shaper-gain": params[1],
            "wet-gain": params[2],
            "out-bias-dc": params[3],
        }

        self.parameters.values = param_dict

    def reset(self):
        self.low_pass.reset_buffer()

    def process_sample(self, input: float) -> float:
        bias = input

        # bias path
        bias = np.abs(bias)
        bias = self.low_pass.process_sample(bias)
        bias = bias * self.parameters.values["bias-gain"]

        # wet path
        wet = input - bias

        shaped = np.tanh(self.parameters.values["shaper-gain"] * wet) / np.tanh(
            self.parameters.values["shaper-gain"]
        )

        wet = (
            shaped * self.parameters.values["wet-gain"]
            + self.parameters.values["out-bias-dc"]
        )

        return wet
