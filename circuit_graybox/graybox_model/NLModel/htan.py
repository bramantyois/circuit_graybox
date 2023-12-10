from ..graybox_model import GrayboxModel


class HypTan(GrayboxModel):
    def __init__(
        self,
        name: str = "HyperbolicTangentExample",
        module_name: str = "HyperbolicTangentExample",
        module_type: str = "non-linear",
    ) -> None:
        super().__init__(name, module_name, module_type)

    def init_param(self):
        param = {
            "pre-bias gain": 
        }
