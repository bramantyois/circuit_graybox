from ..base import GrayboxModel


class Parametric(GrayboxModel):
    def __init__(self):
        super().__init__(name='ParametricExample', module_name='Parametric', module_type='non-linear')

    def init_param(self):
        return

    def generate_init_params(self):
        return {
            'a': self.parameters['a'].default,
            'b': self.parameters['b'].default
        }

    def process_sample(self, input):
        return self.parameters['a'].value * input + self.parameters['b'].value

    def process_block(self, input):
        return self.parameters['a'].value * input + self.parameters['b'].value

    def process_block_with_param(self, input, param):
        return param['a'] * input + param['b']