import numpy as np

from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error

from ..graybox_model.base import LTIModel
from ..data_model import Parameters

from nkss.NKSS import NKSS

from audiodsp.utils import estimate_welch


class LTISolver:
    def __init__(
        self,
        ss_model: NKSS,
        lti_model: LTIModel,
    ) -> None:
        self.lti_model = lti_model
        self.ss_model = ss_model

        self.use_log_scale = use_log_scale
        self.fft_size = fft_size

    def func(self, params_arr: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        y_bar = self.lti_model.process_block_with_param(x, params_arr)
        f, pxx = estimate_welch(
            x=y_bar,
            fft_size=self.fft_size,
            sample_rate=self.lti_model.sample_rate,
            log_freq=self.use_log_scale,
        )
        return y - pxx

    def fit(self, x: np.ndarray, y: np.ndarray):
        init_params = self.lti_model.get_param_array()
        res_lsq = least_squares(self.func, init_params, args=(y, x))
        return res_lsq

    def find_best_param(
        self,
        noise_std: float = 0.1,
        num_trials: int = 10,
    ) -> np.ndarray:
        params = []
        errors = []
        for _ in range(num_trials):
            x = np.random.normal(loc=0.0, scale=noise_std, size=self.lti_model.sample_rate)
            y = self.ss_model.process_block(x).ravel()
            _, y_pxx = estimate_welch(
                x=y,
                fft_size=self.fft_size,
                sample_rate=self.lti_model.sample_rate,
                log_freq=self.use_log_scale,
            )
            new_param = self.fit(x, y_pxx).x
            y_pred = self.lti_model.process_block_with_param(x, new_param)
            error = mean_squared_error(y, y_pred)
            
            params.append(new_param)
            errors.append(error)
        
        best_param = params[np.argmin(errors)]
        return best_param