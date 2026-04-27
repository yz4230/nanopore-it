import numpy as np
import numpy.typing as npt
import scipy.signal

__all__ = ["load_opt_file"]


def load_opt_file(
    data: bytes,
    *,
    lpf_cutoff: float,
    adc_samplerate: float,
    invert: bool,
) -> npt.NDArray[np.float64]:
    raw = np.frombuffer(data, dtype=np.dtype(">d"))
    wn = round(lpf_cutoff / (adc_samplerate / 2), 4)
    b, a = scipy.signal.bessel(4, wn, btype="low")
    filt = scipy.signal.filtfilt(b, a, raw)
    if invert:
        filt = -filt
    return filt
