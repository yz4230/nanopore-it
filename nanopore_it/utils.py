import numpy as np
import numpy.typing as npt
import scipy.signal

__all__ = ["load_opt_file", "fft", "downsample"]


def load_opt_file(
    data: bytes,
    *,
    lpf_cutoff: float,
    adc_samplerate: float,
    invert: bool = False,
    cut_dc: bool = False,
) -> npt.NDArray[np.float64]:
    raw = np.frombuffer(data, dtype=np.dtype(">d")).astype(np.float32)
    if invert:
        raw = -raw
    if cut_dc:
        raw = raw - np.mean(raw)
    wn = round(lpf_cutoff / (adc_samplerate / 2), 4)
    b, a = scipy.signal.bessel(4, wn, btype="low")
    filt = scipy.signal.filtfilt(b, a, raw)
    return filt


def fft(x: npt.NDArray[np.float64], *, fs: int) -> tuple[npt.NDArray, npt.NDArray]:
    N = len(x)
    window = np.hanning(N)
    xw = x * window
    X = np.fft.rfft(xw)
    freq = np.fft.rfftfreq(N, d=1 / fs)
    amp = 2 * np.abs(X) / np.sum(window)
    return freq, amp


def downsample(
    freq: npt.NDArray[np.float64],
    amp: npt.NDArray[np.float64],
    *,
    max_points: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    factor = max(1, len(freq) // max_points)
    if factor > 1:
        freq = freq[::factor]
        amp = amp[::factor]
    return freq, amp
