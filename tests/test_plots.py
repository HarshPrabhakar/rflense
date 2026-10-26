import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from rflens.dsp.plots import plot_psd, plot_spectrogram
from rflens.dsp.psd import welch_psd
from rflens.types import IQRecording


def test_psd_and_spectrogram_plots_return_axes():
    rec = IQRecording(np.exp(2j * np.pi * 0.1 * np.arange(2048)), 1000.0, 5e6)
    ax1 = plot_psd(welch_psd(rec, nperseg=256))
    ax2 = plot_spectrogram(rec, nperseg=256)
    assert ax1.get_xlabel() == "Frequency (Hz)"
    assert ax2.get_ylabel() == "Frequency (Hz)"
    plt.close("all")
