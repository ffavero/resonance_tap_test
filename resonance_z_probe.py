"""
RESONANCE Z PROBE

Copyright (C) 2024  Francesco Favero

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""

import os
import collections
import math
import numpy as np
from datetime import datetime

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager, ticker
import matplotlib.pyplot as plt
from textwrap import wrap


### FROM resonance tester


MIN_FREQ = 5.0
MAX_FREQ = 200.0
WINDOW_T_SEC = 0.1
MAX_SHAPER_FREQ = 150.0


TestPoint = collections.namedtuple(
    "TestPoint",
    (
        "time",
        "accel_x",
        "accel_y",
        "accel_z",
        "current_z",
    ),
)


class CalibrationData:
    def __init__(self, freq_bins, psd_sum, psd_x, psd_y, psd_z):
        self.freq_bins = freq_bins
        self.psd_sum = psd_sum
        self.psd_x = psd_x
        self.psd_y = psd_y
        self.psd_z = psd_z
        self._psd_list = [self.psd_sum, self.psd_x, self.psd_y, self.psd_z]
        self._psd_map = {
            "x": self.psd_x,
            "y": self.psd_y,
            "z": self.psd_z,
            "all": self.psd_sum,
        }
        self.data_sets = 1

    def add_data(self, other):
        joined_data_sets = self.data_sets + other.data_sets
        for psd, other_psd in zip(self._psd_list, other._psd_list):
            # `other` data may be defined at different frequency bins,
            # interpolating to fix that.
            other_normalized = other.data_sets * np.interp(
                self.freq_bins, other.freq_bins, other_psd
            )
            psd *= self.data_sets
            psd[:] = (psd + other_normalized) * (1.0 / joined_data_sets)
        self.data_sets = joined_data_sets

    def set_numpy(self, numpy):
        self.numpy = np

    def normalize_to_frequencies(self):
        for psd in self._psd_list:
            # Avoid division by zero errors
            psd /= self.freq_bins + 0.1
            # Remove low-frequency noise
            psd[self.freq_bins < MIN_FREQ] = 0.0

    def get_psd(self, axis="all"):
        return self._psd_map[axis]


def _split_into_windows(x, window_size, overlap):
    # Memory-efficient algorithm to split an input 'x' into a series
    # of overlapping windows
    step_between_windows = window_size - overlap
    n_windows = (x.shape[-1] - overlap) // step_between_windows
    shape = (window_size, n_windows)
    strides = (x.strides[-1], step_between_windows * x.strides[-1])
    return np.lib.stride_tricks.as_strided(
        x, shape=shape, strides=strides, writeable=False
    )


def _psd(x, fs, nfft):
    # Calculate power spectral density (PSD) using Welch's algorithm
    window = np.kaiser(nfft, 6.0)
    # Compensation for windowing loss
    scale = 1.0 / (window**2).sum()

    # Split into overlapping windows of size nfft
    overlap = nfft // 2
    x = _split_into_windows(x, nfft, overlap)

    # First detrend, then apply windowing function
    x = window[:, None] * x

    # Calculate frequency response for each window using FFT
    result = np.fft.rfft(x, n=nfft, axis=0)
    result = np.conjugate(result) * result
    result *= scale / fs
    # For one-sided FFT output the response must be doubled, except
    # the last point for unpaired Nyquist frequency (assuming even nfft)
    # and the 'DC' term (0 Hz)
    result[1:-1, :] *= 2.0

    # Welch's algorithm: average response over windows
    psd = result.real.mean(axis=-1)

    # Calculate the frequency bins
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs, psd


def calc_freq_response(raw_values):
    if raw_values is None:
        return None
    if isinstance(raw_values, np.ndarray):
        data = raw_values
    else:
        samples = raw_values.get_samples()
        if not samples:
            return None
        data = np.array(samples)

    N = data.shape[1]
    T = data[0, -1] - data[0, 0]
    SAMPLING_FREQ = N / T
    # Round up to the nearest power of 2 for faster FFT
    M = 1 << int(SAMPLING_FREQ * WINDOW_T_SEC - 1).bit_length()
    if N <= M:
        return None

    # Calculate PSD (power spectral density) of vibrations per
    # frequency bins (the same bins for X, Y, and Z)
    fx, px = _psd(data[1, :], SAMPLING_FREQ, M)
    fy, py = _psd(data[2, :], SAMPLING_FREQ, M)
    fz, pz = _psd(data[3, :], SAMPLING_FREQ, M)
    return CalibrationData(fx, px + py + pz, px, py, pz)


class ZVibrationHelper:
    """Helper to dynamically manage Z position and movement, including the vibration"""

    def __init__(self, printer, frequency, accel_per_hz) -> None:
        self.printer = printer
        self.gcode = self.printer.lookup_object("gcode")
        self.frequency = frequency
        self.accel_per_hz = accel_per_hz
        vib_dir = (0.0, 0.0, 1.0)
        s = math.sqrt(sum([d * d for d in vib_dir]))
        self._vib_dir = [d / s for d in vib_dir]
        self.input_shaper_was_on = False
        self.input_shaper = None

    def _set_vibration_variables(self):
        """Calculate the axis coordinate difference to perform the vibration movement"""
        t_seg = 0.25 / self.frequency
        accel = self.accel_per_hz * self.frequency
        self.max_v = accel * t_seg
        toolhead = self.printer.lookup_object("toolhead")
        self.cur_x, self.cur_y, self.cur_z, self.cur_e = toolhead.get_position()
        toolhead.cmd_M204(self.gcode.create_gcode_command("M204", "M204", {"S": accel}))
        self.movement_span = 0.5 * accel * t_seg**2
        self.dX, self.dY, self.dZ = self.get_point(self.movement_span)

    def get_point(self, l):
        return (self._vib_dir[0] * l, self._vib_dir[1] * l, self._vib_dir[2] * l)

    def _vibrate_(self):
        toolhead = self.printer.lookup_object("toolhead")
        for sign in [1, -1]:
            nX = self.cur_x + sign * self.dX
            nY = self.cur_y + sign * self.dY
            nZ = self.cur_z + sign * self.dZ
            toolhead.move([nX, nY, nZ, self.cur_e], self.max_v)
            toolhead.move([self.cur_x, self.cur_y, self.cur_z, self.cur_e], self.max_v)

    def disable_input_shaper(self):
        self.input_shaper = self.printer.lookup_object("input_shaper", None)
        if self.input_shaper is not None:
            self.input_shaper.disable_shaping()
            self.input_shaper_was_on = True

    def restore_input_shaper(self):
        if self.input_shaper_was_on:
            self.input_shaper.enable_shaping()

    def vibrate_n(self, n):
        self._set_vibration_variables()
        counter = 0
        while counter <= n:
            self._vibrate_()
            counter += 1


class TapResonanceData:
    def __init__(self, test_points, out_path, gcmd):
        self.data = test_points
        self.gcmd = gcmd
        self.ts = datetime.timestamp(datetime.now())
        self.pdf_out = os.path.join(out_path, "tap_summary_%s.pdf" % self.ts)
        self.csv_out = os.path.join(out_path, "tap_summary_%s.csv" % self.ts)

    def _n_test(self):
        return len(self.data)

    def _rate_above_threshold(self, threshold):
        rates = []
        z_height = []
        for t, x, y, z, curr_z in self.data:

            rate_above_tr = sum(
                np.logical_or(z > threshold, z < (-1 * threshold))
            ) / len(t)
            rates.append(rate_above_tr)
            z_height.append(curr_z)
        return (z_height, rates)

    def plot(self, threshold, cycles):
        rates_above_tr = self._rate_above_threshold(threshold)
        self.gcmd.respond_info("writing debug plots to %s" % self.pdf_out)

        with PdfPages(self.pdf_out) as pdf:

            plt.plot(
                rates_above_tr[0],
                rates_above_tr[1],
                linestyle="-",
                marker="o",
            )
            fontP = font_manager.FontProperties()
            fontP.set_size("x-small")
            plt.xlabel("Z height")
            plt.ylabel("Rate of points above threshold")
            pdf.savefig(facecolor="white")
            plt.close()
            for z_test in rates_above_tr[0]:
                acc_plot = self.plot_accel(z_test, cycles, threshold)
                pdf.savefig(acc_plot, facecolor="white")
                plt.close()
                freq_plot = self.plot_frequency(z_test, 200)
                pdf.savefig(freq_plot, facecolor="white")
                plt.close()

    def write_data(self):
        self.gcmd.respond_info("writing data to %s" % self.csv_out)
        with open(self.csv_out, "wt") as data_out:
            data_out.write("#time,accel_x,accel_y,accel_z,z_height\n")
            for t, x, y, z, curr_z in self.data:
                for i in range(len(t)):
                    data_out.write(
                        "%.6f,%.6f,%.6f,%.6f,%.6f\n" % (t[i], x[i], y[i], z[i], curr_z)
                    )

    def plot_accel(self, z_height, cycles, threshold):
        data = None
        for t, x, y, z, curr_z in self.data:
            if curr_z == z_height:
                data = np.array((t, x, y, z))
        if data is None:
            self.gcmd.respond_info("No corresponding z_height found")
            return None
        logname = "%.4f" % z_height
        fig, axes = plt.subplots(nrows=3, sharex=True)
        axes[0].set_title("\n".join(wrap("Accelerometer data z=%s" % logname, 15)))
        axis_names = ["Expected taps", "z-accel", "both"]
        first_time = data[0, 0]
        times = data[0, :] - first_time
        time_span = times[-1]
        expect_freq = cycles / time_span
        sin_wave = np.sin(2 * np.pi * expect_freq * times)
        sin_wave[sin_wave < 0] = 0
        cos_wave = np.cos(2 * np.pi * expect_freq * times)
        cos_wave[cos_wave < 0] = 0
        ax = axes[0]
        ax.plot(times, np.flip(sin_wave), alpha=0.8, label="Expected taps, sin flip")
        ax.plot(times, cos_wave, alpha=0.8, label="Expected taps, cos", color="red")
        # times = data[:, 0]
        adata = data[3, :]
        ax = axes[1]
        label = "\n".join(wrap(logname, 60))
        ax.plot(times, adata, alpha=0.8, label="z")
        ax.axhline(y=threshold, linestyle="--", lw=2, label="threshold", color="red")
        ax.axhline(y=-1 * threshold, linestyle="--", lw=2, color="red")
        ax = axes[2]
        ax.plot(times, adata * np.flip(sin_wave), alpha=0.8, label="normalized")
        ax.axhline(y=threshold, linestyle="--", lw=2, label="threshold", color="red")
        ax.axhline(y=-1 * threshold, linestyle="--", lw=2, color="red")
        axes[-1].set_xlabel("Time (s)")
        fontP = font_manager.FontProperties()
        fontP.set_size("x-small")
        for i in range(len(axis_names)):
            ax = axes[i]
            ax.grid(True)
            ax.legend(loc="best", prop=fontP)
            ax.set_ylabel("%s" % (axis_names[i],))
        fig.tight_layout()
        return fig

    def plot_frequency(self, z_height, max_freq):
        calibration_data = None
        for t, x, y, z, curr_z in self.data:
            if curr_z == z_height:
                calibration_data = calc_freq_response(np.array((t, x, y, z)))
        if calibration_data is None:
            self.gcmd.respond_info("No corresponding z_height found")
            return None
        freqs = calibration_data.freq_bins
        psd = calibration_data.psd_sum[freqs <= max_freq]
        px = calibration_data.psd_x[freqs <= max_freq]
        py = calibration_data.psd_y[freqs <= max_freq]
        pz = calibration_data.psd_z[freqs <= max_freq]
        freqs = freqs[freqs <= max_freq]

        fig, ax = plt.subplots()
        ax.set_title("\n".join(wrap("Frequency response (%s)" % str(z_height), 15)))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power spectral density")

        ax.plot(freqs, psd, label="X+Y+Z", alpha=0.6)
        ax.plot(freqs, px, label="X", alpha=0.6)
        ax.plot(freqs, py, label="Y", alpha=0.6)
        ax.plot(freqs, pz, label="Z", alpha=0.6)

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(which="major", color="grey")
        ax.grid(which="minor", color="lightgrey")
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

        fontP = font_manager.FontProperties()
        fontP.set_size("x-small")
        ax.legend(loc="best", prop=fontP)
        fig.tight_layout()
        return fig


class OffsetHelper:
    """
    Wraps the decision making into the next z position to test, and mark when
    a z offset was detected successfully
    """

    def __init__(self, pos, step, min_precision=0.005) -> None:

        self.last_tested_pos = None
        self.step = step
        self.last_tested_pos = (pos, False)
        self.min_precision = min_precision
        self.current_offset = None
        self.finished = False
        self.started = False

    def next_position(self):
        if self.finished:
            return self.current_offset
        else:
            if self.started is False:
                return self.last_tested_pos[0]
            else:
                pos, status = self.last_tested_pos
                if status == True:
                    self.current_offset = pos
                    if self.step <= self.min_precision:
                        self.finished = True
                    return pos + self.step
                else:
                    return pos - self.step

    def last_tested_position(self, position, triggered):
        self.last_tested_pos = (position, triggered)
        self.started = True
        if triggered:
            self.step = self.step / 2.0


class ResonanceZProbe:
    def __init__(self, config):
        self.config = config
        self.printer = self.config.get_printer()
        self.gcode = self.printer.lookup_object("gcode")
        # consider that accel_per_hz * freq might be caped to the printer max accel
        self.accel_per_hz = self.config.getfloat("accel_per_hz", 1.5, above=0.0)

        self.step_size = self.config.getfloat("step_size", 0.01, minval=0.005)
        self.tolerance = config.getfloat("samples_tolerance", None, above=0.0)
        self.z_freq = self.config.getfloat(
            "z_vibration_freq", 80, minval=50.0, maxval=200.0
        )
        self.amp_threshold = self.config.getfloat(
            "amplitude_threshold", 700.0, above=500.0
        )
        self.rate_above_threshold = self.config.getfloat(
            "rate_above_threshold", 0.015, minval=0.0, maxval=1.0
        )
        self.safe_min_z = self.config.getfloat("safe_min_z", 1)
        self.probe_points = self.config.getfloatlist("probe_points", sep=",", count=3)

        self.cycle_per_test = self.config.getint(
            "cycle_per_test", 50, minval=2, maxval=500
        )
        self.accel_chip_name = self.config.get("accel_chip").strip()
        self.gcode.register_command(
            "CALIBRATE_Z_RESONANCE",
            self.cmd_CALIBRATE_Z_RESONANCE,
            desc=self.cmd_CALIBRATE_Z_RESONANCE_help,
        )
        self.printer.register_event_handler("klippy:connect", self.connect)
        self.vibration_helper = ZVibrationHelper(
            self.printer, self.z_freq, self.accel_per_hz
        )
        self.vibration_helper.disable_input_shaper()
        self.debug = 0
        self.dump = 0
        self.data_points = []

    def connect(self):
        self.accel_chips = ("z", self.printer.lookup_object(self.accel_chip_name))

    cmd_CALIBRATE_Z_RESONANCE_help = "Calibrate Z making the bed vibrate while probing with the nozzle and record accelerometer data"

    def cmd_CALIBRATE_Z_RESONANCE(self, gcmd):
        self.babystep_probe(gcmd)

    def babystep_probe(self, gcmd):
        """
        move to the test position, start recording acc while resonate z, measure the amplitude and compare with the amp_threshold.
        lower by babystep until min safe Z is reached or the amp_threshold is passed.
        log stuff in the console
        """
        self.debug = gcmd.get_int("DEBUG", 0, minval=0, maxval=1)
        self.dump = gcmd.get_int("DUMP", 0, minval=0, maxval=1)
        self.safe_min_z = gcmd.get_float("SAFE_Z", self.safe_min_z)
        x_pos = gcmd.get_float("X_POS", self.probe_points[0])
        y_pos = gcmd.get_float("Y_POS", self.probe_points[1])
        z_pos = gcmd.get_float("Z_POS", self.probe_points[2], minval=self.safe_min_z)
        self.probe_points = (x_pos, y_pos, z_pos)
        test_results = []
        self.offset_helper = OffsetHelper(
            self.probe_points[2], self.step_size, self.tolerance
        )
        toolhead = self.printer.lookup_object("toolhead")
        toolhead.manual_move(self.probe_points, 50.0)
        toolhead.wait_moves()
        toolhead.dwell(0.500)
        self.vibration_helper._set_vibration_variables()
        curr_z = self.probe_points[2]
        while curr_z >= self.safe_min_z:
            if self.offset_helper.finished:
                break
            results, z_psd_is_max = self._test(gcmd, curr_z)
            self.offset_helper.last_tested_position(
                curr_z, z_psd_is_max and results >= self.rate_above_threshold
            )
            test_results.append((self.probe_points[2], results))
            self.vibration_helper.cur_z = self.offset_helper.next_position()
            curr_z = self.vibration_helper.cur_z

        # for res in test_results:
        #     gcmd.respond_info("Z:%.4f percentage outside threshold %.2f%%" % res)
        if self.offset_helper.finished:
            gcmd.respond_info(
                "probe at %.4f,%.4f  is z=%.6f"
                % (
                    self.probe_points[0],
                    self.probe_points[1],
                    self.offset_helper.current_offset,
                )
            )

        if self.dump == 1:
            tap_data = TapResonanceData(self.data_points, "/tmp", gcmd)
            tap_data.write_data()
            tap_data.plot(self.amp_threshold, self.cycle_per_test)
            self.data_points = []

    def _test(self, gcmd, curr_z):

        chip = self.accel_chips[1]
        aclient = chip.start_internal_client()
        aclient.msg = []
        aclient.samples = []
        cycle_counter = 0
        while cycle_counter <= self.cycle_per_test:
            self.vibration_helper._vibrate_()
            cycle_counter += 1
        timestamps = []
        x_data = []
        y_data = []
        z_data = []
        aclient.finish_measurements()
        for t, accel_x, accel_y, accel_z in aclient.get_samples():
            timestamps.append(t)
            x_data.append(accel_x)
            y_data.append(accel_y)
            z_data.append(accel_z)

        x = np.asarray(x_data)
        y = np.asarray(y_data)
        z = np.asarray(z_data)
        x = x - np.median(x)
        y = y - np.median(y)
        z = z - np.median(z)
        calibration_data = calc_freq_response(np.array((
            timestamps, x, y, z)))
        freqs = calibration_data.freq_bins
        psd_sums = (
            np.sum(calibration_data.psd_x[freqs >= 80]),
            np.sum(calibration_data.psd_y[freqs >= 80]),
            np.sum(calibration_data.psd_z[freqs >= 80]))
        z_psd_is_max = np.argmax(psd_sums) == 2
        try:
            rate_above_tr = sum(
                np.logical_or(z > self.amp_threshold, z < (-1 * self.amp_threshold))
            ) / len(timestamps)
        except ZeroDivisionError:
            rate_above_tr = 0

        if self.debug == 1:
            if len(timestamps) > 0:
                test_time = timestamps[len(timestamps) - 1] - timestamps[0]
            else:
                test_time = 0
            gcmd.respond_info(
                "Testing Z: %.4f. Received %i samples in %.2f seconds. Percentage above threshold: %.1f%%"
                % (curr_z, len(timestamps), test_time, 100 * rate_above_tr)
            )
            gcmd.respond_info(
                "psd sums x: %.3f y: %.3f z: %.3f" % psd_sums)
        if self.dump:
            self.data_points.append(TestPoint(timestamps, x, y, z, curr_z))
        return (rate_above_tr, z_psd_is_max)


def load_config(config):
    return ResonanceZProbe(config)
