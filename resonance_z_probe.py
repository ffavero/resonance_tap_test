"""


example config

[resonance_z_probe]
accel_chip: adxl345
babystep: 0.01
z_vibration_freq: 180
amplitude_threshold: 700
probe_points: 125, 125, 5
safe_min_z: 0.05
test_time: 3

"""

import os
import collections
import math
import numpy as np
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


class TestZ:
    def __init__(self):
        self._name = "axis=%.3f,%.3f,%.3f" % (0.0, 0.0, 1)
        vib_dir = (0.0, 0.0, 1.0)
        s = math.sqrt(sum([d * d for d in vib_dir]))
        self._vib_dir = [d / s for d in vib_dir]

    def matches(self, chip_axis):
        if self._vib_dir[0] and "x" in chip_axis:
            return True
        if self._vib_dir[1] and "y" in chip_axis:
            return True
        if self._vib_dir[2] and "z" in chip_axis:
            return True
        return False

    def get_name(self):
        return self._name

    def get_point(self, l):
        return (self._vib_dir[0] * l, self._vib_dir[1] * l, self._vib_dir[2] * l)


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


class TapResonanceData:
    def __init__(self, test_points, out_path):
        self.data = test_points
        self.ts = datetime.timestamp(datetime.now())
        self.pdf_out = os.path.join(out_path, "tap_summary_%s.pdf" % self.ts)
        self.csv_out = os.path.join(out_path, "tap_summary_%s.csv" % self.ts)

    def _n_test(self):
        return len(self.data)

    def _rate_above_threshold(self, threshold):
        rates = []
        z_height = []
        for t, x, y, z, curr_z in self.data:
            rates.append(sum(z > threshold) / z.size)
        z_height.append(curr_z)
        return (z_height, rates)

    def plot(self):
        rates_above_tr = self._rate_above_threshold()
        pass

    def write_data(self):
        with open(self.csv_out, "wt") as data_out:
            data_out.write("#time,accel_x,accel_y,accel_z,z_height\n")
            for t, x, y, z, curr_z in self.data:
                for i in range(len(t)):
                    data_out.write(
                        "%.6f,%.6f,%.6f,%.6f,%.6f\n" % (t[i], x[i], y[i], z[i], curr_z)
                    )


class ResonanceZProbe:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object("gcode")
        # consider that accel_per_hz * freq might be caped to the printer max accel
        self.accel_per_hz = config.getfloat("accel_per_hz", 1.5, above=0.0)

        self.step_size = config.getfloat("babystep", 0.01, minval=0.005)
        self.z_freq = config.getfloat("z_vibration_freq", 80, minval=50.0, maxval=200.0)
        self.amp_threshold = config.getfloat("amplitude_threshold", 700.0, above=500.0)
        self.safe_min_z = config.getfloat("safe_min_z", 1)

        self.test_time = config.getint("test_time", 3.0, minval=1, maxval=10)
        self.probe_points = config.getfloatlist("probe_points", sep=",", count=3)
        self.accel_chip_name = config.get("accel_chip").strip()
        self.gcode.register_command(
            "CALIBRATE_Z_RESONANCE",
            self.cmd_CALIBRATE_Z_RESONANCE,
            desc=self.cmd_CALIBRATE_Z_RESONANCE_help,
        )
        self.printer.register_event_handler("klippy:connect", self.connect)

    def connect(self):
        self.accel_chips = ("z", self.printer.lookup_object(self.accel_chip_name))

    def hold_z_freq(self, gcmd, seconds, freq):
        """holds a resonance for N seconds
        resonance code taken from klipper's test_resonances command"""
        axis = TestZ()
        toolhead = self.printer.lookup_object("toolhead")
        X, Y, Z, E = toolhead.get_position()
        sign = 1.0
        input_shaper = self.printer.lookup_object("input_shaper", None)
        if input_shaper is not None and not gcmd.get_int("INPUT_SHAPING", 0):
            input_shaper.disable_shaping()
            gcmd.respond_info("Disabled [input_shaper] for resonance holding")
        else:
            input_shaper = None
        gcmd.respond_info("starting freq %i for %i seconds" % (freq, seconds))
        t_seg = 0.25 / freq
        max_moves = seconds / t_seg / 2
        count_moves = 0
        while count_moves <= max_moves:
            accel = self.accel_per_hz * freq
            max_v = accel * t_seg
            toolhead.cmd_M204(
                self.gcode.create_gcode_command("M204", "M204", {"S": accel})
            )
            L = 0.5 * accel * t_seg**2
            dX, dY, dZ = axis.get_point(L)
            nX = X + sign * dX
            nY = Y + sign * dY
            nZ = Z + sign * dZ
            toolhead.move([nX, nY, nZ, E], max_v)
            toolhead.move([X, Y, Z, E], max_v)
            sign = -sign
            count_moves += 1
        gcmd.respond_info("DONE")

    def _test(self, gcmd):
        toolhead = self.printer.lookup_object("toolhead")

        toolhead.manual_move(self.probe_points, 50.0)
        toolhead.wait_moves()
        toolhead.dwell(0.500)
        chip_axis, chip = self.accel_chips
        aclient = chip.start_internal_client()
        aclient.msg = []
        aclient.samples = []
        self.hold_z_freq(gcmd, self.test_time, self.z_freq)
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
        gcmd.respond_info(
            "Received %i samples, median values x: %i y: %i z: %i"
            % (len(timestamps), np.median(x), np.median(y), np.median(z))
        )
        x = x - np.median(x)
        y = y - np.median(y)
        z = z - np.median(z)
        rate_above_tr = sum(z > self.amp_threshold) / len(timestamps)
        fourier_series = np.log10(np.fft.fft(z).real)
        gcmd.respond_info(
            "normalize %i samples, median values x: %i y: %i z: %i"
            % (len(timestamps), np.median(x), np.median(y), np.median(z))
        )
        gcmd.respond_info(
            "The test lasted %s seconds"
            % (timestamps[len(timestamps) - 1] - timestamps[0])
        )
        gcmd.respond_info("Max amplitude: %s" % np.nanmax(fourier_series))
        gcmd.respond_info("Rate above threshold: %s" % rate_above_tr)
        return TestPoint(timestamps, x, y, z, self.probe_points[2])

    def babystep_probe(self, gcmd):
        """
        move to the test position, start recording acc while resonate z, measure the amplitude and compare with the amp_threshold.
        lower by babystep until min safe Z is reached or the amp_threshold is passed.
        log stuff in the console
        """
        # move thself.probe_pointe toolhead
        data_points = []
        while self.probe_points[2] >= self.safe_min_z:
            gcmd.respond_info("Test Z: %s" % self.probe_points[2])
            data_points.append(self._test(gcmd))
            next_test_z = self.probe_points[2] - self.step_size
            next_test_pos = (self.probe_points[0], self.probe_points[1], next_test_z)
            self.probe_points = next_test_pos
        tap_data = TapResonanceData(data_points, "/tmp")
        tap_data.write_data()

    cmd_CALIBRATE_Z_RESONANCE_help = "Calibrate Z making the bed vibrate while probing with the nozzle and record accelerometer data"

    def cmd_CALIBRATE_Z_RESONANCE(self, gcmd):
        self.babystep_probe(gcmd)


def load_config(config):
    return ResonanceZProbe(config)
