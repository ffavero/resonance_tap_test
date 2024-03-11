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

# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt


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
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object("gcode")
        # consider that accel_per_hz * freq might be caped to the printer max accel
        self.accel_per_hz = config.getfloat("accel_per_hz", 1.5, above=0.0)

        self.step_size = config.getfloat("babystep", 0.01, minval=0.005)
        self.z_freq = config.getfloat("z_vibration_freq", 80, minval=50.0, maxval=200.0)
        self.amp_threshold = config.getfloat("amplitude_threshold", 700.0, above=500.0)
        self.safe_min_z = config.getfloat("safe_min_z", 1)

        self.cycle_per_test = config.getint("cycle_per_test", 50, minval=2, maxval=500)
        self.probe_points = config.getfloatlist("probe_points", sep=",", count=3)
        self.accel_chip_name = config.get("accel_chip").strip()
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
        self.debug = True
        self.data_points = []

    def connect(self):
        self.accel_chips = ("z", self.printer.lookup_object(self.accel_chip_name))

    def _test(self, gcmd):
        toolhead = self.printer.lookup_object("toolhead")

        toolhead.manual_move(self.probe_points, 50.0)
        toolhead.wait_moves()
        toolhead.dwell(0.500)
        chip_axis, chip = self.accel_chips
        aclient = chip.start_internal_client()
        aclient.msg = []
        aclient.samples = []
        self.vibration_helper.vibrate_n(self.cycle_per_test)
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
        try:
            rate_above_tr = sum(
                np.logical_or(z > self.amp_threshold, z < (-1 * self.amp_threshold))
            ) / len(timestamps)
        except ZeroDivisionError:
            rate_above_tr = 0
        cur_z = self.probe_points[2]
        if len(timestamps) > 0:
            test_time = timestamps[len(timestamps) - 1] - timestamps[0]
        else:
            test_time = 0
        if self.debug:
            gcmd.respond_info(
                "Testing Z: %.4f. Received %i samples in %.2f seconds. Percentage above threshold: %.1f%%"
                % (cur_z, len(timestamps), test_time, 100 * rate_above_tr)
            )
        if self.debug:
            self.data_points.append(TestPoint(timestamps, x, y, z, cur_z))
        return rate_above_tr

    def babystep_probe(self, gcmd):
        """
        move to the test position, start recording acc while resonate z, measure the amplitude and compare with the amp_threshold.
        lower by babystep until min safe Z is reached or the amp_threshold is passed.
        log stuff in the console
        """
        # move thself.probe_pointe toolhead
        test_results = []
        self.offset_helper = OffsetHelper(self.probe_points[2], self.step_size, 0.005)

        while self.probe_points[2] >= self.safe_min_z:
            if self.offset_helper.finished:
                break
            results = self._test(gcmd)
            self.offset_helper.last_tested_position(
                self.probe_points[2], results >= 0.01
            )
            test_results.append((self.probe_points[2], results))
            next_test_z = self.offset_helper.next_position()
            next_test_pos = (self.probe_points[0], self.probe_points[1], next_test_z)
            self.probe_points = next_test_pos
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

        if self.debug:
            tap_data = TapResonanceData(self.data_points, "/tmp")
            tap_data.write_data()

    cmd_CALIBRATE_Z_RESONANCE_help = "Calibrate Z making the bed vibrate while probing with the nozzle and record accelerometer data"

    def cmd_CALIBRATE_Z_RESONANCE(self, gcmd):
        self.babystep_probe(gcmd)


def load_config(config):
    return ResonanceZProbe(config)
