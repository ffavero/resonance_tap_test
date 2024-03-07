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

import logging
import math
import time


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


class ResonanceZProbe:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object("gcode")
        self.accel_per_hz = config.getfloat("accel_per_hz", 75.0, above=0.0)

        self.step_size = config.getfloat("babystep", 0.01, minval=0.005)
        # Defaults are such that max_freq * accel_per_hz == 10000 (max_accel)
        self.z_freq = config.getfloat("z_vibration_freq", 180, minval=80, maxval=200.0)
        self.amp_threshold = config.getfloat("amplitude_threshold", 700, above=500)
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
        end = time.time() + seconds
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
        while time.time() < end:
            t_seg = 0.25 / freq
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
        gcmd.respond_info("DONE")

    def babystep_probe(self, gcmd):
        """
        move to the test position, start recording acc while resonate z, measure the amplitude and compare with the amp_threshold.
        lower by babystep untile min safe Z is reached or the amp_threshold is passed.
        log stuff in the console
        """
        toolhead = self.printer.lookup_object("toolhead")
        # move thself.probe_pointe toolhead

        toolhead.manual_move(self.probe_points, 50.0)

        toolhead.wait_moves()
        toolhead.dwell(0.500)

        raw_values = []
        chip_axis, chip = self.accel_chips
        aclient = chip.start_internal_client()
        raw_values.append(("z", aclient, chip.name))
        self.hold_z_freq(gcmd, self.test_time, self.z_freq)
        for chip_axis, aclient, chip_name in raw_values:
            aclient.finish_measurements()
            raw_name = "/tmp/adxl_z_test.csv"
            for sample in aclient.get_samples():
                gcmd.respond_info(str(sample))

    cmd_CALIBRATE_Z_RESONANCE_help = "Calibrate Z making the bed vibrate while probing with the nozzle and record accelerometer data"

    def cmd_CALIBRATE_Z_RESONANCE(self, gcmd):
        self.babystep_probe(gcmd)


def load_config(config):
    return ResonanceZProbe(config)
