# Resonance Z Probe

## Overview
This klipper extension, `resonance_z_probe.py`, is designed to operate a resonance-based Z probe for 3D printers. The script interfaces with the hardware to control an ADXL345 sensor and gather data, facilitating precise Z-axis measurements. It is mainly meant to automatically find the optimal Z-offset value of an existing toolhead probe to the nozzle.

## Requirements
- Python 3.x
- Klipper
- Required Python libraries:
  - `numpy`
  - `matplotlib`

## Installation
1. Clone or download the repository:
    ```
    git clone https://github.com/ffavero/resonance_tap_test.git
    ```
2. Copy/move the file `resonance_z_probe.py` to `~/klipper/klippy/extras/`.
    ```
    cp resonance_tap_test/resonance_z_probe.py ~/klipper/klippy/extras/
    ```

## Configuration

```yaml
[resonance_z_probe]
accel_chip: adxl345
step_size: 0.01
samples_tolerance: 0.001
accel_per_hz: 74
z_vibration_freq: 200
amplitude_threshold: 550
rate_above_threshold: 0.01
probe_points: 125, 125, 5 # the x,y,z coordinate of
                          # the probingposition.
safe_min_z: 4.5 # change this to a z_value you are
                          # sure it's touching the bed
                          # (but not too much to damage anything)
cycle_per_test: 3
calibration_points:
  235,210 # probe location Right Rear
  2,210 #probe location Left Rear
  2,-15 #probe location Left Front
  235,-15 #probe location Right Front
```


## Usage

In klipper console use the macros


```
CALIBRATE_THRESHOLD
CALIBRATE_Z_RESONANCE
```

## Credits
This script was developed by [ffavero](https://github.com/ffavero) for the purpose of operating a resonance-based Z probe. 

For any issues or suggestions, please visit the [GitHub repository](https://github.com/ffavero/resonance_tap_test) and submit an issue or pull request.


## Licence
GPLv3
