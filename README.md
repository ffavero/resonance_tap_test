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
babystep: 0.01
accel_per_hz: 75
z_vibration_freq: 200
amplitude_threshold: 550
probe_points: 80, 150, 0.3
safe_min_z: 0.15
cycle_per_test: 20
```


## Usage

In klipper console use the macro


```
CALIBRATE_Z_RESONANCE
```

## Credits
This script was developed by [ffavero](https://github.com/ffavero) for the purpose of operating a resonance-based Z probe. 

For any issues or suggestions, please visit the [GitHub repository](https://github.com/ffavero/resonance_tap_test) and submit an issue or pull request.


## Licence
GPLv3