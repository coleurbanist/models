# Build Instructions

## 1. Python dependencies

```bash
pip install -r requirements.txt
```

## 2. Rust simulator

Install Rust (one-time):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Windows: download and run rustup-init.exe from https://rustup.rs
```

Build the simulator binary:
```bash
cd simulator
cargo build --release
```

The binary will be at `simulator/target/release/simulator` (or `.exe` on Windows).

## 3. Run a race

From the repo root:

```bash
# IL-09 2026
python -m races.il09_2026.run

# With early/mail vote estimates
python -m races.il09_2026.run --banked

# Chicago Mayor 2027 (once polls and shapefiles are in place)
python -m races.chicago_mayor_2027.run
```

## 4. Pollster calibration (Chicago mayor)

1. Fill in `pollster_calibration/data/2023_chicago_polls.csv`
   (schema in `pollster_calibration/pollster_calibration.py` docstring)
2. Fill in `ACTUAL_2023` dict in `pollster_calibration.py` from BOE certified results
3. Run:
   ```bash
   python pollster_calibration/pollster_calibration.py
   ```
   Outputs `pollster_calibration/pollster_ratings.json`

## 5. Adding a new race

1. Create `races/<race_id>/`
2. Copy `races/il09_2026/race_config.py` → update all fields
3. Copy `races/il09_2026/run.py` → change the CONFIG import
4. Add a nav entry to `core/site_config.py`
5. Drop shapefiles into `races/<race_id>/data/shapefile/`
6. Run `python -m races.<race_id>.run`
