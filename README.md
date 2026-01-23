# cpu-throttle

A predictive CPU thermal control daemon for Linux that dynamically regulates CPU frequency to prevent overheating through feed-forward prediction control. Written in Rust with the Tokio async runtime.

## Overview

Unlike simple temperature-threshold-based solutions, `cpu-throttle` employs a **predictive control algorithm** that anticipates thermal trends before critical temperatures are reached. By analyzing temperature slope and CPU load, the system proactively adjusts frequency to maintain optimal thermal balance.

## Features

- **Feed-Forward Predictive Control**: Anticipates temperature changes using rate-of-rise calculations
- **Adaptive Self-Tuning**: Automatically adjusts prediction parameters based on historical accuracy
- **State Persistence**: Saves optimized parameters across reboots
- **Granular Frequency Control**: 100 MHz adjustment granularity for smooth thermal response
- **Comprehensive Logging**: Detailed operational logs to `/var/log/cpu-throttle.log`

## How It Works

### 1. Temperature Prediction

The system calculates an *effective temperature* for control decisions:

```
predicted_rise = (dT/dt × PREDICT_AHEAD_SEC × k_slope).clamp(-3.0, 8.0)
load_bias = (cpu_load × k_load).clamp(0.0, 4.0)
effective_temp = current_temp + predicted_rise + load_bias
```

Where:
- `dT/dt` = Temperature slope calculated from recent history
- `PREDICT_AHEAD_SEC` = Prediction horizon (default: 2 seconds)
- `k_slope`, `k_load` = Adaptive tuning parameters

### 2. Adaptive Parameter Tuning

Every 2 seconds, the system compares past predictions with actual temperatures:

```
error = predicted_temp - actual_temp
avg_error = mean(recent_10_errors)

k_slope = clamp(k_slope - 0.05 × avg_error, 0.1, 1.5)
k_load = clamp(k_load - 0.005 × avg_error, 0.01, 0.08)
```

Parameters are persisted to `/var/lib/cpu-throttle/state.json`.

### 3. Frequency Control Strategy

The daemon automatically reads the CPU's maximum frequency from the system
(`cpuinfo_max_freq`) and calculates throttling thresholds dynamically:

| Effective Temperature | Frequency Behavior |
|-----------------------|-------------------|
| < 70°C | 100% of CPU max frequency |
| 70-85°C | 100% → 72% (linear throttle) |
| 85-95°C | 72% → 60% (aggressive throttle) |
| ≥ 95°C | 60% (minimum frequency) |

Frequency ratios are configurable via `MIN_FREQ_RATIO` (default: 0.60) and
`MID_FREQ_RATIO` (default: 0.72) in the source code.

## Installation

### Prerequisites

- Linux with sysfs CPU frequency scaling support
- Rust toolchain (1.70+)
- Root access (required for writing to cpufreq sysfs)

### Build from Source

```bash
# Clone or navigate to project directory
cd cpu-throttle

# Build release binary
cargo build --release

# (Optional) Install system-wide
sudo install -m 755 target/release/cpu-throttle /usr/local/bin/
```

## Usage

### Running the Daemon

```bash
# Run directly (requires root)
sudo ./target/release/cpu-throttle

# Or if installed system-wide
sudo cpu-throttle
```

### CLI Commands

#### Validate Configuration

Before starting the daemon, validate your configuration file:

```bash
# Validate default config location (/etc/cpu-throttle/config.toml)
sudo cpu-throttle validate-config

# Validate a custom config file
sudo cpu-throttle validate-config --config /path/to/custom-config.toml

# Validate the example configuration
cpu-throttle validate-config --config config.example.toml
```

**Example output:**
```
Validating configuration file: config.example.toml
✓ Configuration is valid!

Current settings:
  thermal_zone        : thermal_zone6
  predict_ahead_sec   : 2 seconds
  temp_full_speed     : 70°C
  temp_steep_start    : 85°C
  temp_emergency      : 95°C
  granularity_khz     : 100 MHz
  min_freq_ratio      : 0.6 (60%)
  mid_freq_ratio      : 0.72 (72%)

Derived frequency limits:
  max_frequency       : 3900 MHz
  mid_frequency       : 2808 MHz
  min_frequency       : 2340 MHz
```

#### Generate Configuration

Generate a new configuration file with default values:

```bash
# Generate to default location (/etc/cpu-throttle/config.toml)
sudo cpu-throttle generate-config

# Generate to a custom location
cpu-throttle generate-config --output /path/to/config.toml
```

**Example output:**
```
✓ Generated default configuration file: /etc/cpu-throttle/config.toml

Review and adjust the following settings:
  thermal_zone        : thermal_zone6
  predict_ahead_sec   : 2 seconds
  temp_full_speed     : 70°C
  temp_steep_start    : 85°C
  temp_emergency      : 95°C
  granularity_khz     : 100 MHz
  min_freq_ratio      : 0.6 (60%)
  mid_freq_ratio      : 0.72 (72%)

Validate the configuration with:
  sudo cpu-throttle validate-config --config /etc/cpu-throttle/config.toml
```

**Note:** The command will fail if the file already exists. Remove the existing file first if you want to regenerate it.

#### Help and Version

```bash
# Show help
cpu-throttle --help
cpu-throttle validate-config --help

# Show version
cpu-throttle --version
```

### Systemd Service Deployment

For automatic startup at boot, deploy as a systemd service.

#### 1. Create Service File

Create `/etc/systemd/system/cpu-throttle.service`:

```ini
[Unit]
Description=Predictive CPU Thermal Control Daemon
Documentation=https://github.com/unclemate/cpu-throttle
After=multi-user.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/cpu-throttle
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/cpu-throttle /var/log/cpu-throttle.log

[Install]
WantedBy=multi-user.target
```

#### 2. Enable and Start Service

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable cpu-throttle

# Start immediately
sudo systemctl start cpu-throttle
```

#### 3. Service Management Commands

```bash
# Check service status
sudo systemctl status cpu-throttle

# View real-time logs
sudo journalctl -u cpu-throttle -f

# View logs since boot
sudo journalctl -u cpu-throttle --since boot

# Stop the service
sudo systemctl stop cpu-throttle

# Restart the service
sudo systemctl restart cpu-throttle

# Disable auto-start
sudo systemctl disable cpu-throttle
```

## Configuration

The daemon can be configured via a TOML configuration file at `/etc/cpu-throttle/config.toml`. If the file doesn't exist, the daemon uses built-in defaults.

### Configuration File

Create a configuration file using either method:

**Option 1: Generate default configuration**
```bash
sudo mkdir -p /etc/cpu-throttle
sudo cpu-throttle generate-config
sudo editor /etc/cpu-throttle/config.toml
```

**Option 2: Copy example configuration**
```bash
sudo mkdir -p /etc/cpu-throttle
sudo cp config.example.toml /etc/cpu-throttle/config.toml
sudo editor /etc/cpu-throttle/config.toml
```

**Validate your configuration before starting the daemon:**

```bash
sudo cpu-throttle validate-config
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `thermal_zone` | `thermal_zone6` | Path to CPU temperature sensor |
| `predict_ahead_sec` | `2.0` | Prediction horizon in seconds |
| `temp_full_speed` | `70` | Max frequency temperature threshold (°C) |
| `temp_steep_start` | `85` | Aggressive throttle start (°C) |
| `temp_emergency` | `95` | Emergency mode threshold (°C) |
| `min_freq_ratio` | `0.60` | Minimum frequency as ratio of CPU max |
| `mid_freq_ratio` | `0.72` | Mid frequency as ratio of CPU max |
| `granularity_khz` | `100_000` | Frequency adjustment step (Hz) |

**Notes:**
- CPU maximum frequency is read automatically from `/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq`
- You may need to adjust `thermal_zone` depending on your hardware. Check available zones with:
  ```bash
  ls /sys/class/thermal/
  cat /sys/class/thermal/thermal_zone6/type  # Verify it's CPU-related
  ```

## Data Files

### Log File (`/var/log/cpu-throttle.log`)

**Log Rotation Strategy (KISS):** The log file is automatically truncated on each daemon startup, keeping only logs from the current session. This simple approach avoids complex rotation logic while ensuring logs remain fresh and manageable.

The daemon continuously appends operational logs with the following format:

```
2025-01-22 14:30:15 | Temp=72°C | eff=78 | dT/dt=1.234/s | Load=45% → 3500 MHz | pred_err=0.3°C | k_slope=0.520 k_load=0.0250
```

**Field descriptions:**

| Field | Description |
|-------|-------------|
| `Temp` | Current CPU temperature in Celsius |
| `eff` | Effective temperature (prediction-based control value) |
| `dT/dt` | Temperature rate of change (°C/second) |
| `Load` | CPU load percentage (normalized by core count) |
| `→ X MHz` | Target CPU frequency being applied |
| `pred_err` | Prediction error from 2 seconds ago (positive = over-predicted) |
| `k_slope` | Current adaptive slope coefficient |
| `k_load` | Current adaptive load coefficient |

**Monitoring logs in real-time:**
```bash
# Tail the log file
sudo tail -f /var/log/cpu-throttle.log

# View recent temperature trends
sudo tail -n 100 /var/log/cpu-throttle.log | awk '{print $3}' | sort -n

# Check for errors
sudo grep -i error /var/log/cpu-throttle.log
```

### Persistent State (`/var/lib/cpu-throttle/state.json`)

The adaptive tuning parameters are automatically saved and restored across daemon restarts:

```json
{
  "k_slope": 0.523,
  "k_load": 0.0248,
  "error_history": [0.3, -0.1, 0.2, 0.0, 0.1, -0.2, 0.4, 0.1, -0.1, 0.0]
}
```

**Parameter meanings:**

| Parameter | Range | Effect |
|-----------|-------|--------|
| `k_slope` | 0.1 – 1.5 | Sensitivity to temperature rate-of-change (higher = more aggressive response to heating) |
| `k_load` | 0.01 – 0.08 | Load bias coefficient (higher = more throttling under load) |
| `error_history` | Last 10 errors | Used for calculating average prediction error |

**Resetting to defaults:**
```bash
# Stop the service first
sudo systemctl stop cpu-throttle

# Remove saved state to reset parameters
sudo rm /var/lib/cpu-throttle/state.json

# Restart (will use default values: k_slope=0.5, k_load=0.02)
sudo systemctl start cpu-throttle
```

## File System Layout

```
┌─────────────────────────────────────────────────────────────┐
│                       System Paths                          │
├─────────────────────────────────────────────────────────────┤
│ /usr/local/bin/cpu-throttle    → Daemon binary              │
│ /etc/systemd/system/           → Service configuration       │
│    cpu-throttle.service                                     │
├─────────────────────────────────────────────────────────────┤
│ /var/lib/cpu-throttle/         → State storage directory    │
│    state.json                  → Adaptive parameters (auto) │
├─────────────────────────────────────────────────────────────┤
│ /var/log/cpu-throttle.log      → Operational logs           │
└─────────────────────────────────────────────────────────────┘
```

## System Dependencies

| Path | Purpose |
|------|---------|
| `/sys/class/thermal/thermal_zone*/temp` | CPU temperature reading |
| `/sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq` | Frequency control |
| `/proc/loadavg` | CPU load reading |
| `/var/lib/cpu-throttle/` | State persistence |
| `/var/log/cpu-throttle.log` | Operational logs |

## Development

```bash
# Check code (fast compilation)
cargo check

# Run tests
cargo test

# Build with debug symbols
cargo build
```

## License

This project is licensed under the GNU General Public License v3.0 or later. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please ensure any changes maintain the adaptive tuning behavior and thermal safety margins.
