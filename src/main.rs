use anyhow::{Context, Result};
use log::{error, info, warn};
use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time;

const THERMAL_ZONE: &str = "/sys/class/thermal/thermal_zone6/temp";
const CPU_FREQ_DIR: &str = "/sys/devices/system/cpu";
const STATE_FILE: &str = "/var/lib/cpu-throttle/state.json";
const LOG_FILE: &str = "/var/log/cpu-throttle.log";

// Configuration
const PREDICT_AHEAD_SEC: f64 = 2.0;
const TEMP_FULL_SPEED: i32 = 70;
const TEMP_STEEP_START: i32 = 85;
const TEMP_EMERGENCY: i32 = 95;

const GRANULARITY_KHZ: u64 = 100_000;
const MIN_FREQ_RATIO: f64 = 0.60;  // Minimum frequency is 60% of max
const MID_FREQ_RATIO: f64 = 0.72;  // Mid frequency is ~72% of max

/// Dynamic frequency limits derived from system CPU info
#[derive(Debug, Clone)]
struct FreqLimits {
    max_khz: u64,
    mid_khz: u64,
    min_khz: u64,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct State {
    k_slope: f64,
    k_load: f64,
    error_history: Vec<f64>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            k_slope: 0.5,
            k_load: 0.02,
            error_history: Vec::new(),
        }
    }
}

async fn read_cpu_temp() -> Result<i32> {
    let content = fs::read_to_string(THERMAL_ZONE)
        .with_context(|| format!("Failed to read {}", THERMAL_ZONE))?;
    let temp_millicelsius: i32 = content.trim().parse()?;
    Ok(temp_millicelsius / 1000)
}

fn read_cpu_load() -> Result<u32> {
    // Use /proc/loadavg (1-minute average)
    let content = fs::read_to_string("/proc/loadavg")?;
    let load_str = content.split_whitespace().next().unwrap_or("0.0");
    let load_avg: f64 = load_str.parse().unwrap_or(0.0);
    let cores = num_cpus::get() as f64;
    let usage = (load_avg / cores * 100.0).round() as u32;
    Ok(usage.min(100))
}

fn read_current_freq() -> Result<u64> {
    let path = Path::new(CPU_FREQ_DIR).join("cpu0/cpufreq/scaling_max_freq");
    let content = fs::read_to_string(&path)?;
    Ok(content.trim().parse()?)
}

/// Read CPU's hardware maximum frequency from cpuinfo_max_freq
fn read_cpu_max_freq() -> Result<u64> {
    let path = Path::new(CPU_FREQ_DIR).join("cpu0/cpufreq/cpuinfo_max_freq");
    let content = fs::read_to_string(&path)
        .with_context(|| format!("Failed to read cpuinfo_max_freq from {:?}", path))?;
    let max_khz: u64 = content.trim().parse()
        .with_context(|| format!("Failed to parse cpuinfo_max_freq: {}", content))?;
    Ok(max_khz)
}

/// Create frequency limits based on system CPU max frequency
fn create_freq_limits() -> Result<FreqLimits> {
    let max_khz = read_cpu_max_freq()?;
    let min_khz = (max_khz as f64 * MIN_FREQ_RATIO).round() as u64;
    let mid_khz = (max_khz as f64 * MID_FREQ_RATIO).round() as u64;

    info!("CPU frequency limits: max={} MHz, mid={} MHz, min={} MHz",
          max_khz / 1000, mid_khz / 1000, min_khz / 1000);

    Ok(FreqLimits { max_khz, mid_khz, min_khz })
}

fn set_cpu_freq(freq_khz: u64) -> Result<()> {
    let cpu_dirs: Vec<_> = fs::read_dir(CPU_FREQ_DIR)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("cpu"))
        .collect();

    for entry in cpu_dirs {
        let freq_path = entry.path().join("cpufreq/scaling_max_freq");
        if freq_path.exists() {
            fs::write(&freq_path, freq_khz.to_string())?;
        }
    }
    Ok(())
}

fn calculate_target_freq(effective_temp: i32, limits: &FreqLimits) -> u64 {
    if effective_temp >= TEMP_EMERGENCY {
        limits.min_khz
    } else if effective_temp >= TEMP_STEEP_START {
        // Steep throttling: mid → min (85°C → 95°C)
        let range = (TEMP_EMERGENCY - TEMP_STEEP_START) as u64;
        let offset = (effective_temp - TEMP_STEEP_START) as u64;
        let step = (limits.mid_khz - limits.min_khz) / range;
        let khz = limits.mid_khz - offset * step;
        khz.max(limits.min_khz)
    } else if effective_temp > TEMP_FULL_SPEED {
        // Linear throttling: max → mid (70°C → 85°C)
        let range = (TEMP_STEEP_START - TEMP_FULL_SPEED) as u64;
        let offset = (effective_temp - TEMP_FULL_SPEED) as u64;
        let step = (limits.max_khz - limits.mid_khz) / range;
        let khz = limits.max_khz - offset * step;
        khz.max(limits.mid_khz)
    } else {
        limits.max_khz
    }
}

fn quantize_freq(freq: u64, limits: &FreqLimits) -> u64 {
    let rounded = ((freq + GRANULARITY_KHZ / 2) / GRANULARITY_KHZ) * GRANULARITY_KHZ;
    rounded.max(limits.min_khz)
}

async fn save_state(state: &State) -> Result<()> {
    fs::create_dir_all("/var/lib/cpu-throttle")?;
    let json = serde_json::to_string_pretty(state)?;
    fs::write(STATE_FILE, json)?;
    Ok(())
}

async fn load_state() -> Result<State> {
    if Path::new(STATE_FILE).exists() {
        let content = fs::read_to_string(STATE_FILE)?;
        Ok(serde_json::from_str(&content)?)
    } else {
        Ok(State::default())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    // Ensure log file exists
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_FILE)?;

    let mut state = load_state().await.unwrap_or_else(|e| {
        warn!("Failed to load state: {}, using defaults", e);
        State::default()
    });

    // Initialize frequency limits from system CPU info
    let freq_limits = create_freq_limits()?;

    let mut history: Vec<(u64, i32, u32, f64)> = Vec::new(); // (timestamp, temp, load, predicted_future_temp)

    loop {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let temp_c = read_cpu_temp().await?;
        let cpu_load = read_cpu_load()?;

        // Calculate temperature slope
        let temp_slope = if history.len() >= 2 {
            let (t0, temp0, _, _) = history[history.len() - 2];
            let (t1, temp1, _, _) = history[history.len() - 1];
            if t1 > t0 {
                (temp1 - temp0) as f64 / (t1 - t0) as f64
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Predictive control
        let predicted_rise = (temp_slope * PREDICT_AHEAD_SEC * state.k_slope).clamp(-3.0, 8.0);
        let load_bias = (cpu_load as f64 * state.k_load).clamp(0.0, 4.0);
        let effective_temp = (temp_c as f64 + predicted_rise + load_bias).round() as i32;

        // Check prediction accuracy
        let mut pred_err: Option<f64> = None;
        let target_ts = now - 2;
        if let Some((_, _, _, past_pred)) = history.iter().find(|(ts, _, _, _)| *ts == target_ts) {
            let err = past_pred - temp_c as f64;
            pred_err = Some(err);

            // Update self-tuning parameters
            state.error_history.push(err);
            if state.error_history.len() > 10 {
                state.error_history.remove(0);
            }

            if !state.error_history.is_empty() {
                let avg_err: f64 =
                    state.error_history.iter().sum::<f64>() / state.error_history.len() as f64;
                state.k_slope = (state.k_slope - 0.05 * avg_err).clamp(0.1, 1.5);
                state.k_load = (state.k_load - 0.005 * avg_err).clamp(0.01, 0.08);
            }

            save_state(&state).await?;
        }

        // Frequency control
        let target_khz = calculate_target_freq(effective_temp, &freq_limits);
        let rounded_khz = quantize_freq(target_khz, &freq_limits);
        let current_khz = read_current_freq().unwrap_or(rounded_khz);

        if rounded_khz != current_khz {
            if let Err(e) = set_cpu_freq(rounded_khz) {
                error!("Failed to set CPU frequency: {}", e);
            }
        }

        // Save history
        let predicted_future_temp = temp_c as f64 + predicted_rise;
        history.push((now, temp_c, cpu_load, predicted_future_temp));
        if history.len() > 20 {
            history.remove(0);
        }

        // Log
        let freq_mhz = rounded_khz / 1000;
        let mut log_msg = format!(
            "Temp={}°C | eff={} | dT/dt={:.3}/s | Load={}% → {} MHz",
            temp_c, effective_temp, temp_slope, cpu_load, freq_mhz
        );
        if let Some(err) = pred_err {
            log_msg.push_str(&format!(" | pred_err={:.1}°C", err));
        }
        log_msg.push_str(&format!(
            " | k_slope={:.3} k_load={:.4}",
            state.k_slope, state.k_load
        ));

        // Write to log file
        let mut file = fs::OpenOptions::new().append(true).open(LOG_FILE)?;
        use std::io::Write;
        writeln!(
            file,
            "{} | {}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            log_msg
        )?;

        info!("{}", log_msg);

        // Sleep for 1 second
        time::sleep(Duration::from_secs(1)).await;
    }
}
