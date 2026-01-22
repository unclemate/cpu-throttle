use anyhow::{Context, Result};
use log::{error, info, warn};
use std::fs;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::signal;
use tokio::time;

const CPU_FREQ_DIR: &str = "/sys/devices/system/cpu";
const STATE_FILE: &str = "/var/lib/cpu-throttle/state.json";
const LOG_FILE: &str = "/var/log/cpu-throttle.log";
const CONFIG_FILE: &str = "/etc/cpu-throttle/config.toml";

/// Configuration loaded from file or defaults
#[derive(serde::Deserialize, Debug, Clone)]
struct Config {
    #[serde(default = "default_thermal_zone")]
    thermal_zone: String,

    #[serde(default = "default_predict_ahead")]
    predict_ahead_sec: f64,

    #[serde(default = "default_temp_full_speed")]
    temp_full_speed: i32,

    #[serde(default = "default_temp_steep_start")]
    temp_steep_start: i32,

    #[serde(default = "default_temp_emergency")]
    temp_emergency: i32,

    #[serde(default = "default_granularity")]
    granularity_khz: u64,

    #[serde(default = "default_min_freq_ratio")]
    min_freq_ratio: f64,

    #[serde(default = "default_mid_freq_ratio")]
    mid_freq_ratio: f64,
}

// Default functions for serde
fn default_thermal_zone() -> String { "thermal_zone6".to_string() }
fn default_predict_ahead() -> f64 { 2.0 }
fn default_temp_full_speed() -> i32 { 70 }
fn default_temp_steep_start() -> i32 { 85 }
fn default_temp_emergency() -> i32 { 95 }
fn default_granularity() -> u64 { 100_000 }
fn default_min_freq_ratio() -> f64 { 0.60 }
fn default_mid_freq_ratio() -> f64 { 0.72 }

impl Default for Config {
    fn default() -> Self {
        Self {
            thermal_zone: default_thermal_zone(),
            predict_ahead_sec: default_predict_ahead(),
            temp_full_speed: default_temp_full_speed(),
            temp_steep_start: default_temp_steep_start(),
            temp_emergency: default_temp_emergency(),
            granularity_khz: default_granularity(),
            min_freq_ratio: default_min_freq_ratio(),
            mid_freq_ratio: default_mid_freq_ratio(),
        }
    }
}

/// Load configuration from file, falling back to defaults
fn load_config() -> Config {
    if let Ok(content) = fs::read_to_string(CONFIG_FILE) {
        match toml::from_str(&content) {
            Ok(config) => {
                info!("Loaded configuration from {}", CONFIG_FILE);
                return config;
            }
            Err(e) => {
                warn!("Failed to parse config file: {}, using defaults", e);
            }
        }
    } else {
        info!("No config file found at {}, using defaults", CONFIG_FILE);
    }
    Config::default()
}

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

async fn read_cpu_temp(thermal_zone: &str) -> Result<i32> {
    let zone_path = format!("/sys/class/thermal/{}/temp", thermal_zone);
    let content = fs::read_to_string(&zone_path)
        .with_context(|| format!("Failed to read {}", zone_path))?;
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

/// Create frequency limits based on system CPU max frequency and config
fn create_freq_limits(config: &Config) -> Result<FreqLimits> {
    let max_khz = read_cpu_max_freq()?;
    let min_khz = (max_khz as f64 * config.min_freq_ratio).round() as u64;
    let mid_khz = (max_khz as f64 * config.mid_freq_ratio).round() as u64;

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

fn calculate_target_freq(effective_temp: i32, limits: &FreqLimits, config: &Config) -> u64 {
    if effective_temp >= config.temp_emergency {
        limits.min_khz
    } else if effective_temp >= config.temp_steep_start {
        // Steep throttling: mid → min
        let range = (config.temp_emergency - config.temp_steep_start) as u64;
        let offset = (effective_temp - config.temp_steep_start) as u64;
        let step = (limits.mid_khz - limits.min_khz) / range;
        let khz = limits.mid_khz - offset * step;
        khz.max(limits.min_khz)
    } else if effective_temp > config.temp_full_speed {
        // Linear throttling: max → mid
        let range = (config.temp_steep_start - config.temp_full_speed) as u64;
        let offset = (effective_temp - config.temp_full_speed) as u64;
        let step = (limits.max_khz - limits.mid_khz) / range;
        let khz = limits.max_khz - offset * step;
        khz.max(limits.mid_khz)
    } else {
        limits.max_khz
    }
}

fn quantize_freq(freq: u64, limits: &FreqLimits, config: &Config) -> u64 {
    let rounded = ((freq + config.granularity_khz / 2) / config.granularity_khz) * config.granularity_khz;
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

    // Load configuration from file or use defaults
    let config = load_config();
    info!("Using configuration: thermal_zone={}, predict_ahead_sec={}s, temp_full_speed={}°C",
          config.thermal_zone, config.predict_ahead_sec, config.temp_full_speed);

    // Open log file with buffered writer for better performance
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_FILE)
        .context("Failed to open log file")?;
    let log_writer = Arc::new(Mutex::new(BufWriter::new(log_file)));

    let mut state = load_state().await.unwrap_or_else(|e| {
        warn!("Failed to load state: {}, using defaults", e);
        State::default()
    });

    // Initialize frequency limits from system CPU info and config
    let freq_limits = create_freq_limits(&config)?;

    let mut history: Vec<(u64, i32, u32, f64)> = Vec::new(); // (timestamp, temp, load, predicted_future_temp)

    // Set up graceful shutdown handler
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();

    // Spawn signal handler task
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("Failed to setup SIGTERM handler");
            let mut sigint = signal::unix::signal(signal::unix::SignalKind::interrupt())
                .expect("Failed to setup SIGINT handler");
            tokio::select! {
                _ = signal::ctrl_c() => {
                    info!("Received Ctrl-C, initiating graceful shutdown...");
                }
                _ = sigterm.recv() => {
                    info!("Received SIGTERM, initiating graceful shutdown...");
                }
                _ = sigint.recv() => {
                    info!("Received SIGINT, initiating graceful shutdown...");
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = signal::ctrl_c().await;
            info!("Received Ctrl-C, initiating graceful shutdown...");
        }
        shutdown_clone.store(true, Ordering::SeqCst);
    });

    // Main control loop
    loop {
        // Check for shutdown signal
        if shutdown.load(Ordering::SeqCst) {
            info!("Shutdown signal received, stopping control loop...");
            break;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let temp_c = read_cpu_temp(&config.thermal_zone).await?;
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

        // Predictive control using config values
        let predicted_rise = (temp_slope * config.predict_ahead_sec * state.k_slope).clamp(-3.0, 8.0);
        let load_bias = (cpu_load as f64 * state.k_load).clamp(0.0, 4.0);
        let effective_temp = (temp_c as f64 + predicted_rise + load_bias).round() as i32;

        // Check prediction accuracy (use closest match within 1-3 seconds ago)
        let mut pred_err: Option<f64> = None;
        if let Some((_, _, _, past_pred)) = history.iter()
            .filter(|(ts, _, _, _)| *ts >= now - 3 && *ts <= now - 1)
            .min_by_key(|(ts, _, _, _)| ts.abs_diff(now - 2)) {
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
        let target_khz = calculate_target_freq(effective_temp, &freq_limits, &config);
        let rounded_khz = quantize_freq(target_khz, &freq_limits, &config);
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

        // Write to log file using buffered writer
        if let Ok(mut writer) = log_writer.lock() {
            let _ = writeln!(
                writer,
                "{} | {}",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                log_msg
            );
            // Flush periodically to ensure data is written
            let _ = writer.flush();
        }

        info!("{}", log_msg);

        // Sleep for 1 second, but check shutdown every 100ms
        for _ in 0..10 {
            if shutdown.load(Ordering::SeqCst) {
                break;
            }
            time::sleep(Duration::from_millis(100)).await;
        }
    }

    // Save state before shutdown
    info!("Saving state before shutdown...");
    if let Err(e) = save_state(&state).await {
        error!("Failed to save state on shutdown: {}", e);
    } else {
        info!("State saved successfully");
    }

    info!("Shutdown complete");
    Ok(())
}
