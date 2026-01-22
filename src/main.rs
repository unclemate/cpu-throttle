use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
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

/// Predictive control constants
const PREDICTED_RISE_MIN: f64 = -3.0;
const PREDICTED_RISE_MAX: f64 = 8.0;
const LOAD_BIAS_MIN: f64 = 0.0;
const LOAD_BIAS_MAX: f64 = 4.0;

/// Adaptive tuning constants
const ERROR_HISTORY_SIZE: usize = 10;
const HISTORY_SIZE: usize = 20;
const K_SLOPE_MIN: f64 = 0.1;
const K_SLOPE_MAX: f64 = 1.5;
const K_LOAD_MIN: f64 = 0.01;
const K_LOAD_MAX: f64 = 0.08;
const K_SLOPE_ADJUST: f64 = 0.05;
const K_LOAD_ADJUST: f64 = 0.005;

/// Command-line interface arguments
#[derive(Parser, Debug)]
#[command(name = "cpu-throttle")]
#[command(author = "unclemate")]
#[command(version = "0.1.0")]
#[command(about = "Predictive CPU thermal control daemon for Linux", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

/// Available subcommands
#[derive(Subcommand, Debug)]
enum Commands {
    /// Validate configuration file syntax and constraints
    ValidateConfig {
        /// Path to configuration file
        #[arg(short, long, default_value = CONFIG_FILE)]
        config: String,
    },
    /// Generate default configuration file
    GenerateConfig {
        /// Path where configuration file will be written
        #[arg(short, long, default_value = CONFIG_FILE)]
        output: String,
    },
}

/// Configuration loaded from file or defaults
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
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

impl Config {
    /// Validate configuration constraints
    fn validate(&self) -> Result<()> {
        if self.temp_full_speed >= self.temp_steep_start {
            bail!(
                "Invalid temperature thresholds: temp_full_speed ({}°C) must be < temp_steep_start ({}°C)",
                self.temp_full_speed, self.temp_steep_start
            );
        }
        if self.temp_steep_start >= self.temp_emergency {
            bail!(
                "Invalid temperature thresholds: temp_steep_start ({}°C) must be < temp_emergency ({}°C)",
                self.temp_steep_start, self.temp_emergency
            );
        }
        if self.min_freq_ratio >= self.mid_freq_ratio {
            bail!(
                "Invalid frequency ratios: min_freq_ratio ({}) must be < mid_freq_ratio ({})",
                self.min_freq_ratio, self.mid_freq_ratio
            );
        }
        if self.mid_freq_ratio >= 1.0 {
            bail!(
                "Invalid frequency ratio: mid_freq_ratio ({}) must be < 1.0",
                self.mid_freq_ratio
            );
        }
        if self.predict_ahead_sec <= 0.0 {
            bail!(
                "Invalid prediction horizon: predict_ahead_sec ({}) must be > 0",
                self.predict_ahead_sec
            );
        }
        if self.granularity_khz == 0 {
            bail!(
                "Invalid granularity: granularity_khz must be > 0"
            );
        }
        Ok(())
    }
}

/// Validate configuration file and print results
fn validate_config(path: &str) -> Result<()> {
    println!("Validating configuration file: {}", path);

    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path))?;

    let config: Config = toml::from_str(&content)
        .with_context(|| format!("Failed to parse TOML from config file: {}", path))?;

    config.validate()?;

    // Print configuration summary
    println!("✓ Configuration is valid!\n");
    println!("Current settings:");
    println!("  thermal_zone        : {}", config.thermal_zone);
    println!("  predict_ahead_sec   : {} seconds", config.predict_ahead_sec);
    println!("  temp_full_speed     : {}°C", config.temp_full_speed);
    println!("  temp_steep_start    : {}°C", config.temp_steep_start);
    println!("  temp_emergency      : {}°C", config.temp_emergency);
    println!("  granularity_khz     : {} MHz", config.granularity_khz / 1000);
    println!("  min_freq_ratio      : {} ({}%)", config.min_freq_ratio, config.min_freq_ratio * 100.0);
    println!("  mid_freq_ratio      : {} ({}%)", config.mid_freq_ratio, config.mid_freq_ratio * 100.0);

    // Calculate derived frequency limits
    match create_freq_limits(&config) {
        Ok(limits) => {
            println!("\nDerived frequency limits:");
            println!("  max_frequency       : {} MHz", limits.max_khz / 1000);
            println!("  mid_frequency       : {} MHz", limits.mid_khz / 1000);
            println!("  min_frequency       : {} MHz", limits.min_khz / 1000);
        }
        Err(e) => {
            println!("\n⚠ Warning: Could not calculate frequency limits: {}", e);
            println!("  (This is expected if not running on the target system)");
        }
    }

    Ok(())
}

/// Generate default configuration file
fn generate_config(output_path: &str) -> Result<()> {
    let config = Config::default();

    // Check if file already exists
    if Path::new(output_path).exists() {
        bail!(
            "Configuration file already exists: {}\nUse --force to overwrite or remove the file first.",
            output_path
        );
    }

    // Create parent directory if it doesn't exist
    if let Some(parent) = Path::new(output_path).parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {:?}", parent))?;
    }

    // Serialize to TOML
    let toml_content = toml::to_string_pretty(&config)
        .context("Failed to serialize config to TOML")?;

    // Add header comments
    let full_content = format!(
        "# cpu-throttle Configuration File\n\
         # Generated by cpu-th {}\n\
         # Modify values as needed for your system\n\
         \n\
         # Run: ls /sys/class/thermal/ to see available zones\n\
         # Run: cat /sys/class/thermal/thermal_zoneX/type to identify CPU zones\n\
         \n\
         {}",
        env!("CARGO_PKG_VERSION"),
        toml_content
    );

    // Write to file
    fs::write(output_path, full_content)
        .with_context(|| format!("Failed to write config file: {}", output_path))?;

    println!("✓ Generated default configuration file: {}", output_path);
    println!("\nReview and adjust the following settings:");
    println!("  thermal_zone        : {}", config.thermal_zone);
    println!("  predict_ahead_sec   : {} seconds", config.predict_ahead_sec);
    println!("  temp_full_speed     : {}°C", config.temp_full_speed);
    println!("  temp_steep_start    : {}°C", config.temp_steep_start);
    println!("  temp_emergency      : {}°C", config.temp_emergency);
    println!("  granularity_khz     : {} MHz", config.granularity_khz / 1000);
    println!("  min_freq_ratio      : {} ({}%)", config.min_freq_ratio, config.min_freq_ratio * 100.0);
    println!("  mid_freq_ratio      : {} ({}%)", config.mid_freq_ratio, config.mid_freq_ratio * 100.0);
    println!("\nValidate the configuration with:");
    println!("  sudo cpu-throttle validate-config --config {}", output_path);

    Ok(())
}

/// Load configuration from file, falling back to defaults
fn load_config(path: &str) -> Config {
    let config = if let Ok(content) = fs::read_to_string(path) {
        match toml::from_str::<Config>(&content) {
            Ok(mut c) => {
                // Ensure thermal_zone is set even if not in file
                if c.thermal_zone.is_empty() {
                    c.thermal_zone = default_thermal_zone();
                }
                if let Err(e) = c.validate() {
                    warn!("Invalid config file: {}, using defaults", e);
                    Config::default()
                } else {
                    info!("Loaded configuration from {}", path);
                    c
                }
            }
            Err(e) => {
                warn!("Failed to parse config file: {}, using defaults", e);
                Config::default()
            }
        }
    } else {
        info!("No config file found at {}, using defaults", path);
        Config::default()
    };

    // Final validation of loaded/defaults
    if let Err(e) = config.validate() {
        error!("Invalid configuration: {}", e);
        Config::default()
    } else {
        config
    }
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
        .with_context(|| format!("Failed to read thermal zone {}", zone_path))?;
    let temp_millicelsius: i32 = content.trim().parse()
        .with_context(|| format!("Invalid temperature value in {}: '{}'", zone_path, content.trim()))?;
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

    // Validate ordering to prevent precision loss issues
    if mid_khz <= min_khz {
        bail!(
            "Invalid frequency limits: mid ({}) <= min ({}). max={}, min_ratio={}, mid_ratio={}",
            mid_khz, min_khz, max_khz, config.min_freq_ratio, config.mid_freq_ratio
        );
    }
    if max_khz <= mid_khz {
        bail!(
            "Invalid frequency limits: max ({}) <= mid ({}). mid_ratio={}",
            max_khz, mid_khz, config.mid_freq_ratio
        );
    }

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
        if range == 0 {
            return limits.min_khz;
        }
        let offset = (effective_temp - config.temp_steep_start) as u64;
        let step = (limits.mid_khz - limits.min_khz) / range;
        let khz = limits.mid_khz - offset * step;
        khz.max(limits.min_khz)
    } else if effective_temp > config.temp_full_speed {
        // Linear throttling: max → mid
        let range = (config.temp_steep_start - config.temp_full_speed) as u64;
        if range == 0 {
            return limits.mid_khz;
        }
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

async fn run_daemon(config_path: &str) -> Result<()> {
    env_logger::init();

    // Load configuration from file or use defaults
    let config = load_config(config_path);
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

        // Predictive control using config values and constants
        let predicted_rise = (temp_slope * config.predict_ahead_sec * state.k_slope)
            .clamp(PREDICTED_RISE_MIN, PREDICTED_RISE_MAX);
        let load_bias = (cpu_load as f64 * state.k_load)
            .clamp(LOAD_BIAS_MIN, LOAD_BIAS_MAX);
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
            if state.error_history.len() > ERROR_HISTORY_SIZE {
                state.error_history.remove(0);
            }

            if !state.error_history.is_empty() {
                let avg_err: f64 =
                    state.error_history.iter().sum::<f64>() / state.error_history.len() as f64;
                state.k_slope = (state.k_slope - K_SLOPE_ADJUST * avg_err)
                    .clamp(K_SLOPE_MIN, K_SLOPE_MAX);
                state.k_load = (state.k_load - K_LOAD_ADJUST * avg_err)
                    .clamp(K_LOAD_MIN, K_LOAD_MAX);
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
        if history.len() > HISTORY_SIZE {
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

        // Write to log file using buffered writer with proper error handling
        match log_writer.lock() {
            Ok(mut writer) => {
                if let Err(e) = writeln!(
                    writer,
                    "{} | {}",
                    chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                    log_msg
                ) {
                    error!("Failed to write to log file: {}", e);
                }
                if let Err(e) = writer.flush() {
                    error!("Failed to flush log file: {}", e);
                }
            }
            Err(e) => {
                error!("Failed to acquire log writer lock: {}", e);
            }
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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::ValidateConfig { config }) => {
            // Run in validation mode - no daemon, just validate config
            if let Err(e) = validate_config(&config) {
                eprintln!("Configuration validation failed: {}", e);
                std::process::exit(1);
            }
            Ok(())
        }
        Some(Commands::GenerateConfig { output }) => {
            // Generate default configuration file
            if let Err(e) = generate_config(&output) {
                eprintln!("Failed to generate config file: {}", e);
                std::process::exit(1);
            }
            Ok(())
        }
        None => {
            // Run daemon mode with default config path
            run_daemon(CONFIG_FILE).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> Config {
        Config {
            thermal_zone: "thermal_zone0".to_string(),
            predict_ahead_sec: 2.0,
            temp_full_speed: 70,
            temp_steep_start: 85,
            temp_emergency: 95,
            granularity_khz: 100_000,
            min_freq_ratio: 0.60,
            mid_freq_ratio: 0.72,
        }
    }

    fn create_test_freq_limits() -> FreqLimits {
        FreqLimits {
            max_khz: 4_000_000,
            mid_khz: 2_880_000,
            min_khz: 2_400_000,
        }
    }

    #[test]
    fn test_config_default_values() {
        let config = Config::default();
        assert_eq!(config.thermal_zone, "thermal_zone6");
        assert_eq!(config.predict_ahead_sec, 2.0);
        assert_eq!(config.temp_full_speed, 70);
        assert_eq!(config.temp_steep_start, 85);
        assert_eq!(config.temp_emergency, 95);
        assert_eq!(config.granularity_khz, 100_000);
        assert_eq!(config.min_freq_ratio, 0.60);
        assert_eq!(config.mid_freq_ratio, 0.72);
    }

    #[test]
    fn test_freq_limits_calculation() {
        let config = create_test_config();
        let limits = create_freq_limits(&config).unwrap();

        // Verify max frequency is read from system (should be > 0)
        assert!(limits.max_khz > 0);

        // Verify ratios are applied correctly
        let expected_min = (limits.max_khz as f64 * config.min_freq_ratio).round() as u64;
        let expected_mid = (limits.max_khz as f64 * config.mid_freq_ratio).round() as u64;

        assert_eq!(limits.min_khz, expected_min);
        assert_eq!(limits.mid_khz, expected_mid);

        // Verify ordering: max > mid > min
        assert!(limits.max_khz > limits.mid_khz);
        assert!(limits.mid_khz > limits.min_khz);
    }

    #[test]
    fn test_calculate_target_freq_full_speed() {
        let limits = create_test_freq_limits();
        let config = create_test_config();

        // Below full speed threshold should return max frequency
        let result = calculate_target_freq(60, &limits, &config);
        assert_eq!(result, limits.max_khz);
    }

    #[test]
    fn test_calculate_target_freq_linear_throttle() {
        let limits = create_test_freq_limits();
        let config = create_test_config();

        // Middle of linear range (77.5°C)
        let result = calculate_target_freq(77, &limits, &config);
        // Should be between max and mid
        assert!(result < limits.max_khz);
        assert!(result >= limits.mid_khz);
    }

    #[test]
    fn test_calculate_target_freq_steep_throttle() {
        let limits = create_test_freq_limits();
        let config = create_test_config();

        // Middle of steep range (90°C)
        let result = calculate_target_freq(90, &limits, &config);
        // Should be between mid and min
        assert!(result < limits.mid_khz);
        assert!(result >= limits.min_khz);
    }

    #[test]
    fn test_calculate_target_freq_emergency() {
        let limits = create_test_freq_limits();
        let config = create_test_config();

        // At emergency threshold
        let result = calculate_target_freq(95, &limits, &config);
        assert_eq!(result, limits.min_khz);

        // Above emergency threshold
        let result = calculate_target_freq(100, &limits, &config);
        assert_eq!(result, limits.min_khz);
    }

    #[test]
    fn test_calculate_target_freq_threshold_boundaries() {
        let limits = create_test_freq_limits();
        let config = create_test_config();

        // At full speed threshold
        let result = calculate_target_freq(70, &limits, &config);
        assert_eq!(result, limits.max_khz);

        // At steep start threshold
        let result = calculate_target_freq(85, &limits, &config);
        assert_eq!(result, limits.mid_khz);

        // At emergency threshold
        let result = calculate_target_freq(95, &limits, &config);
        assert_eq!(result, limits.min_khz);
    }

    #[test]
    fn test_quantize_freq_rounding() {
        let limits = create_test_freq_limits();
        let config = create_test_config();

        // Test rounding to nearest 100MHz
        let cases = vec![
            (3_950_000, 4_000_000),  // Round up
            (3_949_999, 3_900_000),  // Round down
            (3_850_000, 3_900_000),  // Round up (midpoint)
            (3_849_999, 3_800_000),  // Round down
            (2_450_000, 2_500_000),  // Not a multiple, round to nearest
        ];

        for (input, expected) in cases {
            let result = quantize_freq(input, &limits, &config);
            assert_eq!(result, expected, "Failed for input {}", input);
        }
    }

    #[test]
    fn test_quantize_freq_minimum_clamp() {
        let limits = create_test_freq_limits();
        let config = create_test_config();

        // Should clamp to minimum frequency
        let result = quantize_freq(1_000_000, &limits, &config);
        assert_eq!(result, limits.min_khz);

        let result = quantize_freq(0, &limits, &config);
        assert_eq!(result, limits.min_khz);
    }

    #[test]
    fn test_state_default_values() {
        let state = State::default();
        assert_eq!(state.k_slope, 0.5);
        assert_eq!(state.k_load, 0.02);
        assert!(state.error_history.is_empty());
    }

    #[test]
    fn test_state_serialization() {
        let state = State {
            k_slope: 0.75,
            k_load: 0.035,
            error_history: vec![0.1, -0.2, 0.3, 0.0],
        };

        // Serialize to JSON
        let json = serde_json::to_string(&state).unwrap();

        // Deserialize back
        let deserialized: State = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.k_slope, 0.75);
        assert_eq!(deserialized.k_load, 0.035);
        assert_eq!(deserialized.error_history, vec![0.1, -0.2, 0.3, 0.0]);
    }

    #[test]
    fn test_config_from_toml() {
        let toml_str = r#"
            thermal_zone = "thermal_zone0"
            predict_ahead_sec = 3.0
            temp_full_speed = 65
            temp_steep_start = 80
            temp_emergency = 90
            granularity_khz = 50000
            min_freq_ratio = 0.50
            mid_freq_ratio = 0.70
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();

        assert_eq!(config.thermal_zone, "thermal_zone0");
        assert_eq!(config.predict_ahead_sec, 3.0);
        assert_eq!(config.temp_full_speed, 65);
        assert_eq!(config.temp_steep_start, 80);
        assert_eq!(config.temp_emergency, 90);
        assert_eq!(config.granularity_khz, 50_000);
        assert_eq!(config.min_freq_ratio, 0.50);
        assert_eq!(config.mid_freq_ratio, 0.70);
    }

    #[test]
    fn test_config_partial_toml_uses_defaults() {
        let toml_str = r#"
            thermal_zone = "thermal_zone1"
            temp_full_speed = 75
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();

        // Explicit values
        assert_eq!(config.thermal_zone, "thermal_zone1");
        assert_eq!(config.temp_full_speed, 75);

        // Default values
        assert_eq!(config.predict_ahead_sec, 2.0);
        assert_eq!(config.temp_steep_start, 85);
        assert_eq!(config.min_freq_ratio, 0.60);
    }

    #[test]
    fn test_freq_ratios_within_bounds() {
        let config = Config::default();

        // Verify frequency ratios are within reasonable bounds
        assert!(config.min_freq_ratio >= 0.5 && config.min_freq_ratio <= 0.8);
        assert!(config.mid_freq_ratio >= config.min_freq_ratio);
        assert!(config.mid_freq_ratio <= 0.9);

        // Verify mid is between min and 1.0
        assert!(config.mid_freq_ratio > config.min_freq_ratio);
        assert!(config.mid_freq_ratio < 1.0);
    }

    #[test]
    fn test_temperature_thresholds_ordering() {
        let config = Config::default();

        // Verify temperature thresholds are in correct order
        assert!(config.temp_full_speed < config.temp_steep_start);
        assert!(config.temp_steep_start < config.temp_emergency);

        // Verify reasonable temperature ranges (°C)
        assert!(config.temp_full_speed >= 50 && config.temp_full_speed <= 80);
        assert!(config.temp_steep_start >= 70 && config.temp_steep_start <= 90);
        assert!(config.temp_emergency >= 85 && config.temp_emergency <= 100);
    }

    #[test]
    fn test_prediction_horizon_positive() {
        let config = Config::default();

        // Prediction horizon should be positive
        assert!(config.predict_ahead_sec > 0.0);
        assert!(config.predict_ahead_sec <= 10.0); // Reasonable upper bound
    }

    #[test]
    fn test_granularity_reasonable() {
        let config = Config::default();

        // Granularity should be reasonable (between 10kHz and 1MHz)
        assert!(config.granularity_khz >= 10_000);
        assert!(config.granularity_khz <= 1_000_000);
    }

    #[test]
    fn test_calculate_target_freq_monotonic() {
        let limits = create_test_freq_limits();
        let config = create_test_config();

        // Frequency should decrease monotonically as temperature increases
        let temps = vec![60, 70, 75, 80, 85, 90, 95, 100];
        let mut prev_freq = calculate_target_freq(temps[0], &limits, &config);

        for temp in temps.iter().skip(1) {
            let freq = calculate_target_freq(*temp, &limits, &config);
            assert!(freq <= prev_freq, "Frequency not monotonic at {}°C", temp);
            prev_freq = freq;
        }
    }

    #[test]
    fn test_freq_limits_debug_impl() {
        let limits = create_test_freq_limits();
        let debug_str = format!("{:?}", limits);
        assert!(debug_str.contains("max_khz"));
        assert!(debug_str.contains("mid_khz"));
        assert!(debug_str.contains("min_khz"));
    }

    #[test]
    fn test_freq_limits_clone() {
        let limits1 = create_test_freq_limits();
        let limits2 = limits1.clone();

        assert_eq!(limits1.max_khz, limits2.max_khz);
        assert_eq!(limits1.mid_khz, limits2.mid_khz);
        assert_eq!(limits1.min_khz, limits2.min_khz);
    }

    // CLI validation tests
    #[test]
    fn test_validate_config_valid_toml() {
        let toml_content = r#"
            thermal_zone = "thermal_zone0"
            predict_ahead_sec = 2.5
            temp_full_speed = 65
            temp_steep_start = 80
            temp_emergency = 90
            granularity_khz = 50000
            min_freq_ratio = 0.55
            mid_freq_ratio = 0.75
        "#;

        let temp_path = "/tmp/test_valid_config.toml";
        fs::write(temp_path, toml_content).unwrap();

        let result = validate_config(temp_path);
        assert!(result.is_ok(), "Valid config should pass validation");

        // Cleanup
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_validate_config_invalid_temperature_ordering() {
        let toml_content = r#"
            temp_full_speed = 90
            temp_steep_start = 85
            temp_emergency = 95
        "#;

        let temp_path = "/tmp/test_invalid_temp_config.toml";
        fs::write(temp_path, toml_content).unwrap();

        let result = validate_config(temp_path);
        assert!(result.is_err(), "Invalid temperature ordering should fail");
        assert!(result.unwrap_err().to_string().contains("temp_full_speed"));

        // Cleanup
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_validate_config_invalid_freq_ratios() {
        let toml_content = r#"
            min_freq_ratio = 0.80
            mid_freq_ratio = 0.70
        "#;

        let temp_path = "/tmp/test_invalid_ratio_config.toml";
        fs::write(temp_path, toml_content).unwrap();

        let result = validate_config(temp_path);
        assert!(result.is_err(), "Invalid frequency ratio ordering should fail");
        assert!(result.unwrap_err().to_string().contains("min_freq_ratio"));

        // Cleanup
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_validate_config_invalid_predict_horizon() {
        let toml_content = r#"
            predict_ahead_sec = -1.0
        "#;

        let temp_path = "/tmp/test_invalid_horizon_config.toml";
        fs::write(temp_path, toml_content).unwrap();

        let result = validate_config(temp_path);
        assert!(result.is_err(), "Invalid prediction horizon should fail");
        assert!(result.unwrap_err().to_string().contains("predict_ahead_sec"));

        // Cleanup
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_validate_config_mid_ratio_exceeds_one() {
        let toml_content = r#"
            mid_freq_ratio = 1.0
        "#;

        let temp_path = "/tmp/test_mid_ratio_one_config.toml";
        fs::write(temp_path, toml_content).unwrap();

        let result = validate_config(temp_path);
        assert!(result.is_err(), "mid_freq_ratio >= 1.0 should fail");
        assert!(result.unwrap_err().to_string().contains("mid_freq_ratio"));

        // Cleanup
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_validate_config_nonexistent_file() {
        let result = validate_config("/nonexistent/path/to/config.toml");
        assert!(result.is_err(), "Nonexistent file should fail");
        assert!(result.unwrap_err().to_string().contains("Failed to read"));
    }

    #[test]
    fn test_validate_config_invalid_toml_syntax() {
        let invalid_toml = r#"
            thermal_zone = "thermal_zone0
            # Missing closing quote above
            predict_ahead_sec = 2.0
        "#;

        let temp_path = "/tmp/test_invalid_syntax_config.toml";
        fs::write(temp_path, invalid_toml).unwrap();

        let result = validate_config(temp_path);
        assert!(result.is_err(), "Invalid TOML syntax should fail");
        assert!(result.unwrap_err().to_string().contains("Failed to parse"));

        // Cleanup
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_validate_config_default_values() {
        let temp_path = "/tmp/test_default_config.toml";
        fs::write(temp_path, "").unwrap();

        let result = validate_config(temp_path);
        assert!(result.is_ok(), "Empty config should use defaults and pass validation");

        // Cleanup
        let _ = fs::remove_file(temp_path);
    }

    // CLI generate-config tests
    #[test]
    fn test_generate_config_creates_file() {
        let temp_path = "/tmp/test_gen_config.toml";

        // Ensure file doesn't exist
        let _ = fs::remove_file(temp_path);

        let result = generate_config(temp_path);
        assert!(result.is_ok(), "generate_config should succeed");

        // Verify file was created
        assert!(Path::new(temp_path).exists(), "Config file should be created");

        // Verify content is valid TOML
        let content = fs::read_to_string(temp_path).unwrap();
        let config: Config = toml::from_str(&content).unwrap();
        assert_eq!(config.thermal_zone, "thermal_zone6");
        assert_eq!(config.temp_full_speed, 70);

        // Cleanup
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_generate_config_file_exists_error() {
        let temp_path = "/tmp/test_gen_config_exists.toml";

        // Create the file first
        fs::write(temp_path, "existing content").unwrap();

        let result = generate_config(temp_path);
        assert!(result.is_err(), "Should fail when file exists");
        assert!(result.unwrap_err().to_string().contains("already exists"));

        // Cleanup
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_generate_config_creates_directory() {
        let temp_path = "/tmp/test_cpu_throttle_nested/config.toml";

        // Ensure directory doesn't exist
        let _ = fs::remove_dir_all("/tmp/test_cpu_throttle_nested");

        let result = generate_config(temp_path);
        assert!(result.is_ok(), "generate_config should create parent directory");

        // Verify file was created
        assert!(Path::new(temp_path).exists(), "Config file should be created");

        // Verify content is valid TOML
        let content = fs::read_to_string(temp_path).unwrap();
        let config: Config = toml::from_str(&content).unwrap();
        assert_eq!(config.thermal_zone, "thermal_zone6");

        // Cleanup
        let _ = fs::remove_dir_all("/tmp/test_cpu_throttle_nested");
    }

    #[test]
    fn test_generate_config_default_values() {
        let temp_path = "/tmp/test_gen_config_defaults.toml";

        // Ensure file doesn't exist
        let _ = fs::remove_file(temp_path);

        generate_config(temp_path).unwrap();

        // Load and verify default values
        let content = fs::read_to_string(temp_path).unwrap();
        let config: Config = toml::from_str(&content).unwrap();

        let default = Config::default();
        assert_eq!(config.thermal_zone, default.thermal_zone);
        assert_eq!(config.predict_ahead_sec, default.predict_ahead_sec);
        assert_eq!(config.temp_full_speed, default.temp_full_speed);
        assert_eq!(config.temp_steep_start, default.temp_steep_start);
        assert_eq!(config.temp_emergency, default.temp_emergency);
        assert_eq!(config.granularity_khz, default.granularity_khz);
        assert_eq!(config.min_freq_ratio, default.min_freq_ratio);
        assert_eq!(config.mid_freq_ratio, default.mid_freq_ratio);

        // Cleanup
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_generate_config_has_comments() {
        let temp_path = "/tmp/test_gen_config_comments.toml";

        // Ensure file doesn't exist
        let _ = fs::remove_file(temp_path);

        generate_config(temp_path).unwrap();

        // Verify file has header comments
        let content = fs::read_to_string(temp_path).unwrap();
        assert!(content.contains("# cpu-throttle Configuration File"));
        assert!(content.contains("# Generated by cpu-th"));
        assert!(content.contains("# Run: ls /sys/class/thermal/"));

        // Cleanup
        let _ = fs::remove_file(temp_path);
    }
}

