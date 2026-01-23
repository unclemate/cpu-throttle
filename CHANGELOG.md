# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-01-23

### Added
- Log rotation on startup (KISS: truncate log file on daemon restart)
- Unit tests for log file opening behavior

### Changed
- Extract `open_log_file()` function following SOLID Single Responsibility Principle

---

## [0.1.0] - 2026-01-22

### Added
- CLI commands for configuration management (`validate-config`, `generate-config`)
- Configuration file support (TOML format at `/etc/cpu-throttle/config.toml`)
- Comprehensive unit tests for core functionality
- Graceful shutdown with SIGINT/SIGTERM signal handling
- Buffered writer for log file I/O (performance optimization)
- Dynamic frequency limits from system CPU info

### Fixed
- Improved timestamp matching robustness for prediction accuracy check
- Enhanced code robustness from code review findings

### Docs
- Updated frequency control strategy in README
- Updated GitHub repository URL

### Refactor
- Dynamic frequency limits calculation from system CPU info instead of hardcoded values

---

## Initial Release

### Features
- Predictive CPU thermal control daemon with feed-forward algorithm
- Adaptive self-tuning parameters (k_slope, k_load)
- State persistence across reboots
- Granular frequency control (100 MHz adjustment)
- 4-tier frequency control strategy based on effective temperature
