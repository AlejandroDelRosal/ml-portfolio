# Changelog

Format based on Keep a Changelog (https://keepachangelog.com/en/1.1.0/).

## [Unreleased]
### Added
- Initial repository structure.
- Autograd engine: reverse-mode automatic differentiation and neural network library written from scratch, validated against PyTorch to machine precision.
- Time series forecasting model comparison on real US unemployment data, validated with the Diebold-Mariano test.
- Premier League forecasting: Poisson family goal models, Elo and pi-ratings compared against the bookmakers' closing line in a walk-forward backtest, with value betting simulated under a bootstrap confidence interval.
### Changed
- CI now discovers and tests each project subdirectory independently, including packaged projects declaring a pyproject.toml.
