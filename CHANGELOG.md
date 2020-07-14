# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]


## [0.1.1] - 2020-07-15

### Changed

- All arguments passed to Scales are now `float`s and all Scale fields are assumed to be `float`s



## [0.1.0] - 2020-07-12

### Added

- `PointDensity()` distribution. We now primarily operate on point densities located in the center of what used to be the bins in our histogram.
- "denorm_xs_only" option on PointDensity egress methods. This returns normalized probability densities but on a denormalized x-axis, as Metaculus displays.
- support for LogScale and Metaculus/Elicit LogScale questions

### Changed

- We operate on 200 point densities at the center of bins evenly spaced from 0 to 1 on a normalized scale. If we are passed in anything besides this in from_pairs we interpolate to get 200 points placed in this manner.
- We always operate on normalized points and normalized densities internally, using the `normed_xs` and `normed_densities` fields respectively. We denormalize the density before returning in PDF.
- PointDensity.cdf(x) returns the cdf up to x (not the bin before x, as previously)

### Removed
- the `Histrogram()` distribution

