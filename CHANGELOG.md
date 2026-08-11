# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- `RemoteModelWrapper` (`textattack.models.wrappers.RemoteModelWrapper`): query a model served behind a remote HTTP API instead of running it locally. Request/response handling is adaptable to different endpoint schemas via `request_fn`/`response_fn`.
