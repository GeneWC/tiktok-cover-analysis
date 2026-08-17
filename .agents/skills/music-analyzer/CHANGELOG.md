# Changelog

All notable changes to this project will be documented in this file.

## [1.0.1] - 2026-04-04

### Changed
- Output never includes artist names, band names, album names, song titles, or lyrics quotes

## [1.0.0] - 2026-04-04

### Added
- Comprehensive music analysis pipeline with HDEMUCS vocal separation, Whisper lyrics transcription, and librosa audio feature extraction
- Lyrics analysis: summary, moods, themes, language detection, explicit content check
- Music analysis: genre/subgenre classification, mood detection, instrument identification, BPM & key detection, vocal description
- Production description: detailed prose describing drums, bass, harmony, melody, vocals, and mix characteristics
- CUDA GPU acceleration with automatic CPU fallback
- Chunked processing for long audio files (>60s) with proper overlap handling
- Dependency check on startup with actionable error messages and install commands
- File format validation (MP3, WAV, FLAC, M4A, OGG) and size limit (500MB)
- Genre reference profiles for 17 genres with subgenre indicators, mood mapping, and instrument detection hints
- Cross-platform support (Windows, macOS, Linux)
- MIT License
