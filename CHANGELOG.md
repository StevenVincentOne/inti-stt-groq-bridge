# Changelog

All notable changes to the Inti STT Groq Bridge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-09-10

### Added
- Initial production release of STT WebSocket-to-HTTP bridge
- Full Opus audio decoding support via `opuslib` and native libraries
- Base64-encoded Opus audio processing from Unmute service
- PCM16/float32 fallback support for legacy audio formats
- Integration with Groq Whisper API (`whisper-large-v3-turbo`)
- HTTP health endpoint at `/api/build_info`
- WebSocket protocol for real-time audio transcription
- Docker container with all dependencies pre-installed
- Comprehensive test suite with 3/3 tests passing:
  - Health endpoint validation
  - Direct Groq API integration test  
  - End-to-end WebSocket STT pipeline test
- Production logging with connection tracking and metrics
- Error handling and proper connection management
- Docker Swarm service deployment support

### Technical Details
- Python 3.12 runtime
- WebSocket server on port 8080
- Support for 24kHz mono audio processing
- Base64 audio payload decoding
- WAV format conversion for API compatibility
- Async/await architecture for high performance

### Dependencies
- `websockets==12.0` - WebSocket server implementation
- `aiohttp==3.9.5` - HTTP client for Groq API calls
- `numpy==2.1.1` - Audio data processing
- `opuslib==3.0.1` - Opus audio codec support
- System libraries: `libopus0`, `libopus-dev`

### Deployment
- Docker image: `intellipedia/inti-stt-groq-bridge:v1.0.0`
- Tested on Docker Swarm with overlay networking
- Production deployment validated on TensorDock infrastructure
- Integration tested with Unmute service pipeline

### Testing
- Automated test suite covering all major functionality
- WebSocket connectivity validation
- Audio processing pipeline verification  
- Groq API integration confirmation
- Health endpoint monitoring validation

### Documentation
- Complete implementation guide
- Deployment procedures
- Troubleshooting documentation  
- API reference with examples
- Docker Swarm integration instructions