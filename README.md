# Inti STT Groq Bridge

A WebSocket-to-HTTP bridge that enables integration between Unmute's real-time audio pipeline and Groq's Whisper transcription API.

## Overview

This bridge accepts WebSocket connections from the Unmute service, processes Base64-encoded Opus audio data, converts it to WAV format, and forwards it to Groq's Whisper API for transcription. It returns transcription results via the same WebSocket connection.

## Features

✅ **Opus Audio Support** - Decodes Base64-encoded Opus audio from Unmute  
✅ **PCM Fallback** - Supports legacy PCM16/float32 audio formats  
✅ **Groq Whisper Integration** - Uses Groq's high-performance Whisper API  
✅ **Health Monitoring** - HTTP health endpoint for service discovery  
✅ **Production Ready** - Comprehensive testing with 3/3 tests passing  
✅ **Docker Swarm Compatible** - Designed for Docker Swarm deployments  

## Quick Start

### Docker Hub

```bash
docker pull intellipedia/inti-stt-groq-bridge:v1.0.0
```

### Environment Variables

- `GROQ_API_KEY` or `OPENAI_API_KEY`: Groq API key (required)
- `GROQ_STT_MODEL`: Whisper model (default: `whisper-large-v3-turbo`)
- `GROQ_STT_URL`: Groq API endpoint (default: `https://api.groq.com/openai/v1/audio/transcriptions`)

### Docker Compose Example

```yaml
version: '3.8'
services:
  stt-bridge:
    image: intellipedia/inti-stt-groq-bridge:v1.0.0
    ports:
      - "8080:8080"
    environment:
      - GROQ_API_KEY=your_groq_api_key_here
      - GROQ_STT_MODEL=whisper-large-v3-turbo
    networks:
      - unmute-net
```

### Docker Swarm Service

```bash
docker service create \
  --name unmute_unmute_stt \
  --network unmute-net \
  --env GROQ_API_KEY="your_api_key" \
  --env GROQ_STT_MODEL=whisper-large-v3-turbo \
  intellipedia/inti-stt-groq-bridge:v1.0.0
```

## API Reference

### WebSocket Protocol

**Connect**: `ws://localhost:8080`

**Input Messages**:
```json
// Primary: Opus audio (production mode)
{"type":"input_audio_buffer.append","audio":"<base64-opus-data>"}

// Trigger transcription
{"type":"input_audio_buffer.commit"}
```

**Output Messages**:
```json
// Success
{"type":"transcription","text":"transcribed text here"}

// Error
{"type":"error","message":"error description"}
```

### HTTP Health Check

**Endpoint**: `GET /api/build_info`

**Response**:
```json
{"status":"ok","service":"stt-ws-groq-proxy"}
```

## Architecture

```
Browser PWA → Unmute Service → STT Bridge → Groq Whisper API
                                     ↓
                          WebSocket Response ← Transcription Result
```

### Audio Processing Pipeline

1. **Receive**: Base64-encoded Opus audio via WebSocket
2. **Decode**: Opus → PCM16 using `opuslib` 
3. **Convert**: PCM16 → WAV format
4. **Transcribe**: Send WAV to Groq Whisper API
5. **Respond**: Return transcription via WebSocket

## Development

### Local Build

```bash
git clone https://github.com/intellipedia/inti-stt-groq-bridge.git
cd inti-stt-groq-bridge
docker build -t inti-stt-groq-bridge .
```

### Dependencies

- Python 3.12
- `websockets==12.0` - WebSocket server
- `aiohttp==3.9.5` - HTTP client for Groq API  
- `numpy==2.1.1` - Audio processing
- `opuslib==3.0.1` - Opus audio decoding
- System: `libopus0`, `libopus-dev` - Native Opus libraries

### Testing

The bridge includes a comprehensive test suite:

```bash
# Run automated tests
docker build -f Dockerfile.test -t stt-test .
docker run --rm --env GROQ_API_KEY="your_key" stt-test
```

**Test Coverage**:
- ✅ Health endpoint functionality
- ✅ Groq API integration
- ✅ End-to-end WebSocket STT pipeline

## Production Deployment

### System Requirements

- Docker Swarm or Docker Compose
- Access to Groq API (requires API key)
- Network connectivity for WebSocket clients

### Performance

- **Latency**: Typical transcription response < 2 seconds
- **Throughput**: Handles concurrent WebSocket connections
- **Audio Format**: 24kHz mono, Opus or PCM16/float32
- **Model**: Uses `whisper-large-v3-turbo` for optimal speed/quality balance

### Monitoring

- Health checks via HTTP endpoint
- Structured logging for connection tracking
- Audio processing metrics in logs
- Error handling with detailed error messages

## Configuration

### Unmute Integration

Ensure Unmute service points to the bridge:

```bash
# Set in Unmute service environment
KYUTAI_STT_URL=ws://unmute_unmute_stt:8080
```

### Docker Swarm Network

```bash
# Create overlay network if not exists
docker network create -d overlay unmute-net

# Deploy service on network
docker service create --network unmute-net ...
```

## Troubleshooting

**Connection Issues**:
- Verify WebSocket connectivity on port 8080
- Check Docker network configuration
- Ensure service DNS resolution working

**Audio Processing**:
- Confirm audio format is 24kHz mono
- Verify Opus libraries are installed
- Check Base64 encoding of audio data

**API Issues**:
- Validate Groq API key is correct
- Confirm internet connectivity to api.groq.com
- Check API rate limits and quotas

## License

This project is part of the Inti platform.

## Contributing

This bridge is designed specifically for the Inti/Unmute ecosystem. For issues or feature requests, please refer to the main Inti documentation.

## Version History

### v1.0.0 (2025-09-10)
- ✅ Initial production release
- ✅ Full Opus audio decoding support
- ✅ Comprehensive test suite (3/3 passing)
- ✅ Production deployment validated
- ✅ Docker Hub image available

---

For detailed deployment instructions and operational guidance, see: [STT-Groq-Bridge-Implementation.md](STT-Groq-Bridge-Implementation.md)