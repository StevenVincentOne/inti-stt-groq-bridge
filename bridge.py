import asyncio
import base64
import io
import json
import logging
import os
import wave
from typing import Optional

import aiohttp
import numpy as np
import websockets
from aiohttp import ClientSession
from websockets.server import serve
from websockets.http import Headers
import opuslib

logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("KYUTAI_LLM_API_KEY")
GROQ_STT_URL = os.environ.get("GROQ_STT_URL", "https://api.groq.com/openai/v1/audio/transcriptions")
GROQ_STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")
SAMPLE_RATE = 24000

# Health via websockets.process_request on the same 8080 port
async def process_request(path: str, request_headers: Headers):
    if path == "/api/build_info":
        body = b'{"status":"ok","service":"stt-ws-groq-proxy"}'
        headers = [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
        ]
        return (200, headers, body)
    return None

class OpusDecoder:
    def __init__(self, sample_rate: int = 24000, channels: int = 1):
        self.decoder = opuslib.Decoder(sample_rate, channels)
        self.sample_rate = sample_rate
        self.channels = channels
        
    def decode_frame(self, opus_data: bytes) -> np.ndarray:
        """Decode a single Opus frame to PCM16"""
        try:
            # Decode opus frame to PCM data
            pcm_data = self.decoder.decode(opus_data, decode_fec=False)
            # Convert bytes to numpy array (PCM16)
            pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
            return pcm_array
        except Exception as e:
            logging.warning(f"Failed to decode Opus frame: {e}")
            return np.array([], dtype=np.int16)

class AudioBuffer:
    def __init__(self):
        self._buf = io.BytesIO()
        self.opus_decoder = OpusDecoder(SAMPLE_RATE, 1)
        self.pcm_collector: list[np.ndarray] = []

    def append_bytes(self, data: bytes) -> None:
        self._buf.write(data)

    def append_opus_frame(self, opus_data: bytes) -> None:
        """Decode and append Opus frame data"""
        pcm_data = self.opus_decoder.decode_frame(opus_data)
        if len(pcm_data) > 0:
            self.pcm_collector.append(pcm_data)

    def get_pcm_data(self) -> bytes:
        """Get accumulated PCM data"""
        if self.pcm_collector:
            # Concatenate all PCM arrays and return as bytes
            pcm_array = np.concatenate(self.pcm_collector)
            self.pcm_collector.clear()
            return pcm_array.tobytes()
        else:
            # Fallback to raw buffer data (for non-Opus legacy support)
            raw = self.reset_and_get()
            return to_pcm16_from_unknown(raw)

    def reset_and_get(self) -> bytes:
        data = self._buf.getvalue()
        self._buf.seek(0)
        self._buf.truncate(0)
        return data

    def clear(self) -> None:
        """Clear all buffers"""
        self.pcm_collector.clear()
        self._buf.seek(0)
        self._buf.truncate(0)

async def transcribe(session: ClientSession, wav_bytes: bytes) -> Optional[str]:
    data = aiohttp.FormData()
    data.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
    data.add_field("model", GROQ_STT_MODEL)
    data.add_field("response_format", "json")
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    async with session.post(GROQ_STT_URL, headers=headers, data=data, timeout=aiohttp.ClientTimeout(total=90)) as resp:
        if resp.status != 200:
            logging.error("Groq STT failed: %s %s", resp.status, await resp.text())
            return None
        body = await resp.json()
        return body.get("text")

def pcm16_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return out.getvalue()

def to_pcm16_from_unknown(data: bytes) -> bytes:
    """Fallback function for non-Opus data (legacy support)"""
    # Heuristic: if buffer length divisible by 4, assume float32 little-endian
    if len(data) % 4 == 0 and len(data) >= 4:
        try:
            float_array = np.frombuffer(data, dtype=np.float32)
            float_array = np.clip(float_array, -1.0, 1.0)
            int16 = (float_array * 32767.0).astype(np.int16)
            return int16.tobytes()
        except Exception:
            pass
    # Otherwise assume PCM16 already
    return data

async def handle_ws(websocket):
    buf = AudioBuffer()
    
    logging.info("New WebSocket connection established")

    async with aiohttp.ClientSession() as session:
        try:
            while True:
                msg = await websocket.recv()
                if isinstance(msg, bytes):
                    # Legacy binary frame support
                    buf.append_bytes(msg)
                    continue
                    
                try:
                    payload = json.loads(msg)
                except Exception as e:
                    logging.warning(f"Failed to parse JSON message: {e}")
                    continue

                t = payload.get("type")
                logging.debug(f"Received message type: {t}")
                
                if t in ("InputAudioBufferAppend", "input_audio_buffer.append", "append"):
                    b64_audio = payload.get("audio") or payload.get("data")
                    if b64_audio:
                        try:
                            # Decode base64 to get Opus data
                            opus_data = base64.b64decode(b64_audio)
                            # Decode Opus to PCM and buffer it
                            buf.append_opus_frame(opus_data)
                            logging.debug(f"Processed Opus frame: {len(opus_data)} bytes")
                        except Exception as e:
                            logging.warning(f"Failed to process audio frame: {e}")

                elif t in ("end", "end_audio", "InputAudioBufferCommit", "input_audio_buffer.commit"):
                    logging.info("Processing audio commit/end event")
                    
                    # Get accumulated PCM data
                    pcm_bytes = buf.get_pcm_data()
                    
                    if not pcm_bytes:
                        logging.warning("No audio data to transcribe")
                        await websocket.send(json.dumps({"type": "error", "message": "no audio"}))
                        continue

                    logging.info(f"Transcribing {len(pcm_bytes)} bytes of PCM data")
                    
                    # Convert to WAV and send to Groq
                    wav_bytes = pcm16_wav_bytes(pcm_bytes, SAMPLE_RATE)
                    text = await transcribe(session, wav_bytes)
                    
                    if text is None:
                        logging.error("Transcription failed")
                        await websocket.send(json.dumps({"type": "error", "message": "stt failed"}))
                    else:
                        logging.info(f"Transcription successful: {text[:50]}...")
                        await websocket.send(json.dumps({"type": "transcription", "text": text}))
                    
                    # Clear buffers for next audio session
                    buf.clear()

                else:
                    # ignore other messages
                    logging.debug(f"Ignoring message type: {t}")
                    
        except websockets.exceptions.ConnectionClosed:
            logging.info("WebSocket connection closed")
        except Exception as e:
            logging.error(f"Error in WebSocket handler: {e}")

async def main():
    logging.info("STT WS bridge with Opus decoding starting...")
    async with serve(handle_ws, "0.0.0.0", 8080, ping_interval=20, ping_timeout=20, process_request=process_request):
        logging.info("STT WS bridge listening on :8080")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())