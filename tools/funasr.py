import array
import asyncio
import io
import json
import logging
import queue
import ssl as ssl_module
import threading
import wave
from collections.abc import Generator
from typing import Any

import httpx
import websockets
import miniaudio
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.file.file import File
from dify_plugin.file.entities import FileType

logger = logging.getLogger(__name__)

CHUNK_SIZE = [5, 10, 5]
CHUNK_INTERVAL = 10

_SENTINEL = None

LEFT_LABEL = "left_channel"
RIGHT_LABEL = "right_channel"


class FunASRTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        host = tool_parameters.get("host")
        if not host:
            raise ValueError("Please fill in the FunASR server host")

        port = int(tool_parameters.get("port", 10095))
        use_ssl = int(tool_parameters.get("ssl", 0)) == 1
        use_itn = int(tool_parameters.get("use_itn", 1)) == 1
        hotword = tool_parameters.get("hotword") or ""
        separate_channels = int(tool_parameters.get("separate_channels", 0)) == 1

        audio_file = tool_parameters.get("audio_file")
        print(f"audio_file: {audio_file}")
        if not isinstance(audio_file, File):
            raise ValueError("Invalid audio content format. Expected File object.")
        if audio_file.type != FileType.AUDIO:
            raise ValueError(f"Invalid file type: {audio_file.type}. Expected audio file.")

        response = httpx.get(audio_file.url, verify=False)
        response.raise_for_status()
        raw_bytes = response.content
        # raw_bytes = audio_file.blo
        if not raw_bytes:
            raise ValueError("Audio file is empty.")

        extension = (audio_file.extension or "").lower().lstrip(".")
        wav_name = audio_file.filename or "dify_audio"

        if separate_channels:
            wav_bytes = self._ensure_wav(raw_bytes, extension)
            yield from self._invoke_stereo(
                host=host,
                port=port,
                raw_wav_bytes=wav_bytes,
                wav_name=wav_name,
                use_ssl=use_ssl,
                use_itn=use_itn,
                hotword=hotword,
            )
        else:
            yield from self._invoke_mono(
                host=host,
                port=port,
                raw_bytes=raw_bytes,
                extension=extension,
                wav_name=wav_name,
                use_ssl=use_ssl,
                use_itn=use_itn,
                hotword=hotword,
            )

    @staticmethod
    def _ensure_wav(raw_bytes: bytes, extension: str) -> bytes:
        """Convert audio bytes of any supported format to WAV.

        If the input is already WAV, return as-is. Otherwise use miniaudio
        (bundled decoders, no ffmpeg needed) to decode and re-encode as
        16-bit PCM WAV. Supports MP3, FLAC, WAV, and Vorbis.
        """
        if extension == "wav":
            return raw_bytes

        decoded = miniaudio.decode(raw_bytes, output_format=miniaudio.SampleFormat.SIGNED16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(decoded.nchannels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(decoded.sample_rate)
            wf.writeframes(decoded.samples)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Mono (single-channel) transcription
    # ------------------------------------------------------------------

    def _invoke_mono(
        self,
        host: str,
        port: int,
        raw_bytes: bytes,
        extension: str,
        wav_name: str,
        use_ssl: bool,
        use_itn: bool,
        hotword: str,
    ) -> Generator[ToolInvokeMessage]:
        sample_rate = 16000
        wav_format = "pcm"
        audio_bytes = raw_bytes

        if extension == "pcm":
            wav_format = "pcm"
        elif extension == "wav":
            wav_format = "pcm"
            with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                sample_rate = wf.getframerate()
                audio_bytes = wf.readframes(wf.getnframes())
        else:
            wav_format = "others"

        result_queue: queue.Queue[str | None] = queue.Queue()
        error_holder: list[BaseException] = []

        def _run():
            try:
                asyncio.run(
                    FunASRTool._transcribe(
                        host=host,
                        port=port,
                        audio_bytes=audio_bytes,
                        sample_rate=sample_rate,
                        wav_format=wav_format,
                        wav_name=wav_name,
                        use_ssl=use_ssl,
                        use_itn=use_itn,
                        hotword=hotword,
                        result_queue=result_queue,
                    )
                )
            except Exception as exc:
                error_holder.append(exc)
            finally:
                result_queue.put(_SENTINEL)

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

        for text in iter(result_queue.get, _SENTINEL):
            yield self.create_text_message(text)

        worker.join()
        if error_holder:
            raise error_holder[0]

    # ------------------------------------------------------------------
    # Stereo channel separation
    # ------------------------------------------------------------------

    def _invoke_stereo(
        self,
        host: str,
        port: int,
        raw_wav_bytes: bytes,
        wav_name: str,
        use_ssl: bool,
        use_itn: bool,
        hotword: str,
    ) -> Generator[ToolInvokeMessage]:
        left_pcm, right_pcm, sample_rate = self._split_stereo_wav(raw_wav_bytes)

        left_segments: list[dict] = []
        right_segments: list[dict] = []
        error_holder: list[BaseException] = []

        def _run_channel(pcm_data: bytes, out: list[dict], name: str):
            try:
                result = asyncio.run(
                    FunASRTool._transcribe_collect(
                        host=host,
                        port=port,
                        audio_bytes=pcm_data,
                        sample_rate=sample_rate,
                        wav_name=name,
                        use_ssl=use_ssl,
                        use_itn=use_itn,
                        hotword=hotword,
                    )
                )
                out.extend(result)
            except Exception as exc:
                error_holder.append(exc)

        t_left = threading.Thread(
            target=_run_channel,
            args=(left_pcm, left_segments, f"{wav_name}_L"),
            daemon=True,
        )
        t_right = threading.Thread(
            target=_run_channel,
            args=(right_pcm, right_segments, f"{wav_name}_R"),
            daemon=True,
        )
        t_left.start()
        t_right.start()
        t_left.join()
        t_right.join()

        if error_holder:
            raise error_holder[0]

        text = self._merge_and_format(left_segments, right_segments)
        yield self.create_text_message(text)

    @staticmethod
    def _split_stereo_wav(raw_wav_bytes: bytes) -> tuple[bytes, bytes, int]:
        """Split a stereo WAV file into left/right mono PCM byte streams.

        Returns (left_pcm, right_pcm, sample_rate).
        """
        with wave.open(io.BytesIO(raw_wav_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            if n_channels != 2:
                raise ValueError(
                    f"Expected stereo (2-channel) audio, got {n_channels} channel(s)."
                )
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if sample_width == 2:
            samples = array.array("h", frames)
            left = array.array("h", samples[0::2])
            right = array.array("h", samples[1::2])
            return left.tobytes(), right.tobytes(), sample_rate

        frame_size = sample_width * 2
        n_frames = len(frames) // frame_size
        left_buf = bytearray(n_frames * sample_width)
        right_buf = bytearray(n_frames * sample_width)
        for i in range(n_frames):
            so = i * frame_size
            do = i * sample_width
            left_buf[do : do + sample_width] = frames[so : so + sample_width]
            right_buf[do : do + sample_width] = frames[
                so + sample_width : so + frame_size
            ]
        return bytes(left_buf), bytes(right_buf), sample_rate

    @staticmethod
    async def _transcribe_collect(
        host: str,
        port: int,
        audio_bytes: bytes,
        sample_rate: int,
        wav_name: str,
        use_ssl: bool,
        use_itn: bool,
        hotword: str,
    ) -> list[dict]:
        """Run offline ASR and return all segments with timestamps.

        Each returned dict has at least ``text``; optionally ``start_ms``,
        ``end_ms``, and ``sentence_info``.
        """
        if use_ssl:
            ssl_context = ssl_module.SSLContext(ssl_module.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl_module.CERT_NONE
            uri = f"wss://{host}:{port}"
        else:
            ssl_context = None
            uri = f"ws://{host}:{port}"

        segments: list[dict] = []

        async with websockets.connect(
            uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context
        ) as ws:
            control_msg = json.dumps(
                {
                    "mode": "offline",
                    "chunk_size": CHUNK_SIZE,
                    "chunk_interval": CHUNK_INTERVAL,
                    "encoder_chunk_look_back": 4,
                    "decoder_chunk_look_back": 0,
                    "audio_fs": sample_rate,
                    "wav_name": wav_name,
                    "wav_format": "pcm",
                    "is_speaking": True,
                    "hotwords": hotword,
                    "itn": use_itn,
                },
                ensure_ascii=False,
            )
            await ws.send(control_msg)

            stride = int(
                60 * CHUNK_SIZE[1] / CHUNK_INTERVAL / 1000 * sample_rate * 2
            )
            chunk_num = max(1, (len(audio_bytes) - 1) // stride + 1)

            for i in range(chunk_num):
                beg = i * stride
                await ws.send(audio_bytes[beg : beg + stride])
                if i == chunk_num - 1:
                    await ws.send(
                        json.dumps({"is_speaking": False}, ensure_ascii=False)
                    )
                await asyncio.sleep(0.001)

            while True:
                resp = await asyncio.wait_for(ws.recv(), timeout=300)
                print(f"resp: {resp}")
                msg = json.loads(resp)
                if msg.get("mode") in ["offline", "2pass-offline"]:
                    text = msg.get("text", "")
                    if text:
                        seg: dict[str, Any] = {"text": text}

                        timestamp = msg.get("timestamp")
                        if isinstance(timestamp, str):
                            timestamp = json.loads(timestamp)
                        if timestamp and len(timestamp) > 0:
                            seg["start_ms"] = int(timestamp[0][0])
                            seg["end_ms"] = int(timestamp[-1][-1])

                        stamp_sents = msg.get("stamp_sents") or msg.get("sentence_info")
                        if stamp_sents:
                            seg["sentence_info"] = [
                                {
                                    "start": int(s.get("start", 0)),
                                    "end": int(s.get("end", 0)),
                                    "text": s.get("text") or (
                                        s.get("text_seg", "").replace(" ", "")
                                        + s.get("punc", "")
                                    ),
                                }
                                for s in stamp_sents
                            ]

                        segments.append(seg)

                if msg.get("mode") == "offline" or msg.get("is_final"):
                    break

        return segments

    @staticmethod
    def _merge_and_format(
        left_segments: list[dict],
        right_segments: list[dict],
    ) -> str:
        """Interleave ASR results from two channels by timestamp."""
        entries: list[dict] = []

        for label, segments in [
            (LEFT_LABEL, left_segments),
            (RIGHT_LABEL, right_segments),
        ]:
            for seg in segments:
                if "sentence_info" in seg:
                    for sent in seg["sentence_info"]:
                        start = int(sent.get("start", 0))
                        end = int(sent.get("end", start))
                        text = sent.get("text", "").strip()
                        if text:
                            entries.append(
                                {"start_ms": start, "end_ms": end, "text": text, "speaker": label}
                            )
                elif "start_ms" in seg:
                    entries.append(
                        {
                            "start_ms": seg["start_ms"],
                            "end_ms": seg.get("end_ms", seg["start_ms"]),
                            "text": seg["text"].strip(),
                            "speaker": label,
                        }
                    )
                else:
                    entries.append(
                        {"start_ms": -1, "text": seg["text"].strip(), "speaker": label}
                    )

        has_timestamps = any(e["start_ms"] >= 0 for e in entries)

        if has_timestamps:
            entries.sort(key=lambda e: (e["start_ms"], e["speaker"]))

        merged: list[dict] = []
        for entry in entries:
            if not entry["text"]:
                continue
            if merged and merged[-1]["speaker"] == entry["speaker"]:
                merged[-1]["text"] += entry["text"]
                merged[-1]["end_ms"] = entry.get("end_ms", entry["start_ms"])
            else:
                e = dict(entry)
                e.setdefault("end_ms", e["start_ms"])
                merged.append(e)

        def _fmt_ts(ms: int) -> str:
            total_sec = ms // 1000
            return f"{total_sec // 60:02d}:{total_sec % 60:02d}"

        lines: list[str] = []
        for entry in merged:
            if has_timestamps and entry["start_ms"] >= 0:
                ts = f"[{_fmt_ts(entry['start_ms'])}-{_fmt_ts(entry['end_ms'])}]"
                lines.append(f"{ts} {entry['speaker']}: {entry['text']}")
            else:
                lines.append(f"{entry['speaker']}: {entry['text']}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Original single-channel transcription (streaming via queue)
    # ------------------------------------------------------------------

    @staticmethod
    async def _transcribe(
        host: str,
        port: int,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        wav_format: str = "pcm",
        wav_name: str = "dify_audio",
        use_ssl: bool = True,
        use_itn: bool = True,
        hotword: str = "",
        result_queue: queue.Queue | None = None,
    ) -> None:
        if use_ssl:
            ssl_context = ssl_module.SSLContext(ssl_module.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl_module.CERT_NONE
            uri = f"wss://{host}:{port}"
        else:
            ssl_context = None
            uri = f"ws://{host}:{port}"

        async with websockets.connect(
            uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context
        ) as ws:
            control_msg = json.dumps(
                {
                    "mode": "offline",
                    "chunk_size": CHUNK_SIZE,
                    "chunk_interval": CHUNK_INTERVAL,
                    "encoder_chunk_look_back": 4,
                    "decoder_chunk_look_back": 0,
                    "audio_fs": sample_rate,
                    "wav_name": wav_name,
                    "wav_format": wav_format,
                    "is_speaking": True,
                    "hotwords": hotword,
                    "itn": use_itn,
                },
                ensure_ascii=False,
            )
            await ws.send(control_msg)

            stride = int(60 * CHUNK_SIZE[1] / CHUNK_INTERVAL / 1000 * sample_rate * 2)
            chunk_num = max(1, (len(audio_bytes) - 1) // stride + 1)

            for i in range(chunk_num):
                beg = i * stride
                await ws.send(audio_bytes[beg : beg + stride])

                if i == chunk_num - 1:
                    await ws.send(
                        json.dumps({"is_speaking": False}, ensure_ascii=False)
                    )

                await asyncio.sleep(0.001)

            while True:
                resp = await asyncio.wait_for(ws.recv(), timeout=300)
                print(f"resp: {resp}")
                msg = json.loads(resp)
                if msg.get("mode") in ["offline", "2pass-offline"]:
                    if result_queue is not None:
                        text = msg.get("text", "")
                        result_queue.put(text)
                if msg.get("mode") == "offline" or msg.get("is_final"):
                    break
