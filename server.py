"""Local web server and HTML playground backend for tiny_voca_fi_v4."""
import argparse
import base64
import io
import json
import math
import re
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from vocos import Vocos

from synthesize import AcousticToVocosMel, LIGHT_WORDS, master_audio
from tiny_voca.config import Config
from tiny_voca.infer import finished, limit_peak, sentences, shorter
from tiny_voca.model import TinyVoca, UtteranceTooLong
from tiny_voca.text import NothingToPronounce, PHONE_SET, SP, espeak_version, phonemize

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
DEFAULT_CKPT = BASE_DIR / "weights" / "joint_inference.pt"


class FinnishTTSEngine:
    def __init__(self, ckpt_path: Path):
        self.ckpt_path = Path(ckpt_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Mallin painotiedostoa ei löydy: {self.ckpt_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(self.ckpt_path, map_location=self.device, weights_only=True)

        self.t_mod = TinyVoca(Config.from_dict(state["config"])).to(self.device)
        self.t_mod.load_state_dict(state["tiny_model"])
        self.t_mod.eval()

        self.b_mod = AcousticToVocosMel().to(self.device)
        self.b_mod.load_state_dict(state["bridge"])
        self.b_mod.eval()

        self.voc_mod = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device)
        self.voc_mod.load_state_dict(state["vocos"])
        self.voc_mod.eval()

        self.speakers = list(self.t_mod.config.speakers)
        self.sample_rate = int(self.t_mod.config.sample_rate)
        self.step = state.get("step")
        self.lock = threading.Lock()
        try:
            self.espeak_info = espeak_version()
        except Exception as exc:
            self.espeak_info = f"Virhe: {exc}"

    def info(self) -> dict:
        device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        return {
            "model": "tiny_voca_fi_v4",
            "step": self.step,
            "checkpoint": self.ckpt_path.name,
            "speakers": self.speakers,
            "default_speaker": "M2" if "M2" in self.speakers else self.speakers[0],
            "sample_rate": self.sample_rate,
            "device": self.device.type,
            "device_name": device_name,
            "espeak_version": self.espeak_info,
        }

    def _synthesize_piece(
        self,
        raw_sent: str,
        spk_idx: int,
        speed: float,
        pitch_range: float,
        pitch_shift: float,
        wobble: float,
        end_taper: float,
        sentence_stress: bool,
    ):
        is_q = raw_sent.strip().endswith("?")
        words_clean = [re.sub(r"[^A-Za-zÄÖÅäöå0-9]", "", tok).lower() for tok in raw_sent.split()]
        ph = phonemize(raw_sent)
        if len(ph) > self.t_mod.config.max_tokens:
            raise UtteranceTooLong("liian pitkä lauseke")

        tokens = torch.tensor([ph.ids], device=self.device)
        stress = torch.tensor([ph.stress], device=self.device)
        lengths = torch.tensor([len(ph)], device=self.device)
        speakers = torch.tensor([spk_idx], device=self.device)

        hidden, mask = self.t_mod.encode(tokens, stress, lengths, speakers)
        durs = self.t_mod.predict_durations(hidden, mask, tokens, speed)

        light_tok = torch.zeros(len(ph), dtype=torch.bool, device=self.device)
        heavy_tok = torch.zeros(len(ph), dtype=torch.bool, device=self.device)
        if sentence_stress:
            w_idx, in_w = 0, False
            for i, sym in enumerate(ph.symbols):
                if sym in (",", ";", ":", "—"):
                    durs[0, i] = durs[0, i].clamp_min(10)
                elif sym in PHONE_SET:
                    in_w = True
                    if w_idx < len(words_clean):
                        w_str = words_clean[w_idx]
                        if w_str in LIGHT_WORDS:
                            light_tok[i] = True
                        elif len(w_str) >= 5:
                            heavy_tok[i] = True
                elif sym == SP and in_w:
                    w_idx += 1
                    in_w = False
            if light_tok.any():
                durs = torch.where(
                    light_tok[None, :] & (durs > 1),
                    torch.round(durs.float() * 0.90).long().clamp_min(1),
                    durs,
                )
            if heavy_tok.any():
                durs = torch.where(
                    heavy_tok[None, :] & (durs > 0),
                    torch.round(durs.float() * 1.05).long().clamp_min(1),
                    durs,
                )

        frames, f_mask, f_len = self.t_mod.expand(hidden, durs, mask, speakers)
        p_hid, prosody = self.t_mod.predict_prosody(frames, f_mask)
        ends = durs.cumsum(1)
        f_idx = torch.arange(frames.shape[1], device=self.device)[None]
        owner = torch.searchsorted(ends, f_idx, right=True).clamp(max=tokens.shape[1] - 1)
        vowel = self.t_mod.vowel_tokens[tokens].gather(1, owner) & f_mask
        prosody = torch.cat(
            (
                prosody[..., :1],
                torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])[..., None],
                prosody[..., 2:],
            ),
            dim=-1,
        )
        prosody = self.t_mod.widen_pitch(prosody, f_mask, pitch_range)

        prosody = prosody.clone()
        n_f = int(f_len[0].item())
        if pitch_shift != 0.0:
            prosody[..., 0] = prosody[..., 0] + pitch_shift * (math.log(2.0) / 12.0)

        if sentence_stress:
            light_f = light_tok.gather(0, owner[0])[None, :] & f_mask & vowel
            heavy_f = heavy_tok.gather(0, owner[0])[None, :] & f_mask & vowel
            prosody[..., 0] = torch.where(light_f, prosody[..., 0] - 0.25 * (math.log(2.0) / 12.0), prosody[..., 0])
            prosody[..., 0] = torch.where(heavy_f, prosody[..., 0] + 0.30 * (math.log(2.0) / 12.0), prosody[..., 0])

        if wobble > 0 and n_f > 8:
            noise = torch.randn(1, n_f, device=self.device)
            smooth = F.avg_pool1d(noise[:, None, :], kernel_size=7, stride=1, padding=3)[:, 0]
            smooth = smooth / smooth.std().clamp_min(1e-4)
            prosody[:, :n_f, 0] += smooth * (wobble * math.log(2.0) / 12.0) * vowel[:, :n_f].float()

        if n_f > 20 and end_taper != 0.0:
            tail = min(18, n_f // 4)
            delta = 0.7 if is_q else -end_taper
            ramp = torch.linspace(0.0, delta * (math.log(2.0) / 12.0), tail, device=self.device)
            prosody[0, n_f - tail : n_f, 0] += ramp

        acoustic = self.t_mod.decode(p_hid, self.t_mod.hard_prosody(prosody), f_mask)
        pred_mel = self.b_mod(acoustic)
        wave = self.voc_mod.decode(F.pad(pred_mel, (0, 1), mode="replicate"))[0, : n_f * 256].cpu()
        fade = min(600, len(wave) // 6)
        if fade > 0:
            wave[-fade:] *= torch.linspace(1.0, 0.0, fade)
        return wave, " ".join(ph.symbols)

    def _synthesize_with_split(self, piece: str, spk_idx: int, **kwargs):
        try:
            wave, ph_str = self._synthesize_piece(piece, spk_idx, **kwargs)
            return [(wave, ph_str)]
        except NothingToPronounce:
            return []
        except UtteranceTooLong:
            subpieces = shorter(piece)
            if not subpieces:
                raise ValueError(f"Yksittäinen sana on liian pitkä syntetisoitavaksi: {piece[:40]!r}")
            results = []
            for sub in subpieces:
                results.extend(self._synthesize_with_split(sub, spk_idx, **kwargs))
            return results

    @torch.inference_mode()
    def synthesize_wav(
        self,
        text: str,
        speaker: str = "M2",
        speed: float = 1.0,
        pitch_range: float = 1.25,
        pitch_shift: float = 0.0,
        wobble: float = 0.18,
        end_taper: float = 0.75,
        sentence_stress: bool = True,
        master: bool = True,
    ) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Kirjoita ensin syntetisoitava teksti.")
        if speaker not in self.speakers:
            raise ValueError(f"Tuntematon puhuja {speaker!r}. Sallitut: {', '.join(self.speakers)}")

        speed = max(0.5, min(2.0, float(speed)))
        pitch_range = max(0.0, min(3.0, float(pitch_range)))
        pitch_shift = max(-12.0, min(12.0, float(pitch_shift)))
        wobble = max(0.0, min(1.0, float(wobble)))
        end_taper = max(-2.0, min(2.0, float(end_taper)))

        spk_idx = self.speakers.index(speaker)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        waves = []
        phonemes_list = []

        t0 = time.perf_counter()
        with self.lock:
            for p_idx, para in enumerate(paragraphs):
                parts = sentences(finished(para))
                for s_idx, raw_sent in enumerate(parts):
                    outputs = self._synthesize_with_split(
                        raw_sent,
                        spk_idx,
                        speed=speed,
                        pitch_range=pitch_range,
                        pitch_shift=pitch_shift,
                        wobble=wobble,
                        end_taper=end_taper,
                        sentence_stress=sentence_stress,
                    )
                    for wave, ph_str in outputs:
                        waves.append(wave)
                        phonemes_list.append(ph_str)
                    if outputs and s_idx < len(parts) - 1:
                        waves.append(torch.zeros(int(0.30 * self.sample_rate)))
                if p_idx < len(paragraphs) - 1 and waves:
                    waves.append(torch.zeros(int(0.55 * self.sample_rate)))

        if not waves:
            raise NothingToPronounce("Tekstistä ei löytynyt äännettäviä sanoja.")

        audio = master_audio(limit_peak(torch.cat(waves), 0.98).numpy(), self.sample_rate, master=master)
        gen_time_s = time.perf_counter() - t0
        duration_s = float(len(audio) / self.sample_rate)

        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV", subtype="PCM_16")
        wav_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "audio_base64": wav_b64,
            "sample_rate": self.sample_rate,
            "duration_s": round(duration_s, 3),
            "gen_time_s": round(gen_time_s, 3),
            "rtf": round(duration_s / max(gen_time_s, 1e-4), 2),
            "utterances": len(phonemes_list),
            "phonemes": phonemes_list,
        }


def make_handler(engine: FinnishTTSEngine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Concise server log
            print(f"[{self.address_string()}] {format % args}")

        def _send_json(self, status: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                html_path = WEB_DIR / "index.html"
                if not html_path.exists():
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "web/index.html puuttuu"})
                    return
                data = html_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return

            if path == "/api/info":
                self._send_json(HTTPStatus.OK, engine.info())
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Tuntematon reitti"})

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path != "/api/synthesize":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Tuntematon reitti"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                data = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Virheellinen JSON-pyyntö."})
                return

            try:
                result = engine.synthesize_wav(
                    text=str(data.get("text", "")),
                    speaker=str(data.get("speaker", "M2")),
                    speed=float(data.get("speed", 1.0)),
                    pitch_range=float(data.get("pitch_range", 1.25)),
                    pitch_shift=float(data.get("pitch_shift", 0.0)),
                    wobble=float(data.get("wobble", 0.18)),
                    end_taper=float(data.get("end_taper", 0.75)),
                    sentence_stress=bool(data.get("sentence_stress", True)),
                    master=bool(data.get("master", True)),
                )
                self._send_json(HTTPStatus.OK, result)
            except (ValueError, NothingToPronounce) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Synteesivirhe: {exc}"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Paikallinen HTML-palvelin tiny_voca_fi_v4 -mallille")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT, help="Polku joint_inference.pt -tiedostoon")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Avaa selain automaattisesti käynnistyksen jälkeen")
    args = parser.parse_args()

    print(f"Ladataan tiny_voca_fi_v4 ({args.ckpt})...")
    engine = FinnishTTSEngine(args.ckpt)
    info = engine.info()
    print(f"Malli valmis! Laite: {info['device_name']} ({info['device']}), puhujat: {', '.join(info['speakers'])}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    url = f"http://{args.host}:{args.port}"
    print(f"Käyttöliittymä käynnissä osoitteessa: {url}")
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSammutetaan palvelin...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
