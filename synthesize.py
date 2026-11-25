"""Local inference script for Finnish tiny-voca + Vocos (v5, 65k steps)."""
import argparse
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, sosfiltfilt
from torch import nn
from torch.nn import functional as F
from vocos import Vocos

from tiny_voca.config import Config
from tiny_voca.infer import finished, limit_peak, sentences
from tiny_voca.model import TinyVoca
from tiny_voca.text import PHONE_SET, SP, phonemize

LIGHT_WORDS = {
    "ja", "on", "oli", "ole", "ei", "en", "et", "emme", "ette", "eivät",
    "että", "kun", "jos", "niin", "se", "sen", "sitä", "ne", "tämä", "tuo",
    "mutta", "vaan", "tai", "sekä", "myös", "vain", "jo", "vielä", "nyt",
    "sitten", "kuin", "joka", "jotka", "mikä", "mitä", "ovat", "olla", "olin",
}


class AcousticToVocosMel(nn.Module):
    def __init__(self, in_channels=192, mel_bins=100, hidden=256):
        super().__init__()
        self.proj_in = nn.Conv1d(in_channels, hidden, 5, padding=2)
        self.block1 = nn.Sequential(nn.GELU(), nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(), nn.Conv1d(hidden, hidden, 5, padding=2))
        self.block2 = nn.Sequential(nn.GELU(), nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(), nn.Conv1d(hidden, hidden, 5, padding=2))
        self.proj_out = nn.Conv1d(hidden, mel_bins, 1)

    def forward(self, acoustic):
        x = self.proj_in(acoustic.transpose(1, 2))
        x = x + self.block1(x)
        x = x + self.block2(x)
        return self.proj_out(x)


def master_audio(wave_np, sr=24000, master=True):
    if not master or len(wave_np) < 512:
        return wave_np
    w = wave_np.astype(np.float64)
    sos_low = butter(2, 220.0, btype="low", fs=sr, output="sos")
    sos_hi = butter(2, 6500.0, btype="high", fs=sr, output="sos")
    w = w + 0.14 * sosfiltfilt(sos_low, w) - 0.18 * sosfiltfilt(sos_hi, w)
    reverb = np.zeros_like(w)
    for delay_ms, gain in ((19, 0.018), (37, 0.012), (61, 0.008)):
        d = int(sr * delay_ms / 1000)
        if d < len(w):
            reverb[d:] += gain * w[:-d]
    w = w + reverb
    peak = float(np.max(np.abs(w)))
    if peak > 0.98:
        w = w * (0.98 / peak)
    return w.astype(np.float32)


@torch.inference_mode()
def synthesize(text, ckpt_path, out_path="output.wav", speed=1.0, pitch_range=1.25, wobble=0.18, end_taper=0.75, sentence_stress=True, master=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(ckpt_path, map_location=device, weights_only=True)
    t_mod = TinyVoca(Config.from_dict(st["config"])).to(device)
    t_mod.load_state_dict(st["tiny_model"])
    t_mod.eval()
    b_mod = AcousticToVocosMel().to(device)
    b_mod.load_state_dict(st["bridge"])
    b_mod.eval()
    voc_mod = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    voc_mod.load_state_dict(st["vocos"])
    voc_mod.eval()

    spk_idx = t_mod.config.speakers.index("M2")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    waves = []
    for p_idx, para in enumerate(paragraphs):
        parts = sentences(finished(para))
        for s_idx, raw_sent in enumerate(parts):
            is_q = raw_sent.strip().endswith("?")
            words_clean = [re.sub(r"[^A-Za-zÄÖÅäöå0-9]", "", tok).lower() for tok in raw_sent.split()]
            ph = phonemize(raw_sent)
            tokens = torch.tensor([ph.ids], device=device)
            stress = torch.tensor([ph.stress], device=device)
            lengths = torch.tensor([len(ph)], device=device)
            speakers = torch.tensor([spk_idx], device=device)
            hidden, mask = t_mod.encode(tokens, stress, lengths, speakers)
            durs = t_mod.predict_durations(hidden, mask, tokens, speed)

            light_tok = torch.zeros(len(ph), dtype=torch.bool, device=device)
            heavy_tok = torch.zeros(len(ph), dtype=torch.bool, device=device)
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
                    durs = torch.where(light_tok[None, :] & (durs > 1), torch.round(durs.float() * 0.90).long().clamp_min(1), durs)
                if heavy_tok.any():
                    durs = torch.where(heavy_tok[None, :] & (durs > 0), torch.round(durs.float() * 1.05).long().clamp_min(1), durs)

            frames, f_mask, f_len = t_mod.expand(hidden, durs, mask, speakers)
            p_hid, prosody = t_mod.predict_prosody(frames, f_mask)
            ends = durs.cumsum(1)
            f_idx = torch.arange(frames.shape[1], device=device)[None]
            owner = torch.searchsorted(ends, f_idx, right=True).clamp(max=tokens.shape[1] - 1)
            vowel = t_mod.vowel_tokens[tokens].gather(1, owner) & f_mask
            prosody = torch.cat((prosody[..., :1], torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])[..., None], prosody[..., 2:]), dim=-1)
            prosody = t_mod.widen_pitch(prosody, f_mask, pitch_range)

            prosody = prosody.clone()
            n_f = int(f_len[0].item())
            if sentence_stress:
                light_f = light_tok.gather(0, owner[0])[None, :] & f_mask & vowel
                heavy_f = heavy_tok.gather(0, owner[0])[None, :] & f_mask & vowel
                prosody[..., 0] = torch.where(light_f, prosody[..., 0] - 0.25 * (math.log(2.0) / 12.0), prosody[..., 0])
                prosody[..., 0] = torch.where(heavy_f, prosody[..., 0] + 0.30 * (math.log(2.0) / 12.0), prosody[..., 0])

            if wobble > 0 and n_f > 8:
                noise = torch.randn(1, n_f, device=device)
                smooth = F.avg_pool1d(noise[:, None, :], kernel_size=7, stride=1, padding=3)[:, 0]
                smooth = smooth / smooth.std().clamp_min(1e-4)
                prosody[:, :n_f, 0] += smooth * (wobble * math.log(2.0) / 12.0) * vowel[:, :n_f].float()
            if n_f > 20:
                tail = min(18, n_f // 4)
                delta = 0.7 if is_q else -end_taper
                ramp = torch.linspace(0.0, delta * (math.log(2.0) / 12.0), tail, device=device)
                prosody[0, n_f - tail:n_f, 0] += ramp

            acoustic = t_mod.decode(p_hid, t_mod.hard_prosody(prosody), f_mask)
            pred_mel = b_mod(acoustic)
            wave = voc_mod.decode(F.pad(pred_mel, (0, 1), mode="replicate"))[0, :n_f * 256].cpu()
            fade = min(600, len(wave) // 6)
            if fade > 0:
                wave[-fade:] *= torch.linspace(1.0, 0.0, fade)
            waves.append(wave)
            if s_idx < len(parts) - 1:
                waves.append(torch.zeros(int(0.30 * 24000)))
        if p_idx < len(paragraphs) - 1:
            waves.append(torch.zeros(int(0.55 * 24000)))

    audio = master_audio(limit_peak(torch.cat(waves), 0.98).numpy(), 24000, master=master)
    sf.write(out_path, audio, 24000, subtype="PCM_16")
    print(f"Saved: {out_path} ({len(audio) / 24000:.2f}s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("text", help="Finnish text to synthesize")
    p.add_argument("--ckpt", default=str(Path(__file__).parent / "weights" / "joint_inference.pt"))
    p.add_argument("--out", default="output.wav")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--no-master", action="store_true")
    args = p.parse_args()
    synthesize(args.text, args.ckpt, args.out, speed=args.speed, master=not args.no_master)
