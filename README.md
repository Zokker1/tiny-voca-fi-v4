# tiny-voca-fi-v4 (50k steps)

Suomenkielinen kevyt neuro-TTS ([`tiny-voca`](https://github.com/n4ze3m/tiny-voca)-pohjainen 92-foneemin frontend + `AcousticToVocosMel` + `Vocos 24 kHz`), **pääversio v4 (50 000 askelta)**.

Puhujat: **`M2`** (mies) ja **`F2`** (nainen).

## Painojen lisääminen paikallisesti
Lataa `joint_inference.pt` tämän GitHub-repon **Releases**-osiosta (`v4`) ja tallenna se kansioon `weights/joint_inference.pt`.

*(Tai jos sinulla on `tiny_voca_fi_v4_bundle.zip`, pura se suoraan tähän kansioon niin, että `weights/joint_inference.pt` menee paikalleen.)*

## Asennus ja ajo Windowsissa
1. Asenna [`espeak-ng`](https://github.com/espeak-ng/espeak-ng/releases) (tai `winget install eSpeak-NG.eSpeak-NG`).
2. Luo virtuaaliympäristö ja asenna Python-riippuvuudet:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **HTML-käyttöliittymä (suositus):**
   ```bash
   python server.py --open
   ```
   Tai käynnistä suoraan Windowsissa tuplaklikkaamalla `start_ui.bat`. Palvelin avaa selaimessa osoitteen `http://127.0.0.1:8000`.
4. **Komentoriviajo:**
   ```bash
   python synthesize.py "Hyvää iltaa! Tämä on pääversion suomenkielinen neuro-TTS." --out testi_v4.wav
   ```

## Lisenssi ja pohjatyö
- Pohjautuu [`n4ze3m/tiny-voca`](https://github.com/n4ze3m/tiny-voca)-arkkitehtuuriin (Apache-2.0) ja käyttää [`charactr/vocos-mel-24khz`](https://huggingface.co/charactr/vocos-mel-24khz)-vokooderia.
- Lisenssi: Apache-2.0 (`LICENSE`).
