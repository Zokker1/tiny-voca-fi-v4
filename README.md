# tiny-voca-fi-v4 (50k steps)

Suomenkielinen TTS-mallin koulutusharjoitus Google Colabissa. Malli on koulutettu alusta asti suomenkielisellä puhedatalla [`n4ze3m/tiny-voca`](https://github.com/n4ze3m/tiny-voca)-arkkitehtuuria käyttäen, ja aaltomuodon synteesiin on yhdistetty `AcousticToVocosMel`-silta sekä yhdessä hienosäädetty **Vocos 24 kHz** -vokooderi.

Projektin tarkoituksena on ollut kokeilla kevyen suomenkielisen neuro-TTS-mallin rakentamista ja kouluttamista tyhjästä alkaen sekä testata sitä paikallisesti HTML-käyttöliittymän kautta.

## Malli ja koulutus

- **Kieli ja puhujat:** Suomi (`fi`), kaksi ääntä: **`M2`** (mies) ja **`F2`** (nainen).
- **Koulutus Google Colabissa:** Pohjamalli koulutettiin alusta asti (`0 → 30 000` askelta) ja jatkokoulutettiin yhteensä **50 000 askeleeseen** (`v4`, `weights/joint_inference.pt`).
- **Aineisto:** `4 855` koulutusnäytettä ja `145` validointinäytettä (`CSS10 Finnish` `M2` + `VoxPopuli Finnish` `F2`), kohdistettu foneemitasolle mallilla `facebook/wav2vec2-lv-60-espeak-cv-ft`.
- **Suomenkielinen frontend (`tiny_voca/text.py`):** 92 foneemin suomalainen merkistö (mm. geminaatat `pː, tː, kː`, `r`, pitkät ja lyhyet vokaalit sekä diftongien purkusäännöt `espeak-ng -v fi`:n päälle).
- **Vokooderi:** `TinyVoca` (akustiset piirteet) → `AcousticToVocosMel` (192 → 100 mel-kanavaa) → hienosäädetty **Vocos 24 kHz** (`charactr/vocos-mel-24khz`).
- **Välivaiheen arviointi (`step 30 000`, `meta/fi_evaluation_report_step30000.json`):** Whisperillä mitattuna `mean_cer = 2.64 %` ja `mean_wer = 20.72 %`.

## Painojen lisääminen paikallisesti
Lataa `joint_inference.pt` tämän GitHub-repon [**Releases**](https://github.com/Zokker1/tiny-voca-fi-v4/releases/tag/v4)-osiosta (`v4`) ja tallenna se kansioon `weights/joint_inference.pt`.

*(Tai jos sinulla on `tiny_voca_fi_v4_bundle.zip`, pura se suoraan tähän kansioon niin, että `weights/joint_inference.pt` ja `meta/` menevät paikalleen.)*

## Asennus ja ajo Windowsissa
1. Asenna [`espeak-ng`](https://github.com/espeak-ng/espeak-ng/releases) (tai `winget install eSpeak-NG.eSpeak-NG`).
2. Luo virtuaaliympäristö ja asenna Python-riippuvuudet:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **HTML-käyttöliittymä:**
   ```bash
   python server.py --open
   ```
   Tai käynnistä suoraan Windowsissa tuplaklikkaamalla `start_ui.bat`. Palvelin avaa selaimessa osoitteen `http://127.0.0.1:8000`.
4. **Komentoriviajo:**
   ```bash
   python synthesize.py "Hyvää iltaa! Tämä on pääversion suomenkielinen neuro-TTS." --out testi_v4.wav
   ```

## Lisenssi ja viittaukset
- Pohjautuu [`n4ze3m/tiny-voca`](https://github.com/n4ze3m/tiny-voca)-arkkitehtuuriin (Apache-2.0) ja [`charactr/vocos-mel-24khz`](https://huggingface.co/charactr/vocos-mel-24khz)-vokooderiin.
- Lisenssi: Apache-2.0 ([`LICENSE`](LICENSE)).
