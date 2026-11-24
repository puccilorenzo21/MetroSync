# MetroSync Studio Pro

**English** | [Italiano](README.it.md)

[![Platform: Windows](https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows&logoColor=white)](https://github.com/puccilorenzo21/MetroSync)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Compatible: Metrolist](https://img.shields.io/badge/Compatible-Metrolist%20%7C%20BetterLyrics%20%7C%20LRCLIB-8A2BE2)](https://github.com/metrolistgroup/metrolist)

A Windows desktop studio to create, search, download, and publish **word-by-word** and **line-by-line** synchronized lyrics compatible with [**Metrolist**](https://github.com/metrolistgroup/metrolist), **BetterLyrics / Unison**, and **LRCLIB**.

<!-- TODO: Replace with a real screenshot or GIF of the Live Karaoke Preview / Editor -->
<!-- ![MetroSync Studio Pro Preview](docs/preview.png) -->

---

## Key Features

- **High-Precision Hybrid AI Synchronization**: vocal isolation with **Demucs**, **Whisper** macro-anchoring, **CTC phonetic forced alignment (`MMS_FA`)**, and `1 ms` multi-band acoustic refinement.
- **Online Lyrics Search & Download**: search directly from the app on **BetterLyrics / Unison** and **LRCLIB**, preview converted output, and download locally as **Metrolist LRC**, **TTML**, **Lyricsfile YAML**, or **plain text (`.txt`)** ready for AI alignment.
- **Publishing Service Authentication**:
  - **BetterLyrics / Unison**: import/export ECDSA P-256 identities (`unison_identity.json` or BetterLyrics JWK format), generate new keys, configure profile labels, custom endpoints, and optional Bearer Tokens.
  - **LRCLIB**: supports automatic PoW Challenge (`lrclib.net`), manual `X-Publish-Token`, Bearer Token for private instances, custom endpoints, and custom `User-Agent`.
- **Metrolist-Style Live Karaoke Preview**: word-by-word wipe effect, elongated word handling, intra-line pauses, and instrumental countdowns.
- **One-Click Export & `.metro` Projects**: save and reload full projects (audio link, metadata, settings, generated outputs) and export local files in one click.

---

## Requirements & Installation

1. **Python 3.12** (`py -3.12` recommended on Windows).
2. **FFmpeg & yt-dlp**: on Windows, if not already in your `PATH`, MetroSync automatically downloads verified official builds (`yt-dlp/yt-dlp` and `yt-dlp/FFmpeg-Builds`) into the local `tools/` subfolder when first needed.
3. Install PyTorch (with CUDA if you have an NVIDIA GPU):
   ```powershell
   py -3.12 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
4. Install project dependencies:
   ```powershell
   py -3.12 -m pip install -r requirements.txt
   ```
5. Verify CUDA GPU detection:
   ```powershell
   py -3.12 app\check_gpu.py
   ```
6. *(Recommended for offline mode)* Place Whisper weights (`.pt`) in the `models/` folder (or download them directly inside the app under **Settings & AI**).

---

## Launch

Double-click `MetroSync.bat` (or `MetroSync.lnk`), or run from terminal:

```powershell
py -3.12 app\main.py
```

---

## Folder Structure

- `app/` — Python backend (`main.py`, `check_gpu.py`) and web UI (`index.html`, `script.js`, `style.css`).
- `models/` — Whisper model weights (`.pt`).
- `output/` — per-track output folders (`output/<Artist - Title>/`) containing synchronized files (`.lrc`, `.ttml`, `.lyricsfile.yaml`).
- `tools/` — official `yt-dlp.exe` and `ffmpeg.exe` binaries (auto-downloaded on demand).

---

## Supported Formats & Metrolist Compatibility

Generated files follow [**Metrolist**](https://github.com/metrolistgroup/metrolist) specifications for word-by-word and line-by-line karaoke playback:

- **`.lrc` (Metrolist Duet / Standard)**: includes metadata headers (`[ti:]`, `[ar:]`, `[al:]`, `[au:]`, `[length:]`), vocal tags (`{agent:v1}`, `{agent:v2}`, `{bg}`), and word-level timing `<word:start:end|...>`.
- **`.ttml`**: Apple Music / BetterLyrics / Metrolist format (`itunes:timing="Word"` or `"Line"`).
- **`.lyricsfile.yaml`**: Lyricsfile 1.0 YAML format compatible with LRCLIB.
- **`.metro`**: JSON project file storing linked audio, metadata, settings, and generated outputs.

---

## Security Notes Before Sharing

- **Never share `unison_identity.json`**: it contains your ECDSA P-256 private key for publishing to Unison (already excluded via `.gitignore`).
- **`config.json`** stores local preferences and optional custom tokens (also excluded via `.gitignore`).

---

## License & Attribution

Distributed under the **MIT License** ([LICENSE](LICENSE)).

You are free to use, modify, and redistribute this software, **provided that you keep the copyright notice and credit the original author**:

- **Author**: `puccilorenzo21`
- **Original repository**: https://github.com/puccilorenzo21/MetroSync
