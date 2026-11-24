# MetroSync Studio Pro

[English](README.md) | **Italiano**

Applicazione desktop Windows per creare, cercare, scaricare e pubblicare testi sincronizzati **parola per parola** e **riga per riga** compatibili con [**Metrolist**](https://github.com/metrolistgroup/metrolist), **BetterLyrics / Unison** e **LRCLIB**.

## Funzionalità Principali

- **Sincronizzazione AI Ibrida ad Alta Precisione**: isolamento vocale con **Demucs**, macro-ancoraggio Whisper, forced alignment fonetico CTC (`MMS_FA`) e raffinamento acustico multi-banda a `1 ms`.
- **Ricerca & Download Testi Online**: cerca direttamente dall'app su **BetterLyrics / Unison** e **LRCLIB**, visualizza l'anteprima convertita e scarica in locale nei formati **Metrolist LRC**, **TTML**, **Lyricsfile YAML** o **Testo grezzo (`.txt`)** pronto per la sincronizzazione AI.
- **Gestione Autenticazione Servizi di Pubblicazione**:
  - **BetterLyrics / Unison**: importa/esporta identità ECDSA P-256 (`unison_identity.json` o formato JWK BetterLyrics), genera nuove chiavi, imposta etichette profilo, endpoint personalizzati e Bearer Token opzionale.
  - **LRCLIB**: scegli tra PoW Challenge automatico (`lrclib.net`), `X-Publish-Token` manuale, Bearer Token per istanze private, endpoint custom e `User-Agent` personalizzato.
- **Anteprima Karaoke Live stile Metrolist**: effetto wipe parola per parola, parole allungate, pause intra-riga e countdown pause strumentali.
- **Esportazione & Progetti `.metro`**: salva/ricarica progetti completi ed esporta i file locali in un clic.

## Requisiti & Installazione

1. **Python 3.12** (consigliato `py -3.12` su Windows).
2. **FFmpeg & yt-dlp**: su Windows, se non sono già presenti nel `PATH`, MetroSync scarica automaticamente le build ufficiali verificate (`yt-dlp/yt-dlp` e `yt-dlp/FFmpeg-Builds`) nella sottocartella locale `tools/` alla prima necessità.
3. Installa PyTorch (con CUDA se hai una GPU NVIDIA):
   ```powershell
   py -3.12 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
4. Installa le dipendenze del progetto:
   ```powershell
   py -3.12 -m pip install -r requirements.txt
   ```
5. Verifica il rilevamento GPU CUDA:
   ```powershell
   py -3.12 app\check_gpu.py
   ```
6. *(Consigliato per modalità offline)* I pesi Whisper (`.pt`) vanno nella cartella `models/` (puoi scaricarli direttamente dall'app in **Impostazioni & AI**).

## Avvio

Avvia l'app con doppio clic su `MetroSync.bat` (o `MetroSync.lnk`) oppure da terminale:

```powershell
py -3.12 app\main.py
```

## Struttura delle Cartelle

- `app/` — codice Python (`main.py`, `check_gpu.py`) e interfaccia web (`index.html`, `script.js`, `style.css`).
- `models/` — pesi dei modelli Whisper (`.pt`).
- `output/` — sottocartelle per ciascun brano (`output/<Artista - Titolo>/`) contenenti i file sincronizzati (`.lrc`, `.ttml`, `.lyricsfile.yaml`).
- `tools/` — eseguibili ufficiali `yt-dlp.exe` e `ffmpeg.exe` (scaricati automaticamente su richiesta).

## Formati Generati e Compatibilità Metrolist

I file generati rispettano le specifiche di [**Metrolist**](https://github.com/metrolistgroup/metrolist) (`https://github.com/metrolistgroup/metrolist`) per la riproduzione karaoke parola per parola e riga per riga:

- **`.lrc` (Metrolist Duet / Normale)**: include header metadati (`[ti:]`, `[ar:]`, `[al:]`, `[au:]`, `[length:]`), tag vocali `{agent:v1}` / `{agent:v2}` / `{bg}` e timing parola per parola `<parola:inizio:fine|...>`.
- **`.ttml`**: formato Apple Music / BetterLyrics / Metrolist (`itunes:timing="Word"` o `"Line"`).
- **`.lyricsfile.yaml`**: formato Lyricsfile 1.0 YAML compatibile con LRCLIB.
- **`.metro`**: file progetto JSON con audio collegato, metadati, impostazioni e output generati.

## Note di Sicurezza prima di Condividere

- **Non condividere mai `unison_identity.json`**: contiene la tua chiave privata ECDSA P-256 per pubblicare su Unison. È già escluso automaticamente tramite `.gitignore`.
- **`config.json`** contiene preferenze locali ed eventuali token personalizzati ed è anch'esso escluso da `.gitignore`.

## Licenza & Obbligo di Citazione

Questo progetto è distribuito gratuitamente sotto **Licenza MIT** ([LICENSE](LICENSE)).

Puoi usare, modificare e ridistribuire il software liberamente, **a condizione di mantenere l'avviso di copyright e citare l'autore originale**:

- **Autore**: `puccilorenzo21`
- **Repository originale**: https://github.com/puccilorenzo21/MetroSync
