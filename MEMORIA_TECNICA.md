# MetroSync Studio Pro — Memoria Tecnica Completa (AI Context & Architecture Guide)

> Documento di riferimento rapido per qualsiasi agente AI o sviluppatore che interviene su `MetroSync`. Contiene architettura, contratti dati, punti di modifica chiave, invarianti critiche e trappole note verificate sullo stato attuale del codice.

---

## 1. Cos'è MetroSync Studio Pro

Applicazione desktop Windows (PyWebView + HTML/CSS/JS + backend Python/PyTorch CUDA) per creare testi sincronizzati **parola per parola** e **riga per riga** ad altissima precisione partendo da audio (`.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`) e testo grezzo o già sincronizzato (`.txt`, `.lrc`, `.ttml`, `.yaml`), con:
- Separazione stem vocale neurale con **Demucs** (`stable_whisper.audio.demucs_audio`).
- Allineamento ibrido **Whisper Macro-Anchors + Forced Alignment fonetico CTC (`torchaudio.pipelines.MMS_FA` a 20 ms) + Raffinamento acustico multi-banda a 1 ms** (con fallback automatico `stable_whisper` se `MMS_FA` non è disponibile o il testo non è in alfabeto latino).
- Estrazione automatica dei metadati dai tag file (`mutagen`), dal nome file, dalle descrizioni YouTube e dai file testo.
- Gestione crediti/note e riga finale personalizzabile nella coda strumentale (`outro_credits` / `outro_text`).
- Anteprima karaoke live in-app stile Metrolist (wipe progressivo per parola, parole allungate, punti respiro intra-riga, countdown pause strumentali).
- Esportazione automatica locale in **`.lrc`**, **`.lyricsfile.yaml`**, **`.ttml`** e salvataggio/caricamento progetti **`.metro`** (JSON), oltre a pubblicazione opzionale su **LRCLIB** e **BetterLyrics / Unison** (solo su azione esplicita dell'utente).

---

## 2. Mappa dei File nel Progetto (`MetroSync/`)

| File / Cartella | Ruolo | Note importanti |
| :--- | :--- | :--- |
| `app/main.py` | Backend completo + finestra PyWebView (`BackendAPI`) | Punto di ingresso unico dell'app (`py -3.12 app/main.py`) |
| `app/index.html` | UI Studio Pro (2 tab: `Studio`, `Impostazioni & AI` + modali) | Tutti i controlli e metadati della UI |
| `app/script.js` | Logica frontend, bridge `window.pywebview.api.*`, anteprima karaoke HTML5 Canvas/DOM | Validabile con `node -c app/script.js` |
| `app/style.css` | Tema scuro neon/cyan + layout anteprima karaoke e toggle | `.word-base` / `.word-fill` per effetto wipe |
| `MetroSync.bat` | Launcher Windows | Lancia `start "" py -3.12 "app\main.py"` |
| `app/check_gpu.py` | Diagnostica rapida PyTorch CUDA / VRAM | Eseguibile con `py -3.12 app/check_gpu.py` |
| `config.json` | Impostazioni utente persistenti | Escluso da Git; chiavi: `model_path`, `language`, `demucs`, `vad`, `format`, `offset`, `auto_offset`, `outro_credits` |
| `unison_identity.json` | Chiave ECDSA P-256 generata localmente per firma payload Unison/BetterLyrics | Contenuto sensibile locale (escluso da Git, non condividere) |
| `models/*.pt` | Pesi Whisper locali offline | Esclusi da Git; `HF_HUB_OFFLINE=1` e `TRANSFORMERS_OFFLINE=1` forzati per l'inferenza |
| `output/<Brano>/` | Sottocartelle locali dei testi sincronizzati (`.lrc`, `.ttml`, `.lyricsfile.yaml`) | Escluse da Git |

### Stack / Dipendenze Python usate in `main.py`
- `python 3.12` (`py -3.12`)
- `webview` (`pywebview`), `torch`, `torchaudio` (`MMS_FA`), `openai-whisper`, `stable_whisper` (`stable-ts`), `demucs`, `numpy`
- Opzionali con fallback: `scipy.signal` (`butter`, `sosfiltfilt`, `fftconvolve`), `mutagen` (`HAS_MUTAGEN`), `cryptography` (`HAS_CRYPTO`, curva `SECP256R1` per Unison)

---

## 3. Architettura Backend (`main.py`)

### 3.1 Motore Acustico a 1 ms (`AcousticPrecisionEngine`, ~righe 546–1189)
Calcola inviluppi energetici logaritmici a `1 ms` (`sr=16000, hop_len=16`) sul vocal stem:
- **4 bande/inviluppi**:
  - `env_sib` (4000–7500 Hz, sibilanti/consonanti esplosive),
  - `env_body` (220–3500 Hz, corpo vocale),
  - `env_low` (85–250 Hz, fondamentale),
  - `env_sustain` (potenza armonica finestra 24 ms per vocali tenute e vibrato).
- **Floor adattivo locale**: `local_floor_total` / `local_floor_sustain` combinano floor locale scorrevole e floor globale (`12° percentile`) per gestire variazioni tra intro e ritornello.
- **Funzione di flusso d'attacco `odf`**: derivata positiva pesata (`3.0*diff_sib + 1.6*diff_body + 0.5*diff_low`) normalizzata localmente.
- **Metodi chiave**:
  - `detect_vocal_islands(merge_gap_ms=260, min_dur_ms=70)`: segmenti `(start_s, end_s)` di energia vocale reale.
  - `snap_onset_micro(...)`: micro-aggancio `±25ms` al piede reale del transiente.
  - `expand_phrase_onset(...)`: recupera attacchi consonantici/vocalici deboli fino a `500ms` indietro se separati da pause.
  - `refine_phrase_offset(..., extend_sustain=True)`: taglia le code di silenzio e allunga automaticamente vocali tenute / melismi quando `env_sustain` resta attivo.
  - `find_gap_adlib_span(t0, t1)` e `find_strong_burst_in_gap(t0, t1)`: individuano ad-lib e sillabe isolate nelle pause tra righe.

### 3.2 Auto-Compensazione Temporale (`calculate_auto_compensation_offset`, ~righe 1190–1275 & `analyze_auto_compensation`, ~righe 3964–4132)
- **Funzionamento bidirezionale**: per un campione fino a `120` token filtrati (`0.10s <= dur <= 1.60s`), cerca il picco `odf` reale nella finestra `±search_window` e misura `delta = acoustic_t - t_raw`:
  - **`> 0` (`+ms`)**: il modello anticipa → bisogna spostare avanti.
  - **`< 0` (`-ms`)**: il modello ritarda → bisogna anticipare.
  - Filtro outlier IQR + deadzone (`8 ms`), clamp a `±180 ms`.
- **Pre-analisi UI (`analyze_auto_compensation`)**:
  1. Seleziona una finestra attiva di `32 s` centrata sui primi versi vocali reali (`win_s..win_e`).
  2. Trascrive con `large-v3-turbo.pt` (se presente), calcola il bias Whisper grezzo riportando sempre i token in coordinate globali (`win_s + tok.start`).
  3. Se `HAS_MMS_FA` e testo latino presenti, esegue anche `HybridCTCAligner.align` in coordinate locali su `probe_audio` e misura il bias residuo reale del motore ibrido (`mode = "Hybrid CTC + Acustico"` vs `"Whisper"`).
  4. Emette verso JS `window.autoOffsetFinished(payload)` con `offset_ms`, `whisper_offset_ms`, `mode`.
- **In sincronizzazione (`_execute_synchronization` / `_run_whisper_fallback_pipeline`)**: misura il bias residuo (`search_window=60ms` su CTC, `180ms` su fallback Whisper) e nel fallback somma `auto_shift_sec` prima dei raffinamenti acustici.

### 3.3 Motore Ibrido CTC (`HybridCTCAligner`, ~righe 1504–2328)
Pipeline principale quando `HAS_MMS_FA` e `is_latin_script_compatible(cleaned_text)` sono veri (altrimenti passa automaticamente a `_run_whisper_fallback_pipeline`):
1. **Macro-ancoraggio Whisper**: usa preferibilmente `large-v3-turbo.pt` anche se in UI è selezionato `large-v3.pt` / `large-v2.pt` per velocità (`~10–18s`), poi esegue recovery automatico su gap >= `3.8s` con isole vocali attive e suddivisione di segmenti sparsi troppo lunghi (`>= 5.5s`).
2. **Emissioni CTC `MMS_FA` per finestra (`compute_ctc_emissions`)**:
   - Processa il vocal stem in **chunk corti da `8.0 s`** con padding `1.5 s` passando direttamente tramite **`mms_model(seg_t)`** (`layer_norm` locale + `log_softmax` + `*` star).
   - > ⚠️ **Critico**: usare chunk lunghi (`25s+`) o normalizzazione globale schiaccia le intro/strofe basse adiacenti ai ritornelli forti (crollo probabilità fino a `~0.0002`). Mantenere `keep_model=True` in `align()` per consentire la ri-valutazione locale.
3. **DP Fonetico & Varianti Multilingua (`_clean_phonetic_ctc`, `word_variants`)**:
   - Normalizza elisioni/accenti e genera varianti per numeri `< 10000` in `it/en/es/fr/de` (`600` → `seicento`, `112` → `centododici`/`uno uno due`), oltre ad acronimi/slang (`tv`, `ok`).
   - Allinea tramite Programmazione Dinamica i testi alle parole macro-ancora Whisper escludendo temporaneamente gli eco/BG brevi (`_is_echo_bg`) ed evitando di inglobare ad-lib separati da pause >= `0.45s`/`0.50s`.
4. **Allineamento a 3 Passate (`_align_with_resources`)**:
   - **Pass 1A (`_align_and_refine_block`)**: blocchi continui ancorati allineati con `_align_block_ctc`. Se la confidenza CTC media è `< 0.08`, una parola dura `>= 1.25s`, o `prefer_local=True`, ricalcola `_run_ctc_on_emission` su una slice locale isolata `[win_s - 0.12, win_e + 0.12]` con `mms_model(loc_t)`.
   - **Pass 1B**: recupera righe lessicali/interiezioni non ancorate mappandole sulle `vocal_islands` nei gap liberi.
   - **Pass 2**: riallinea righe `{bg}` / `(parentesi)`. Se c'è un `gap_adlib` distinto prima di `n_start`, isola la finestra `[max(gap_s, gap_adlib[0]-0.12), min(n_start-0.03, ...)]` con `prefer_local=True`.
5. **Post-processing non-cascading (`_finalize_processed_lines`, ~righe 4137–4200)**:
   - Applica `manual_offset_ms`, `smooth_and_clamp_words` (`min_word_duration = 0.035s`), garantisce monotonicità e **rifila la coda della riga precedente** se sconfina sulla successiva (evitando slittamenti a cascata sull'intera canzone).

### 3.4 Metadati, Outro Personalizzabile & Generazione Formati
- **Estrazione (`extract_audio_metadata` & `extract_lyrics_metadata`, ~righe 221–440)**:
  - Da audio (`mutagen` + filename `"Artista - Titolo"` / `"Titolo by Artista"` + URL YouTube/descrizioni `Lyricist`/`Writer`/`Songwriter`/`Composer`).
  - Da testo (`.ttml`, `.lyricsfile.yaml`, tag `.lrc` `[ti:]`, `[ar:]`, `[al:]`, `[au:]`, `[la:]`, `[by:]`, `[# Note:]`).
- **Override UI sicuro (`_pick_optional` in `_execute_synchronization`, ~righe 4804–4829)**:
  - Quando `params` arriva dalla UI con la chiave presente (anche stringa vuota `""`), il valore UI **vince sempre** e **non** viene sovrascritto dai tag MP3/testo. I fallback su `audio_meta`/`lyr_meta` avvengono solo se la chiave manca in `params`.
- **Riga Finale / Crediti in Coda (`apply_outro_credits_to_structured`, ~righe 2919–2994)**:
  - Idempotente su tutti i formati tramite `strip_outro_credits_from_lrc_lines`, `strip_outro_credits_from_plain` e `_last_outro_candidates`.
  - **Ordine di priorità contenuto**:
    1. `outro_text` (campo `#metaOutroText`) **verbatim**, se compilato.
    2. `"Testo: <sw1, sw2, ...>"` se `songwriters` non vuoto.
    3. `notes` (`#metaNotes`) se anche `songwriters` è vuoto.
  - **Vincolo temporale rigido**: calcola `eff_dur = min(duration_sec, calculated_duration_exact)` e `limit_ms = round(eff_dur * 1000) - 120`. Aggiunge la riga **solo se `limit_ms - last_end_ms >= 900 ms`** (gap iniziale `250..700ms`, durata max `7.5s`). Garantisce `timestamp <= duration` per validazione LRCLIB e fine audio reale.
- **Aggiornamento live su memoria + disco (`update_outputs_metadata` + `_write_synced_output_files`, ~righe 3403–3628)**:
  - Ogni modifica post-sync ai metadati, checkbox outro o testo finale aggiorna subito RAM e file `.lrc` / `.ttml` / `.lyricsfile.yaml` dell'ultima sincronizzazione (`self.last_output_paths`).

### 3.5 Formati & Publisher Remoti
1. **LRC Esteso (`convert_structured_to_lrc` + `format_lrc_metadata_headers`)**:
   - Header `[ti:]`, `[ar:]`, `[al:]`, `[au:]`, `[length:MM:SS]`, `[re:MetroSync]`, `[# Note: ...]`.
   - Righe `[mm:ss.cc]{agent:v1}Testo riga` seguite da `<parola1:start:end|parola2:start:end>`.
   - Righe background marcate con `{bg}`. In modalità `normale`, inserisce timestamp strumentali `[mm:ss.cc]` nelle pause `>= 1.4s`.
   - `format_lrc_for_lrclib()` rimuove metadati e tag `{agent:vN}`/`{bg}` prima dell'invio a LRCLIB.
2. **Lyricsfile 1.0 YAML (`build_lyricsfile_yaml` / `parse_yaml_to_structured_lines`)**:
   - Serializzazione/parsing senza dipendenze PyYAML; include `songwriters` (lista) e `notes` sotto `metadata:`.
3. **TTML Apple/BetterLyrics (`build_ttml` / `parse_ttml_to_structured_lines`)**:
   - Namespace `itunes:timing="Word"`, `<ttm:title>`, `<ttm:desc>` (note), `<iTunesMetadata><songwriters><songwriter>`, `<p begin end ttm:agent ttm:role="x-bg">` e `<span begin end>`.
4. **LRCLIB (`publish_and_verify_lrclib`, ~righe 3167–3250)**:
   - `POST https://lrclib.net/api/request-challenge` → risolve PoW SHA-256 (`solve_lrclib_challenge`) → `POST https://lrclib.net/api/publish` (`X-Publish-Token`).
5. **BetterLyrics / Unison (`publish_and_verify_betterlyrics`, ~righe 3253–3353)**:
   - Estrae Video ID YouTube da 11 caratteri (`extract_youtube_id`), firma canonical JSON con chiave ECDSA `P-256` (`unison_identity.json`) in formato P1363 (con retry DER su `INVALID_SIGNATURE`) → `POST https://unison.betterlyrics.org/lyrics/submit`.

---

## 4. Bridge UI ↔ Backend (`index.html`, `script.js`, `BackendAPI`)

### Metodi Python esposti a JS (`window.pywebview.api.*`)
- `init_info()` → hardware, modelli disponibili, config iniziale (`config.json`), lingue.
- `save_user_settings(settings)` → merge e salvataggio su `config.json` + aggiornamento `current_outro_credits`.
- `select_audio_file()` → dialog audio + `extract_audio_metadata` + popolamento stato.
- `select_lyrics_file()` / `import_synced_lyrics_file()` → carica `.txt` grezzo o importa file già sincronizzati (`.lrc`/`.ttml`/`.yaml`) via `_process_synced_content`.
- `select_model_file()` → seleziona peso `.pt` custom.
- `auto_fill_metadata_from_files(audio_path, lyrics_path)` → estrae/fonde metadati dai file selezionati.
- `update_outputs_metadata(metadata)` → rigenera output in RAM e su `last_output_paths` senza ri-eseguire l'AI.
- `save_project_file(data)` / `load_project_file()` → serializza/ripristina `.metro` JSON completo.
- `get_audio_data_url()` → legge l'audio locale come Base64 Data URL per `#html5Audio` in anteprima.
- `analyze_auto_compensation(params)` → thread asincrono, callback `window.autoOffsetFinished(res)`.
- `start_synchronization(params)` → thread asincrono (`_execute_synchronization`), progress `window.updateProgress(pct)`, log `window.appendLog(msg)`, callback `window.syncFinished(ok, data)`.
- `publish_lrclib(metadata)` → callback `window.publishFinished(res)`.
- `publish_betterlyrics(metadata)` → callback `window.publishBetterLyricsFinished(res)`.

### Elementi UI principali (`index.html` & `script.js`)
- **File Sorgente**: `#audioPath`, `#lyricsPath`, bottoni Sfoglia + `⚡ Importa Sincronizzato`.
- **Metadati (`#tabStudio`)**: `#btnAutoFillMeta`, `#metaTitle`, `#metaArtist`, `#metaAlbum`, `#metaDuration`, `#studioOffsetBadge`, `#btnQuickCalibrate`, `#metaYoutube`, `#metaSongwriters`, `#metaNotes`, checkbox `#settingOutroCredits`, input `#metaOutroText`.
- **Azioni**: `#btnSync`, `#btnPreview`, `#btnPublish` (LRCLIB), `#btnPublishBL` (BetterLyrics), `#btnCopy`, `#btnSaveProj`, progresso (`#progressBar`/`#progressText`), `#logConsole`.
- **Impostazioni (`#tabSettings`)**: `#settingLanguage`, `#settingModelSelect`/`#settingModel`, `#settingDemucs`, `#settingVAD`, radio `settingFormat` (`metrolist`/`normale`), `#settingAutoOffset`, `#autoOffsetBadge`, `#btnCalibrate`, `#settingOffset` (`-800..+800ms`).
- **Anteprima Karaoke (`#previewModal`)**: `openPreview()` chiama prima `refreshOutputsFromForm()`, poi `parseLRC()` → `renderLyricsDOM()` (con `document.createTextNode(' ')` tra `.word` per mantenere gli spazi in copia) → loop `requestAnimationFrame(updateLoop)` con sweep `.word-fill`, pausa strumentale `#silenceCountdown` se gap >= `1.25s` o inizio > `1.2s`, indicatori `.intra-pause-dot` se gap intra-riga >= `0.35s`.

---

## 5. Invarianti & Regole Critiche (Non Rompere!)

1. **Formattazione Timestamp**: usare sempre `divmod(int(round(max(0.0, s) * 100)), 6000)` in `format_lrc_time` e `divmod(..., 60000)` in `format_ttml_time` per evitare rollover invalidi tipo `[01:60.00]`.
2. **Parsing token con `:`**: parole come `"Testo:"` o `"dire:"` contengono due punti. Usare sempre `":".join(parts[:-2])` (Python) e `parts.slice(0, parts.length - 2).join(':')` (JS).
3. **Chunk `MMS_FA` e normalizzazione**: mantenere chunk corti (`8.0s`) via `mms_model(seg_t)` + fallback locale isolato nel `HybridCTCAligner`; non ripristinare chunk lunghi o normalizzazione globale dell'onda.
4. **Rispetto input utente svuotati**: non sostituire `""` con vecchi tag MP3 quando le chiavi UI sono presenti in `params`.
5. **Outro Credits entro EOF audio**: la riga finale deve sempre chiudersi sotto `limit_ms = round(min(duration_sec, exact_dur) * 1000) - 120` (solo se spazio libero `>= 900ms`) ed essere rimossa/sostituita in modo idempotente al cambio checkbox o testo.
6. **Sicurezza pubblicazione**: mai invocare `publish_and_verify_lrclib` o `publish_and_verify_betterlyrics` nei test automatici o senza richiesta esplicita dell'utente.
7. **Verifica rapida dopo ogni modifica**:
   - Sintassi Python: `py -3.12 -m py_compile app/main.py`
   - Sintassi JS: `node -c app/script.js`
   - CUDA check: `py -3.12 app/check_gpu.py`

---

## 6. Struttura File Progetto `.metro` (JSON)

Salvato da `saveProject()` (`script.js`) + `save_project_file()` (`main.py`), ricaricato da `load_project_file()` (`main.py`) + `loadProject()` (`script.js`):
- **Metadati**: `title`, `artist`, `album`, `youtube`, `youtube_id`, `songwriters`, `notes`, `outro_text`, `outro_credits`, `duration`, `language`
- **Sorgenti & Output**: `audio_path`, `lyrics_text`, `plain_lyrics`, `generated_lrc`, `generated_lyricsfile`, `generated_ttml`
- **Impostazioni locali (`settings`)**: `format` (`"metrolist"` | `"normale"`), `offset`, `auto_offset`, `outro_credits`
