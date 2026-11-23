import base64
import gc
import hashlib
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import warnings
import webbrowser
import xml.sax.saxutils as saxutils
import zipfile
from difflib import SequenceMatcher

import numpy as np
import stable_whisper as stable_ts
import torch
import webview
import whisper

# Disabilita chiamate remote a Hugging Face che causano blocchi o rate-limit
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

import unicodedata
from difflib import SequenceMatcher
from itertools import product

try:
    from scipy.signal import butter, sosfilt, sosfiltfilt, fftconvolve
    from numpy.lib.stride_tricks import sliding_window_view
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import torchaudio
    HAS_MMS_FA = hasattr(torchaudio, "pipelines") and hasattr(torchaudio.pipelines, "MMS_FA")
except Exception:
    HAS_MMS_FA = False

try:
    import mutagen
    HAS_MUTAGEN = True
except Exception:
    HAS_MUTAGEN = False

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        load_pem_private_key,
    )
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

LOCAL_MODEL_V3 = os.path.join(MODELS_DIR, "large-v3.pt")
LOCAL_MODEL_V2 = os.path.join(MODELS_DIR, "large-v2.pt")
LOCAL_MODEL_TURBO = os.path.join(MODELS_DIR, "large-v3-turbo.pt")
LOCAL_MODEL_DISTIL_IT = os.path.join(MODELS_DIR, "whisper-large-v3-distil-it-v0.2.pt")
LOCAL_MODEL_MEDIUM = os.path.join(MODELS_DIR, "medium.pt")
LOCAL_MODEL_SMALL = os.path.join(MODELS_DIR, "small.pt")

WHISPER_MODEL_CATALOG = [
    {
        "id": "large-v3",
        "filename": "large-v3.pt",
        "path": LOCAL_MODEL_V3,
        "label": "large-v3.pt",
        "dropdown_name": "large-v3.pt (Massima Precisione Multilingua • 32 Layer / 128 Mel)",
        "preset": "large-v3",
        "description": "Il modello OpenAI Whisper più preciso in assoluto (multilingua, 32 Layer / 128 Mel) • ~2.88 GB",
        "approx_gb": 2.88,
        "recommended": True,
        "url": getattr(whisper, "_MODELS", {}).get(
            "large-v3",
            "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
        ),
        "sha256": "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb",
    },
    {
        "id": "large-v3-turbo",
        "filename": "large-v3-turbo.pt",
        "path": LOCAL_MODEL_TURBO,
        "label": "large-v3-turbo.pt",
        "dropdown_name": "large-v3-turbo.pt (Modalità Rapida • 4 Layer)",
        "preset": "turbo",
        "description": "Veloce e preciso per macro-ancoraggio & Auto-Calibrazione (4 Layer) • ~1.51 GB",
        "approx_gb": 1.51,
        "recommended": True,
        "url": getattr(whisper, "_MODELS", {}).get(
            "large-v3-turbo",
            "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
        ),
        "sha256": "aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a",
    },
    {
        "id": "distil-it-v0.2",
        "filename": "whisper-large-v3-distil-it-v0.2.pt",
        "path": LOCAL_MODEL_DISTIL_IT,
        "label": "whisper-large-v3-distil-it-v0.2.pt",
        "dropdown_name": "whisper-large-v3-distil-it-v0.2.pt (Specializzato Italiano • Distil large-v3)",
        "preset": "distil-it",
        "description": "Specializzato per brani in Italiano (basato su large-v3, fino a 5.8x più rapido, meno allucinazioni) • ~1.41 GB",
        "approx_gb": 1.41,
        "recommended": True,
        "url": "https://huggingface.co/bofenghuang/whisper-large-v3-distil-it-v0.2/resolve/main/original_model.pt",
        "sha256": "22052307d94e071d9ec8fa87d779999e24d87086e6225fd973ef14b6f3539ba7",
    },
    {
        "id": "large-v2",
        "filename": "large-v2.pt",
        "path": LOCAL_MODEL_V2,
        "label": "large-v2.pt",
        "dropdown_name": "large-v2.pt (Alta Precisione • 32 Layer / 80 Mel)",
        "preset": "large-v2",
        "description": "Alta Precisione Alternativa (32 Layer / 80 Mel) • ~2.87 GB",
        "approx_gb": 2.87,
        "recommended": False,
        "url": getattr(whisper, "_MODELS", {}).get(
            "large-v2",
            "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
        ),
        "sha256": "81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524",
    },
    {
        "id": "medium",
        "filename": "medium.pt",
        "path": LOCAL_MODEL_MEDIUM,
        "label": "medium.pt",
        "dropdown_name": "medium.pt (Bilanciato • 24 Layer / GPU 6GB)",
        "preset": "medium",
        "description": "Buon compromesso multilingua per GPU con 6 GB VRAM • ~1.42 GB",
        "approx_gb": 1.42,
        "recommended": False,
        "url": getattr(whisper, "_MODELS", {}).get(
            "medium",
            "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
        ),
        "sha256": "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1",
    },
    {
        "id": "small",
        "filename": "small.pt",
        "path": LOCAL_MODEL_SMALL,
        "label": "small.pt",
        "dropdown_name": "small.pt (Leggero • 12 Layer / CPU & GPU 4GB)",
        "preset": "small",
        "description": "Leggero e velocissimo per PC con poca VRAM o senza GPU • ~0.46 GB",
        "approx_gb": 0.46,
        "recommended": False,
        "url": getattr(whisper, "_MODELS", {}).get(
            "small",
            "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
        ),
        "sha256": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
    },
]

DEMUCS_MODELS_META = {
    "htdemucs": {
        "id": "htdemucs",
        "label": "htdemucs (Standard Veloce • 8.93 dB Vocals SDR)",
        "checkpoints": ["955717e8-8726e21a.th"],
    },
    "htdemucs_ft": {
        "id": "htdemucs_ft",
        "label": "htdemucs_ft (Massima Precisione Vocale • 9.20 dB Vocals SDR, 4x più lento)",
        "checkpoints": [
            "f7e0c4bc-ba3fe64a.th",
            "d12395a8-e57c48e6.th",
            "92cfc3b6-ef3bcb9c.th",
            "04573f0d-f3cf25b2.th",
        ],
    },
}


def get_demucs_models_status() -> dict:
    ckpt_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    out = {}
    for mid, meta in DEMUCS_MODELS_META.items():
        files = meta["checkpoints"]
        installed = all(os.path.exists(os.path.join(ckpt_dir, fn)) for fn in files)
        out[mid] = {
            "id": mid,
            "label": meta["label"],
            "installed": installed,
        }
    return out


def get_available_models_list() -> list:
    out = []
    for item in WHISPER_MODEL_CATALOG:
        if os.path.exists(item["path"]):
            out.append({
                "path": item["path"],
                "name": item["dropdown_name"],
                "preset": item["preset"],
            })
    return out


def get_whisper_models_status(active_id: str = "", active_pct: float = 0.0) -> list:
    res = []
    for item in WHISPER_MODEL_CATALOG:
        p = item["path"]
        exists = os.path.exists(p) and os.path.getsize(p) > 100_000_000
        size_gb = round(os.path.getsize(p) / (1024 ** 3), 2) if exists else 0.0
        res.append({
            "id": item["id"],
            "filename": item["filename"],
            "path": p,
            "label": item["label"],
            "description": item["description"],
            "approx_gb": item["approx_gb"],
            "recommended": item["recommended"],
            "installed": exists,
            "size_gb": size_gb,
            "downloading": bool(active_id and active_id == item["id"]),
            "progress_pct": active_pct if active_id == item["id"] else 0.0,
        })
    return res


def get_best_available_model_path() -> str:
    if os.path.exists(LOCAL_MODEL_V3):
        return LOCAL_MODEL_V3
    if os.path.exists(LOCAL_MODEL_V2):
        return LOCAL_MODEL_V2
    if os.path.exists(LOCAL_MODEL_TURBO):
        return LOCAL_MODEL_TURBO
    return "large-v3"


DEFAULT_MODEL_PATH = get_best_available_model_path()
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
UNISON_KEY_FILE = os.path.join(BASE_DIR, "unison_identity.json")
YT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "MetroSync_YTCache")
OUTPUT_ROOT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
YT_OUTPUT_DIR = OUTPUT_ROOT_DIR
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
DEFAULT_OUTRO_TEXT = "MetroSync Studio"


def _build_song_folder_name(title: str = "", artist: str = "", fallback: str = "") -> str:
    t = str(title or "").strip()
    a = str(artist or "").strip()
    if a and t:
        raw = f"{a} - {t}"
    elif t:
        raw = t
    elif fallback:
        raw = str(fallback).strip()
    else:
        raw = "Senza Titolo"
    cleaned = re.sub(r'[\\/*?:"<>|]', "", raw).strip(" .")
    return cleaned or "Senza Titolo"


def _get_song_output_dir(title: str = "", artist: str = "", fallback: str = "") -> str:
    folder = _build_song_folder_name(title=title, artist=artist, fallback=fallback)
    path = os.path.join(OUTPUT_ROOT_DIR, folder)
    os.makedirs(path, exist_ok=True)
    return path

if os.path.isdir(TOOLS_DIR) and TOOLS_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = TOOLS_DIR + os.pathsep + os.environ.get("PATH", "")

YTDLP_OFFICIAL_EXE_URL = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"
YTDLP_OFFICIAL_SHA256_URL = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/SHA2-256SUMS"
FFMPEG_OFFICIAL_ZIP_URL = "https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
FFMPEG_OFFICIAL_SHA256_URL = "https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest/checksums.sha256"

LANGUAGES = {
    "🌐 Auto / Multilingua (Bilingue)": "auto",
    "🇮🇹 Italiano (it)": "it",
    "🇬🇧 Inglese (en)": "en",
    "🇪🇸 Spagnolo (es)": "es",
    "🇫🇷 Francese (fr)": "fr",
    "🇩🇪 Tedesco (de)": "de",
    "🇵🇹 Portoghese (pt)": "pt",
    "🇯🇵 Giapponese (ja)": "ja",
    "🇰🇷 Coreano (ko)": "ko",
}

LRCLIB_CLIENT_IDENTIFIER = "MetroSync-Studio/1.0.0 (https://github.com/tranxuanthang/lrclib)"
UNISON_CLIENT_IDENTIFIER = "MetroSync-Studio/1.0.0 (https://github.com/better-lyrics/unison)"

DEFAULT_PUBLISHING_CONFIG = {
    "lrclib_url": "https://lrclib.net",
    "lrclib_auth_mode": "auto_pow",
    "lrclib_token": "",
    "lrclib_client_id": LRCLIB_CLIENT_IDENTIFIER,
    "unison_url": "https://unison.betterlyrics.org",
    "unison_auth_mode": "ecdsa_identity",
    "unison_bearer_token": "",
    "unison_client_id": UNISON_CLIENT_IDENTIFIER,
    "unison_label": "",
}

SECTION_HEADER_RE = re.compile(
    r"^\[?\s*(?:intro|verse|strofa|chorus|ritornello|bridge|outro|instrumental|strumentale|solo|drop|pre-chorus|pre-ritornello|special|hook|parte\s+\d+|post-chorus|refrain|couplet|pont|estribillo|interlude|skit|breakdown)\b.*\]?$",
    re.IGNORECASE,
)

SHORT_FUNCTION_WORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "e", "ed", "o", "ma", "se", "che", "non", "ne", "né",
    "mi", "ti", "si", "ci", "vi", "li", "al", "del", "nel", "sul", "col",
    "the", "to", "of", "in", "on", "at", "by", "for", "and", "or", "is", "it", "an",
    "el", "de", "en", "y", "que", "du", "au", "et", "ou",
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(data):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def _normalize_service_url(url: str, default_url: str) -> str:
    raw = str(url or "").strip().rstrip("/")
    if not raw:
        return default_url
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return default_url
    return raw


def get_publishing_config(override: dict = None) -> dict:
    cfg = load_config()
    saved = cfg.get("publishing") if isinstance(cfg.get("publishing"), dict) else {}
    merged = dict(DEFAULT_PUBLISHING_CONFIG)
    for k, v in saved.items():
        if k in merged and v is not None:
            merged[k] = str(v).strip()
    if isinstance(override, dict):
        for k, v in override.items():
            if k in merged and v is not None:
                merged[k] = str(v).strip()

    merged["lrclib_url"] = _normalize_service_url(merged.get("lrclib_url", ""), DEFAULT_PUBLISHING_CONFIG["lrclib_url"])
    merged["unison_url"] = _normalize_service_url(merged.get("unison_url", ""), DEFAULT_PUBLISHING_CONFIG["unison_url"])
    if merged.get("lrclib_auth_mode") not in ("auto_pow", "manual_token", "bearer_auth"):
        merged["lrclib_auth_mode"] = "auto_pow"
    if merged.get("unison_auth_mode") not in ("ecdsa_identity", "ecdsa_plus_bearer"):
        merged["unison_auth_mode"] = "ecdsa_identity"
    if not merged.get("lrclib_client_id"):
        merged["lrclib_client_id"] = LRCLIB_CLIENT_IDENTIFIER
    if not merged.get("unison_client_id"):
        merged["unison_client_id"] = UNISON_CLIENT_IDENTIFIER
    return merged


def extract_youtube_id(url_or_id: str) -> str:
    s = url_or_id.strip()
    if len(s) == 11 and re.fullmatch(r"[0-9A-Za-z_-]{11}", s):
        return s
    m = re.search(r"(?:v=|\/embed\/|\/v\/|youtu\.be\/|\/shorts\/)([0-9A-Za-z_-]{11})", s)
    if m:
        return m.group(1)
    return s


def is_youtube_audio_source(url_or_id: str) -> bool:
    s = (url_or_id or "").strip()
    if not s:
        return False
    if len(s) == 11 and re.fullmatch(r"[0-9A-Za-z_-]{11}", s):
        return True
    parsed = urllib.parse.urlparse(s)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        host = parsed.netloc.lower()
        return any(d in host for d in ("youtube.com", "youtu.be", "youtube-nocookie.com"))
    return False


def _find_ffmpeg_dir() -> str:
    if os.path.exists(os.path.join(TOOLS_DIR, "ffmpeg.exe")):
        return TOOLS_DIR
    if os.path.exists(os.path.join(BASE_DIR, "ffmpeg.exe")):
        return BASE_DIR
    ff = shutil.which("ffmpeg")
    if ff:
        return os.path.dirname(ff)
    return ""


def _find_ytdlp_exe() -> str:
    tools_exe = os.path.join(TOOLS_DIR, "yt-dlp.exe")
    if os.path.exists(tools_exe):
        return tools_exe
    local_exe = os.path.join(BASE_DIR, "yt-dlp.exe")
    if os.path.exists(local_exe):
        return local_exe
    exe = shutil.which("yt-dlp")
    if exe:
        return exe
    return ""


def _fetch_official_sha256(sums_url: str, target_filename: str) -> str:
    try:
        req = urllib.request.Request(sums_url, headers={"User-Agent": "MetroSync-Studio/1.0.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                digest = parts[0].strip().lower()
                fname = parts[-1].lstrip("*").strip()
                if os.path.basename(fname).lower() == target_filename.lower() and re.fullmatch(r"[0-9a-f]{64}", digest):
                    return digest
    except Exception:
        pass
    return ""


def _download_official_binary(url: str, dest_path: str, expected_sha256: str = "", log_func=None, label: str = ""):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    part_path = f"{dest_path}.part"
    req = urllib.request.Request(url, headers={"User-Agent": "MetroSync-Studio/1.0.0"})
    hasher = hashlib.sha256()
    downloaded = 0
    last_pct = -1
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        with open(part_path, "wb") as out_f:
            while True:
                chunk = resp.read(512 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)
                hasher.update(chunk)
                downloaded += len(chunk)
                if log_func and total > 0:
                    pct = int(round(downloaded * 100.0 / total))
                    if pct >= last_pct + 20:
                        last_pct = pct
                        log_func(f"⬇️ Download {label or os.path.basename(dest_path)}: {pct}%")

    got_sha = hasher.hexdigest().lower()
    if expected_sha256 and got_sha != expected_sha256.lower():
        try:
            os.remove(part_path)
        except Exception:
            pass
        raise Exception(f"Checksum SHA-256 non valido per {os.path.basename(dest_path)}.")
    os.replace(part_path, dest_path)


def _ensure_ffmpeg_dir(log_func=None) -> str:
    existing = _find_ffmpeg_dir()
    if existing:
        if existing not in os.environ.get("PATH", ""):
            os.environ["PATH"] = existing + os.pathsep + os.environ.get("PATH", "")
        return existing

    os.makedirs(TOOLS_DIR, exist_ok=True)
    if log_func:
        log_func("📦 FFmpeg non trovato sul PC: download dalla fonte ufficiale (yt-dlp/FFmpeg-Builds)...")
    zip_path = os.path.join(TOOLS_DIR, "ffmpeg-win64-gpl.zip")
    expected_sha = _fetch_official_sha256(FFMPEG_OFFICIAL_SHA256_URL, "ffmpeg-master-latest-win64-gpl.zip")
    _download_official_binary(FFMPEG_OFFICIAL_ZIP_URL, zip_path, expected_sha256=expected_sha, log_func=log_func, label="FFmpeg ufficiale")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                base_n = os.path.basename(member).lower()
                if base_n in ("ffmpeg.exe", "ffprobe.exe"):
                    dest_bin = os.path.join(TOOLS_DIR, base_n)
                    with zf.open(member) as src, open(dest_bin, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    finally:
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass

    if not os.path.exists(os.path.join(TOOLS_DIR, "ffmpeg.exe")):
        raise Exception("Impossibile estrarre ffmpeg.exe dall'archivio ufficiale.")

    if TOOLS_DIR not in os.environ.get("PATH", ""):
        os.environ["PATH"] = TOOLS_DIR + os.pathsep + os.environ.get("PATH", "")
    if log_func:
        log_func("✅ FFmpeg ufficiale installato in MetroSync/tools!")
    return TOOLS_DIR


def _ensure_ytdlp_exe(log_func=None) -> str:
    existing = _find_ytdlp_exe()
    if existing:
        return existing

    os.makedirs(TOOLS_DIR, exist_ok=True)
    if log_func:
        log_func("📦 yt-dlp non trovato: download dalla fonte ufficiale (github.com/yt-dlp/yt-dlp)...")
    dest_exe = os.path.join(TOOLS_DIR, "yt-dlp.exe")
    expected_sha = _fetch_official_sha256(YTDLP_OFFICIAL_SHA256_URL, "yt-dlp.exe")
    _download_official_binary(YTDLP_OFFICIAL_EXE_URL, dest_exe, expected_sha256=expected_sha, log_func=log_func, label="yt-dlp.exe ufficiale")
    if log_func:
        log_func("✅ yt-dlp ufficiale installato in MetroSync/tools!")
    return dest_exe


def _parse_yt_info_metadata(info: dict, fallback_vid: str = "") -> dict:
    raw_title = str(info.get("track") or info.get("title") or "").strip()
    raw_artist = str(info.get("artist") or info.get("creator") or info.get("uploader") or info.get("channel") or "").strip()
    raw_artist = re.sub(r"\s*-\s*Topic$", "", raw_artist, flags=re.IGNORECASE).strip()
    album = str(info.get("album") or "").strip()
    duration = int(round(float(info.get("duration") or 0)))
    vid = str(info.get("id") or fallback_vid or "").strip()

    title, artist = raw_title, raw_artist
    if title and " - " in title:
        left, right = [p.strip() for p in title.split(" - ", 1)]
        if left and right and (not artist or artist.lower() in left.lower() or info.get("track") is None):
            if not info.get("track"):
                artist = left
                title = right

    writer_items = []
    desc = str(info.get("description") or "")
    for line in desc.splitlines():
        m_cred = re.match(r"^(?:Lyricist|Lyrics(?:\s+by)?|Writer|Songwriter|Composer|Author|Autore)\s*:\s*(.+)$", line.strip(), re.IGNORECASE)
        if m_cred:
            for sw in _split_songwriters(m_cred.group(1)):
                if sw.lower() not in {x.lower() for x in writer_items}:
                    writer_items.append(sw)

    return {
        "title": title,
        "artist": artist,
        "album": album,
        "duration": duration,
        "youtube": f"https://music.youtube.com/watch?v={vid}" if len(vid) == 11 else str(info.get("webpage_url") or ""),
        "songwriters": ", ".join(writer_items),
        "notes": "",
        "language": "",
    }


def download_youtube_audio_for_sync(url_or_id: str, log_func=None, progress_cb=None) -> dict:
    """Scarica in background l'audio da YouTube/YouTube Music usando yt-dlp (cache locale in %TEMP%) per usarlo subito in MetroSync."""
    raw_in = str(url_or_id or "").strip()
    if not is_youtube_audio_source(raw_in):
        raise Exception("Link o Video ID YouTube non valido.")

    vid = extract_youtube_id(raw_in)
    if len(vid) == 11 and re.fullmatch(r"[0-9A-Za-z_-]{11}", vid):
        target_url = f"https://www.youtube.com/watch?v={vid}"
        cache_key = vid
    else:
        target_url = raw_in
        cache_key = hashlib.sha1(raw_in.encode("utf-8")).hexdigest()[:12]

    vid_dir = os.path.join(YT_CACHE_DIR, cache_key)
    os.makedirs(vid_dir, exist_ok=True)
    meta_cache_path = os.path.join(vid_dir, "meta.json")

    # Cache hit veloce se già scaricato in questa sessione/PC
    cached_audio = ""
    for ext in (".mp3", ".m4a", ".opus", ".wav", ".flac", ".ogg"):
        cand = os.path.join(vid_dir, f"{cache_key}{ext}")
        if os.path.exists(cand) and os.path.getsize(cand) > 32768:
            cached_audio = cand
            break
    if not cached_audio:
        for fn in os.listdir(vid_dir):
            if fn.lower().endswith((".mp3", ".m4a", ".opus", ".wav", ".flac", ".ogg")) and os.path.getsize(os.path.join(vid_dir, fn)) > 32768:
                cached_audio = os.path.join(vid_dir, fn)
                break

    if cached_audio and os.path.exists(meta_cache_path):
        try:
            with open(meta_cache_path, "r", encoding="utf-8") as mf:
                cached_meta = json.load(mf)
            if log_func:
                log_func(f"⚡ Audio YouTube già pronto in cache: {cached_meta.get('artist', '')} - {cached_meta.get('title', cache_key)}")
            if progress_cb:
                progress_cb(12)
            base_raw = f"{cached_meta.get('artist', '')} - {cached_meta.get('title', '')}".strip(" -") or cache_key
            base_name = re.sub(r'[\\/*?:"<>|]', "", base_raw).strip() or cache_key
            return {"audio_path": cached_audio, "meta": cached_meta, "base_name": base_name, "video_id": cache_key, "from_cache": True}
        except Exception:
            pass

    if log_func:
        log_func(f"🎬 Download audio da YouTube in background tramite yt-dlp ({cache_key})...")
    if progress_cb:
        progress_cb(4)

    ffmpeg_dir = _ensure_ffmpeg_dir(log_func=log_func)
    info_dict = None
    out_audio_path = ""

    tools_ytdlp = os.path.join(TOOLS_DIR, "yt-dlp.exe")
    prefer_official_exe = os.path.exists(tools_ytdlp)
    try:
        import yt_dlp
        has_py_ytdlp = not prefer_official_exe
    except Exception:
        has_py_ytdlp = False

    if has_py_ytdlp:
        last_pct = [-1]

        def _hook(d):
            st = d.get("status")
            if st == "downloading":
                tot = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                dl = d.get("downloaded_bytes") or 0
                if tot > 0:
                    pct_dl = int(round(dl * 100.0 / tot))
                    if pct_dl >= last_pct[0] + 15:
                        last_pct[0] = pct_dl
                        if log_func:
                            log_func(f"⬇️ Download audio YouTube: {pct_dl}%")
                        if progress_cb:
                            progress_cb(4 + int(pct_dl * 0.08))
            elif st == "finished":
                if log_func:
                    log_func("🔄 Conversione audio con FFmpeg in corso...")

        ydl_opts = {
            "format": "bestaudio/best",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "outtmpl": os.path.join(vid_dir, f"{cache_key}.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "0",
                },
                {"key": "FFmpegMetadata", "add_metadata": True},
            ],
            "progress_hooks": [_hook],
        }
        if ffmpeg_dir:
            ydl_opts["ffmpeg_location"] = ffmpeg_dir

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(target_url, download=True)
    else:
        ytdlp_exe = _ensure_ytdlp_exe(log_func=log_func)
        cmd = [
            ytdlp_exe,
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "--embed-metadata",
            "--no-playlist",
            "--print-json",
            "-o", os.path.join(vid_dir, f"{cache_key}.%(ext)s"),
            target_url,
        ]
        if ffmpeg_dir:
            cmd.extend(["--ffmpeg-location", ffmpeg_dir])
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", creationflags=creationflags)
        if proc.returncode != 0:
            raise Exception(f"yt-dlp errore: {(proc.stderr or proc.stdout).strip()[:400]}")
        for line in proc.stdout.splitlines():
            line_s = line.strip()
            if line_s.startswith("{"):
                try:
                    info_dict = json.loads(line_s)
                except Exception:
                    pass

    for ext in (".mp3", ".m4a", ".opus", ".wav", ".flac", ".ogg"):
        cand = os.path.join(vid_dir, f"{cache_key}{ext}")
        if os.path.exists(cand) and os.path.getsize(cand) > 32768:
            out_audio_path = cand
            break
    if not out_audio_path:
        for fn in os.listdir(vid_dir):
            if fn.lower().endswith((".mp3", ".m4a", ".opus", ".wav", ".flac", ".ogg")):
                out_audio_path = os.path.join(vid_dir, fn)
                break

    if not out_audio_path or not os.path.exists(out_audio_path):
        raise Exception("Download YouTube completato ma nessun file audio trovato.")

    file_meta = extract_audio_metadata(out_audio_path)
    yt_meta = _parse_yt_info_metadata(info_dict or {}, fallback_vid=cache_key)
    merged_meta = dict(yt_meta)
    for k, v in file_meta.items():
        if v and not merged_meta.get(k):
            merged_meta[k] = v
    if not merged_meta.get("duration") and file_meta.get("duration"):
        merged_meta["duration"] = file_meta["duration"]

    try:
        with open(meta_cache_path, "w", encoding="utf-8") as mf:
            json.dump(merged_meta, mf, indent=2, ensure_ascii=False)
    except Exception:
        pass

    base_raw = f"{merged_meta.get('artist', '')} - {merged_meta.get('title', '')}".strip(" -") or cache_key
    base_name = re.sub(r'[\\/*?:"<>|]', "", base_raw).strip() or cache_key

    if log_func:
        log_func(f"✅ Audio YouTube pronto: {merged_meta.get('artist', '')} - {merged_meta.get('title', cache_key)}")
    if progress_cb:
        progress_cb(12)

    return {
        "audio_path": out_audio_path,
        "meta": merged_meta,
        "base_name": base_name,
        "video_id": cache_key,
        "from_cache": False,
    }


LRC_META_TAG_RE = re.compile(
    r"^\[(?:ti|ar|al|au|by|re|ve|la|length|offset|tool)\s*:.*\]$|^\[#.*\]$",
    re.IGNORECASE,
)

ISO3_TO_ISO2 = {
    "ita": "it", "eng": "en", "spa": "es", "fra": "fr", "fre": "fr",
    "deu": "de", "ger": "de", "por": "pt", "jpn": "ja", "kor": "ko",
}


def _normalize_lang_code(raw_lang: str) -> str:
    s = (raw_lang or "").strip().lower()
    if not s or s in ("auto", "🌐 auto / multilingua (bilingue)", "und", "mul"):
        return ""
    s = re.split(r"[-_;\s,]", s)[0]
    if s in ISO3_TO_ISO2:
        return ISO3_TO_ISO2[s]
    if s in ("it", "en", "es", "fr", "de", "pt", "ja", "ko"):
        return s
    return ""


def _split_songwriters(raw: str) -> list:
    if not raw:
        return []
    parts = re.split(r"[,;\n]+|\s+/\s+", str(raw))
    seen = set()
    out = []
    for p in parts:
        clean = " ".join(p.split()).strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def _find_youtube_in_text(text: str) -> str:
    if not text:
        return ""
    m = re.search(
        r"https?://(?:www\.|music\.)?(?:youtube\.com/watch\?[^\s\]\)>\"']*v=[0-9A-Za-z_-]{11}[^\s\]\)>\"']*|youtu\.be/[0-9A-Za-z_-]{11}[^\s\]\)>\"']*)",
        str(text),
        re.IGNORECASE,
    )
    return m.group(0).strip() if m else ""


def _tag_vals_to_list(val) -> list:
    if val is None:
        return []
    if hasattr(val, "text"):
        val = val.text
    if isinstance(val, (list, tuple)):
        out = []
        for item in val:
            if isinstance(item, bytes):
                out.append(item.decode("utf-8", errors="ignore").strip())
            elif hasattr(item, "text"):
                out.extend(_tag_vals_to_list(item.text))
            else:
                s = str(item).strip()
                if s:
                    out.append(s)
        return [x for x in out if x]
    if isinstance(val, bytes):
        s = val.decode("utf-8", errors="ignore").strip()
        return [s] if s else []
    s = str(val).strip()
    return [s] if s else []


def extract_audio_metadata(file_path: str) -> dict:
    """Estrae titolo, artista, album, durata, link YouTube, parolieri e lingua dai tag audio e dal nome file."""
    meta = {
        "title": "",
        "artist": "",
        "album": "",
        "duration": 0,
        "youtube": "",
        "songwriters": "",
        "notes": "",
        "language": "",
    }
    if not file_path:
        return meta

    text_blobs = []
    writer_items = []

    if HAS_MUTAGEN and os.path.exists(file_path):
        try:
            mf = mutagen.File(file_path)
            if mf is not None:
                if getattr(mf, "info", None) and getattr(mf.info, "length", 0):
                    meta["duration"] = int(round(float(mf.info.length)))

                tags = getattr(mf, "tags", None)
                if tags:
                    def _first(*keys):
                        for k in keys:
                            for tk in tags.keys():
                                if tk.lower() == k.lower():
                                    vals = _tag_vals_to_list(tags[tk])
                                    if vals:
                                        return vals[0]
                        return ""

                    def _all(*keys):
                        res = []
                        for k in keys:
                            for tk in tags.keys():
                                if tk.lower() == k.lower() or tk.lower().startswith(k.lower() + ":"):
                                    res.extend(_tag_vals_to_list(tags[tk]))
                        return res

                    meta["title"] = _first("TIT2", "©nam", "\xa9nam", "title")
                    meta["artist"] = _first("TPE1", "©ART", "\xa9ART", "artist", "TPE2", "aART", "albumartist")
                    meta["album"] = _first("TALB", "©alb", "\xa9alb", "album")
                    raw_lang = _first("TLAN", "language", "----:com.apple.iTunes:LANGUAGE", "©lan", "\xa9lan")
                    meta["language"] = _normalize_lang_code(raw_lang)

                    # Parolieri / Compositori espliciti nei tag
                    for wv in _all("TEXT", "TOLY", "©wrt", "\xa9wrt", "lyricist", "songwriter", "writer", "author", "TCOM", "composer",
                                   "TXXX:lyricist", "TXXX:songwriter", "TXXX:writer", "----:com.apple.iTunes:LYRICIST", "----:com.apple.iTunes:SONGWRITER"):
                        for sw in _split_songwriters(wv):
                            if sw.lower() not in {x.lower() for x in writer_items}:
                                writer_items.append(sw)

                    # Raccogli campi commento / URL / descrizione per YouTube ID e crediti
                    for ck in tags.keys():
                        cl = ck.lower()
                        if any(tok in cl for tok in ("cmt", "comm", "purl", "url", "woas", "desc", "synopsis", "comment")):
                            for blob in _tag_vals_to_list(tags[ck]):
                                text_blobs.append(blob)
                                if not meta["youtube"]:
                                    yt = _find_youtube_in_text(blob)
                                    if yt:
                                        meta["youtube"] = yt
        except Exception:
            pass

    # Cerca YouTube ID o crediti (Lyricist / Writer / Composer) nelle descrizioni YouTube
    for blob in text_blobs:
        for line in blob.splitlines():
            m_cred = re.match(r"^(?:Lyricist|Lyrics(?:\s+by)?|Writer|Songwriter|Composer|Author|Autore)\s*:\s*(.+)$", line.strip(), re.IGNORECASE)
            if m_cred:
                for sw in _split_songwriters(m_cred.group(1)):
                    if sw.lower() not in {x.lower() for x in writer_items}:
                        writer_items.append(sw)

    if writer_items:
        meta["songwriters"] = ", ".join(writer_items)

    # Pulizia titolo se contiene "Artista - Titolo" (es. video YT scaricati)
    if meta["title"] and " - " in meta["title"]:
        left, right = [p.strip() for p in meta["title"].split(" - ", 1)]
        if left and right:
            if not meta["artist"] or meta["artist"].lower() in left.lower():
                meta["artist"] = left if len(left) >= len(meta["artist"]) else meta["artist"]
                meta["title"] = right

    # Fallback intelligente dal nome file ("Artista - Titolo", "Titolo by Artista", suffisso _12345)
    base = os.path.splitext(os.path.basename(file_path))[0].strip()
    if not meta["youtube"]:
        m_yt_fn = re.search(r"[\[\(]([0-9A-Za-z_-]{11})[\]\)]", base)
        if m_yt_fn:
            meta["youtube"] = f"https://music.youtube.com/watch?v={m_yt_fn.group(1)}"
            base = (base[:m_yt_fn.start()] + base[m_yt_fn.end():]).strip()

    clean_base = re.sub(r"_\d{4,}$", "", base).strip()
    fn_artist, fn_title = "", clean_base
    if re.search(r"\s+[-–—]\s+", clean_base):
        parts = re.split(r"\s+[-–—]\s+", clean_base, maxsplit=1)
        fn_artist, fn_title = parts[0].strip(), parts[1].strip()
    else:
        m_by = re.match(r"^(.+?)\s+by\s+(.+)$", clean_base, re.IGNORECASE)
        if m_by:
            fn_title, fn_artist = m_by.group(1).strip(), m_by.group(2).strip()

    if not meta["title"]:
        meta["title"] = fn_title
    if not meta["artist"] and fn_artist:
        meta["artist"] = fn_artist

    return meta


def extract_lyrics_metadata(content: str) -> dict:
    """Estrae eventuali metadati (titolo, artista, album, autore, note, lingua) da file LRC, TTML, YAML o TXT."""
    meta = {
        "title": "",
        "artist": "",
        "album": "",
        "songwriters": "",
        "notes": "",
        "language": "",
    }
    if not content:
        return meta

    low = content.lower()
    if "<tt" in low:
        m_lang = re.search(r'xml:lang\s*=\s*"([^"]+)"', content, re.IGNORECASE)
        if m_lang:
            meta["language"] = _normalize_lang_code(m_lang.group(1))
        m_ti = re.search(r"<ttm:title[^>]*>(.*?)</ttm:title>", content, re.DOTALL | re.IGNORECASE)
        if m_ti:
            meta["title"] = saxutils.unescape(m_ti.group(1)).strip()
        m_desc = re.search(r"<ttm:desc[^>]*>(.*?)</ttm:desc>", content, re.DOTALL | re.IGNORECASE)
        if m_desc:
            meta["notes"] = saxutils.unescape(m_desc.group(1)).strip()
        sws = [saxutils.unescape(s).strip() for s in re.findall(r"<songwriter[^>]*>(.*?)</songwriter>", content, re.DOTALL | re.IGNORECASE)]
        sws = [s for s in sws if s]
        if sws:
            meta["songwriters"] = ", ".join(_split_songwriters(", ".join(sws)))
        return meta

    # Controlla formato YAML (lyricsfile)
    if re.search(r"^\s*version\s*:", content, re.MULTILINE) and re.search(r"^\s*metadata\s*:", content, re.MULTILINE):
        def _unquote_yaml(s: str) -> str:
            s = s.strip()
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                try:
                    return str(json.loads(s)) if s.startswith('"') else s[1:-1]
                except Exception:
                    return s[1:-1]
            return s

        in_meta = False
        in_sw = False
        sw_list = []
        for raw_l in content.splitlines():
            if re.match(r"^\s*metadata\s*:\s*$", raw_l):
                in_meta = True
                continue
            if in_meta and raw_l and not raw_l.startswith(" ") and not raw_l.startswith("\t"):
                break
            if not in_meta:
                continue
            if re.match(r"^\s{2}songwriters\s*:\s*$", raw_l):
                in_sw = True
                continue
            if in_sw:
                m_item = re.match(r"^\s{4}-\s*(.+)$", raw_l)
                if m_item:
                    val = _unquote_yaml(m_item.group(1))
                    if val:
                        sw_list.append(val)
                    continue
                in_sw = False
            m_kv = re.match(r"^\s{2}(title|artist|album|language|author|notes)\s*:\s*(.+)$", raw_l)
            if m_kv:
                k, v = m_kv.group(1), _unquote_yaml(m_kv.group(2))
                if k == "language":
                    meta["language"] = _normalize_lang_code(v)
                elif k == "author":
                    for sw in _split_songwriters(v):
                        sw_list.append(sw)
                else:
                    meta[k] = v
        if sw_list:
            meta["songwriters"] = ", ".join(_split_songwriters(", ".join(sw_list)))
        return meta

    # Controlla tag LRC [ti:...], [ar:...], [al:...], [au:...], [by:...], [la:...], [#...]
    notes_parts = []
    for raw_l in content.splitlines():
        s = raw_l.strip()
        m_tag = re.match(r"^\[([a-zA-Z]+)\s*:\s*(.*)\]$", s)
        if m_tag:
            k, v = m_tag.group(1).lower(), m_tag.group(2).strip()
            if k == "ti" and not meta["title"]:
                meta["title"] = v
            elif k == "ar" and not meta["artist"]:
                meta["artist"] = v
            elif k == "al" and not meta["album"]:
                meta["album"] = v
            elif k == "au" and not meta["songwriters"]:
                meta["songwriters"] = ", ".join(_split_songwriters(v))
            elif k == "la" and not meta["language"]:
                meta["language"] = _normalize_lang_code(v)
            elif k == "by" and v and v.lower() != "metrosync":
                notes_parts.append(v)
            continue
        m_note = re.match(r"^\[#\s*(?:Note:\s*)?(.*)\]$", s, re.IGNORECASE)
        if m_note and m_note.group(1).strip():
            notes_parts.append(m_note.group(1).strip())
    if notes_parts:
        meta["notes"] = " | ".join(notes_parts)
    return meta


def parse_time_to_seconds(time_str: str) -> float:
    time_str = time_str.strip().replace("s", "")
    if ":" in time_str:
        parts = time_str.split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
    return float(time_str)


def _b64url_to_bytes(val: str) -> bytes:
    s = str(val or "").strip()
    pad = "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _private_key_to_identity_bundle(private_key) -> dict:
    pub_numbers = private_key.public_key().public_numbers()
    priv_numbers = private_key.private_numbers()

    x_bytes = pub_numbers.x.to_bytes(32, byteorder="big")
    y_bytes = pub_numbers.y.to_bytes(32, byteorder="big")
    d_bytes = priv_numbers.private_value.to_bytes(32, byteorder="big")

    x_b64 = base64.urlsafe_b64encode(x_bytes).rstrip(b"=").decode("ascii")
    y_b64 = base64.urlsafe_b64encode(y_bytes).rstrip(b"=").decode("ascii")
    d_b64 = base64.urlsafe_b64encode(d_bytes).rstrip(b"=").decode("ascii")

    public_jwk = {
        "crv": "P-256",
        "kty": "EC",
        "x": x_b64,
        "y": y_b64,
    }
    private_jwk = {
        "crv": "P-256",
        "d": d_b64,
        "ext": True,
        "kty": "EC",
        "x": x_b64,
        "y": y_b64,
    }
    canonical_jwk = json.dumps(public_jwk, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    key_id = hashlib.sha256(canonical_jwk).hexdigest()
    pem = private_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption(),
    ).decode("ascii")

    return {
        "private_key": private_key,
        "key_id": key_id,
        "private_key_pem": pem,
        "public_jwk": public_jwk,
        "private_jwk": private_jwk,
    }


def parse_unison_identity_input(raw_text: str) -> dict:
    """Parses MetroSync JSON, BetterLyrics JWK identity JSON, or raw ECDSA P-256 PEM."""
    if not HAS_CRYPTO:
        raise Exception(
            "Il modulo 'cryptography' non è installato.\nEsegui nel terminale: pip install cryptography"
        )
    text = (raw_text or "").strip()
    if not text:
        raise Exception("Contenuto identità vuoto.")

    if not text.startswith("{") and "-----BEGIN " in text and "PRIVATE KEY" in text:
        pk = load_pem_private_key(text.replace("\\n", "\n").encode("utf-8"), password=None)
        if not isinstance(pk, ec.EllipticCurvePrivateKey) or pk.curve.name != "secp256r1":
            raise Exception("La chiave PEM deve essere una chiave ECDSA P-256 (secp256r1).")
        return _private_key_to_identity_bundle(pk)

    try:
        data = json.loads(text)
    except Exception as ex:
        raise Exception(f"Formato JSON o PEM non valido: {ex}")

    if not isinstance(data, dict):
        raise Exception("Il file identità JSON deve essere un oggetto.")

    pem_candidate = data.get("private_key_pem") or data.get("privateKeyPem") or data.get("pem")
    if isinstance(pem_candidate, str) and "PRIVATE KEY" in pem_candidate:
        pk = load_pem_private_key(pem_candidate.encode("utf-8"), password=None)
        if not isinstance(pk, ec.EllipticCurvePrivateKey) or pk.curve.name != "secp256r1":
            raise Exception("La chiave PEM deve essere ECDSA P-256 (secp256r1).")
        return _private_key_to_identity_bundle(pk)

    jwk_candidate = None
    for key_name in ("private_jwk", "privateKeyJwk", "privateKey", "jwk", "keyPair"):
        val = data.get(key_name)
        if isinstance(val, dict) and val.get("d"):
            jwk_candidate = val
            break
        if isinstance(val, str) and "PRIVATE KEY" in val:
            pk = load_pem_private_key(val.encode("utf-8"), password=None)
            return _private_key_to_identity_bundle(pk)
    if jwk_candidate is None and data.get("d"):
        jwk_candidate = data

    if isinstance(jwk_candidate, dict) and jwk_candidate.get("d"):
        crv = str(jwk_candidate.get("crv", "P-256"))
        kty = str(jwk_candidate.get("kty", "EC"))
        if kty.upper() != "EC" or crv not in ("P-256", "secp256r1"):
            raise Exception(f"Curva JWK non supportata ({kty}/{crv}): richiesta EC P-256.")
        d_int = int.from_bytes(_b64url_to_bytes(jwk_candidate["d"]), byteorder="big")
        pk = ec.derive_private_key(d_int, ec.SECP256R1())
        return _private_key_to_identity_bundle(pk)

    raise Exception("Nessuna chiave privata ECDSA P-256 trovata (attesi 'private_key_pem' oppure JWK con campo 'd').")


def save_unison_identity_bundle(bundle: dict):
    payload = {
        "key_id": bundle["key_id"],
        "private_key_pem": bundle["private_key_pem"],
        "public_jwk": bundle["public_jwk"],
    }
    with open(UNISON_KEY_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_unison_identity_status() -> dict:
    pub_cfg = get_publishing_config()
    unison_url = pub_cfg["unison_url"]
    label = pub_cfg.get("unison_label", "")
    status = {
        "exists": False,
        "valid": False,
        "key_id": "",
        "public_jwk": None,
        "label": label,
        "curator_url": "",
        "crypto_available": HAS_CRYPTO,
    }
    if not HAS_CRYPTO or not os.path.exists(UNISON_KEY_FILE):
        return status
    try:
        with open(UNISON_KEY_FILE, "r", encoding="utf-8") as f:
            raw = f.read()
        bundle = parse_unison_identity_input(raw)
        status["exists"] = True
        status["valid"] = True
        status["key_id"] = bundle["key_id"]
        status["public_jwk"] = bundle["public_jwk"]
        status["curator_url"] = f"{unison_url}/curator/{bundle['key_id']}"
    except Exception:
        status["exists"] = os.path.exists(UNISON_KEY_FILE)
        status["valid"] = False
    return status


def generate_new_unison_identity() -> dict:
    if not HAS_CRYPTO:
        raise Exception(
            "Il modulo 'cryptography' non è installato.\nEsegui nel terminale: pip install cryptography"
        )
    private_key = ec.generate_private_key(ec.SECP256R1())
    bundle = _private_key_to_identity_bundle(private_key)
    save_unison_identity_bundle(bundle)
    return bundle


def get_or_create_unison_identity():
    if not HAS_CRYPTO:
        raise Exception(
            "Il modulo 'cryptography' non è installato.\nEsegui nel terminale: pip install cryptography"
        )

    if os.path.exists(UNISON_KEY_FILE):
        try:
            with open(UNISON_KEY_FILE, "r", encoding="utf-8") as f:
                raw = f.read()
            bundle = parse_unison_identity_input(raw)
            return bundle["private_key"], bundle["key_id"], bundle["public_jwk"]
        except Exception:
            pass

    bundle = generate_new_unison_identity()
    return bundle["private_key"], bundle["key_id"], bundle["public_jwk"]


class RealtimeStreamInterceptor:
    def __init__(self, api, original_stream):
        self.api = api
        self.orig = original_stream
        self.last_pct = -1
        self.last_log_time = 0.0

    def write(self, text):
        if self.orig:
            try:
                self.orig.write(text)
                self.orig.flush()
            except Exception:
                pass

        if not text or not self.api.is_alive:
            return

        match = re.search(r"(\d+)%", text)
        if match:
            pct = int(match.group(1))
            if pct != self.last_pct:
                self.last_pct = pct
                self.api.progress(pct)

        clean = text.replace("\r", "").strip()
        if clean:
            is_bar = any(k in clean for k in ["Align:", "Adjustment:", "Refine:", "Demucs:", "| "])
            now = time.time()
            if not is_bar:
                self.api.log(clean)
            elif (now - self.last_log_time) >= 1.0:
                self.last_log_time = now
                self.api.log(clean)

    def flush(self):
        if self.orig:
            try:
                self.orig.flush()
            except Exception:
                pass


# ==============================================================================
# HIGH-PRECISION ACOUSTIC ENGINE v3
# (Zero-Phase Transient, Pause Truncation & Vocal Sustain/Melisma Tracking)
# ==============================================================================
class AcousticPrecisionEngine:
    def __init__(self, audio_16k: np.ndarray, sr: int = 16000):
        self.sr = sr
        self.audio = audio_16k.astype(np.float32) if audio_16k is not None else np.zeros(sr, dtype=np.float32)
        if len(self.audio) == 0:
            self.audio = np.zeros(sr, dtype=np.float32)

        self.hop_len = int(sr * 0.001)           # Risoluzione 1ms (1000 fps)
        self.win_len = int(sr * 0.008)           # Finestra 8ms per transienti rapidi
        self.sustain_win_len = int(sr * 0.024)   # Finestra 24ms per vocali allungate / vibrato

        if HAS_SCIPY and len(self.audio) > int(sr * 0.1):
            try:
                sos_sib = butter(4, [4000, 7500], btype="bandpass", fs=sr, output="sos")
                sos_body = butter(4, [220, 3500], btype="bandpass", fs=sr, output="sos")
                sos_low = butter(3, [85, 250], btype="bandpass", fs=sr, output="sos")
                x_sib = sosfiltfilt(sos_sib, self.audio)
                x_body = sosfiltfilt(sos_body, self.audio)
                x_low = sosfiltfilt(sos_low, self.audio)
            except Exception:
                x_sib = self.audio.copy()
                x_sib[1:] = x_sib[1:] - 0.97 * x_sib[:-1]
                x_body = self.audio.copy()
                x_low = self.audio.copy()
        else:
            x_sib = self.audio.copy()
            x_sib[1:] = x_sib[1:] - 0.97 * x_sib[:-1]
            x_body = self.audio.copy()
            x_low = self.audio.copy()

        win_fast = np.hanning(self.win_len).astype(np.float32)
        win_fast_sum = float(np.sum(win_fast))
        if win_fast_sum > 0:
            win_fast /= win_fast_sum

        win_sus = np.hanning(self.sustain_win_len).astype(np.float32)
        win_sus_sum = float(np.sum(win_sus))
        if win_sus_sum > 0:
            win_sus /= win_sus_sum

        harmonic_power = (x_body ** 2) + 0.45 * (x_low ** 2)

        if HAS_SCIPY:
            e_sib = fftconvolve(x_sib ** 2, win_fast, mode="same")
            e_body = fftconvolve(x_body ** 2, win_fast, mode="same")
            e_low = fftconvolve(x_low ** 2, win_fast, mode="same")
            e_sustain = fftconvolve(harmonic_power, win_sus, mode="same")
        else:
            e_sib = np.convolve(x_sib ** 2, win_fast, mode="same")
            e_body = np.convolve(x_body ** 2, win_fast, mode="same")
            e_low = np.convolve(x_low ** 2, win_fast, mode="same")
            e_sustain = np.convolve(harmonic_power, win_sus, mode="same")

        self.env_sib = np.log1p(200.0 * np.maximum(0.0, e_sib[::self.hop_len]))
        self.env_body = np.log1p(80.0 * np.maximum(0.0, e_body[::self.hop_len]))
        self.env_low = np.log1p(40.0 * np.maximum(0.0, e_low[::self.hop_len]))
        self.env_sustain = np.log1p(110.0 * np.maximum(0.0, e_sustain[::self.hop_len]))
        self.num_frames = len(self.env_sib)

        self.env_total = self.env_sib + self.env_body + 0.35 * self.env_low

        if self.num_frames > 0:
            self.global_floor_total = float(np.percentile(self.env_total, 12))
            self.global_peak_total = float(np.percentile(self.env_total, 88))
            self.global_floor_sustain = float(np.percentile(self.env_sustain, 12))
            self.global_peak_sustain = float(np.percentile(self.env_sustain, 88))

            step_f = 50
            coarse_tot = self.env_total[::step_f]
            coarse_sus = self.env_sustain[::step_f]
            half_w = 60
            n_c = len(coarse_tot)
            c_floor_tot = np.empty(n_c, dtype=np.float32)
            c_floor_sus = np.empty(n_c, dtype=np.float32)
            for idx_c in range(n_c):
                s_c = max(0, idx_c - half_w)
                e_c = min(n_c, idx_c + half_w + 1)
                c_floor_tot[idx_c] = np.percentile(coarse_tot[s_c:e_c], 15)
                c_floor_sus[idx_c] = np.percentile(coarse_sus[s_c:e_c], 15)

            x_coarse = np.arange(n_c, dtype=np.float32) * step_f
            x_full = np.arange(self.num_frames, dtype=np.float32)
            self.local_floor_total = 0.65 * np.interp(x_full, x_coarse, c_floor_tot).astype(np.float32) + 0.35 * self.global_floor_total
            self.local_floor_sustain = 0.65 * np.interp(x_full, x_coarse, c_floor_sus).astype(np.float32) + 0.35 * self.global_floor_sustain
        else:
            self.global_floor_total = 0.0
            self.global_peak_total = 1.0
            self.global_floor_sustain = 0.0
            self.global_peak_sustain = 1.0
            self.local_floor_total = np.zeros(1, dtype=np.float32)
            self.local_floor_sustain = np.zeros(1, dtype=np.float32)

        diff_sib = np.maximum(0.0, np.diff(self.env_sib, prepend=self.env_sib[0]))
        diff_body = np.maximum(0.0, np.diff(self.env_body, prepend=self.env_body[0]))
        diff_low = np.maximum(0.0, np.diff(self.env_low, prepend=self.env_low[0]))
        self.odf_raw = 3.0 * diff_sib + 1.6 * diff_body + 0.5 * diff_low

        local_window = int(0.500 * sr / self.hop_len)
        if local_window > 0 and len(self.odf_raw) > local_window:
            padded = np.pad(self.odf_raw, (local_window // 2, local_window // 2), mode="reflect")
            try:
                from numpy.lib.stride_tricks import sliding_window_view
                local_max = sliding_window_view(padded, local_window).max(axis=1)[:len(self.odf_raw)]
                local_max = local_max.astype(np.float32)
            except Exception:
                local_max = np.array([
                    np.max(padded[i:i + local_window])
                    for i in range(len(self.odf_raw))
                ], dtype=np.float32)
            local_max = np.maximum(local_max, 1e-6)
            self.odf = self.odf_raw / local_max
        else:
            max_odf = float(np.max(self.odf_raw)) if len(self.odf_raw) > 0 else 0.0
            self.odf = self.odf_raw / max(max_odf, 1e-6)

    def _time_to_frame(self, t: float) -> int:
        if self.num_frames == 0:
            return 0
        return max(0, min(self.num_frames - 1, int(round(t * self.sr / self.hop_len))))

    def _frame_to_time(self, f: int) -> float:
        return round(float(f * self.hop_len / self.sr), 3)

    def detect_leading_silence_end(self, start_time: float, end_time: float) -> float:
        if self.num_frames == 0 or (end_time - start_time) < 0.24:
            return start_time

        f_start = self._time_to_frame(start_time)
        f_end = self._time_to_frame(end_time)
        if f_end - f_start < 120:
            return start_time

        seg_tot = self.env_total[f_start:f_end]
        seg_floor = self.local_floor_total[f_start:f_end]
        p92 = float(np.percentile(seg_tot, 92))
        p98 = float(np.percentile(seg_tot, 98))
        peak_e = max(p92, 0.85 * p98)
        base_floor = float(np.median(seg_floor))
        dyn = peak_e - base_floor
        if dyn < 0.18:
            return start_time

        silence_ceil = base_floor + 0.16 * dyn
        active_thresh = base_floor + 0.24 * dyn

        head_len = min(len(seg_tot), 55)
        if float(np.mean(seg_tot[:head_len])) > silence_ceil:
            return start_time

        active_run = 0
        rise_offset = -1
        max_scan = len(seg_tot) - 35
        for i in range(head_len // 2, max_scan):
            if seg_tot[i] >= active_thresh:
                active_run += 1
                if active_run >= 22:
                    rise_offset = i - active_run + 1
                    break
            else:
                active_run = 0

        if rise_offset > 45:
            adjusted_f = f_start + max(0, rise_offset - 15)
            return self._frame_to_time(adjusted_f)

        return start_time

    def refine_onset(
        self,
        target_time: float,
        min_time: float = 0.0,
        max_time: float = 9999.0,
        word_end: float = None,
    ) -> float:
        if self.num_frames == 0:
            return target_time

        anchor_time = target_time
        if word_end is not None and (word_end - target_time) > 0.24:
            skipped_t = self.detect_leading_silence_end(target_time, min(max_time, word_end))
            if skipped_t > target_time + 0.040:
                anchor_time = skipped_t

        # Evita di invadere la parola precedente nel parlato continuo:
        # lookback esteso ammesso solo quando c'è una reale pausa pre-parola (>= 120ms).
        is_isolated_phrase = (anchor_time - min_time) >= 0.120
        lookback_sec = 0.140 if is_isolated_phrase else 0.030
        w_start = max(min_time, anchor_time - lookback_sec)
        w_end = min(max_time, anchor_time + 0.045)
        if w_end <= w_start:
            return anchor_time

        f_start = self._time_to_frame(w_start)
        f_end = self._time_to_frame(w_end)
        if f_end <= f_start:
            return anchor_time

        sub_odf = self.odf[f_start:f_end]
        if len(sub_odf) == 0:
            return anchor_time

        peak_idx = int(np.argmax(sub_odf))
        peak_val = float(sub_odf[peak_idx])

        if peak_val >= 0.060:
            best_f = f_start + peak_idx
            lookback = min(best_f - f_start, 24)
            foot_f = best_f

            peak_energy = float(self.env_total[best_f]) if best_f < self.num_frames else 0.0
            odf_threshold = peak_val * 0.18

            for step_back in range(1, lookback + 1):
                check_f = best_f - step_back
                if check_f < 0:
                    break
                odf_here = float(self.odf[check_f]) if check_f < len(self.odf) else 0.0
                if odf_here < odf_threshold:
                    e_here = float(self.env_total[check_f]) if check_f < self.num_frames else 0.0
                    if peak_energy > 0 and e_here > peak_energy * 0.70:
                        continue
                    foot_f = check_f + 1
                    break

            min_f = self._time_to_frame(min_time)
            foot_f = max(foot_f, min_f)

            onset_sec = self._frame_to_time(foot_f)
            diff = onset_sec - anchor_time
            if -lookback_sec <= diff <= 0.045:
                return onset_sec

        return round(anchor_time, 3)

    def refine_offset(self, target_time: float, min_time: float = 0.0, max_time: float = 9999.0) -> float:
        w_start = max(min_time, target_time - 0.120)
        w_end = min(max_time, target_time + 0.100)
        if w_end <= w_start or self.num_frames == 0:
            return target_time

        f_start = self._time_to_frame(w_start)
        f_end = self._time_to_frame(w_end)
        if f_end <= f_start:
            return target_time

        seg_env = self.env_total[f_start:f_end]
        if len(seg_env) == 0:
            return target_time

        max_e = float(np.max(seg_env))
        min_e = float(np.min(seg_env))
        dynamic_range = max_e - min_e
        if dynamic_range < 0.15:
            return target_time

        thresh = min_e + 0.20 * dynamic_range
        target_f = int(round((target_time - w_start) * self.sr / self.hop_len))
        consecutive_below = 0
        required_consecutive = 4

        search_start = max(0, target_f - 20)
        search_end = min(len(seg_env), target_f + 25)

        for f in range(search_start, search_end):
            if seg_env[f] <= thresh:
                consecutive_below += 1
                if consecutive_below >= required_consecutive:
                    first_below_f = f - consecutive_below + 1
                    drop_sec = self._frame_to_time(f_start + first_below_f)
                    diff = drop_sec - target_time
                    if -0.100 <= diff <= 0.090:
                        return drop_sec
                    break
            else:
                consecutive_below = 0

        return target_time

    def refine_offset_and_sustain(
        self,
        word_start: float,
        target_end: float,
        min_end: float,
        max_end: float,
        word_text: str = "",
        is_line_end: bool = False,
    ) -> tuple:
        if self.num_frames == 0 or max_end <= min_end:
            return round(max(min_end, min(max_end, target_end)), 3), "normal"

        clean_w = clean_token_str(word_text)
        has_pause_punct = bool(re.search(r"[,.;:!?…\)]$", word_text.strip()))
        has_elongated_spelling = bool(re.search(r"([a-zàèéìòù])\1{2,}", word_text.lower()))
        is_short_func = (clean_w in SHORT_FUNCTION_WORDS) and not is_line_end and not has_elongated_spelling

        f_w_start = self._time_to_frame(word_start)
        f_min_end = self._time_to_frame(min_end)
        f_target = self._time_to_frame(max(min_end, min(max_end, target_end)))
        f_max_end = self._time_to_frame(max_end)

        if f_target <= f_w_start + 30:
            return self._frame_to_time(f_target), "normal"

        core_end_f = max(f_w_start + 40, min(f_target, f_w_start + 900))
        core_tot = self.env_total[f_w_start:core_end_f]
        core_sus = self.env_sustain[f_w_start:core_end_f]
        if len(core_tot) == 0:
            return self._frame_to_time(f_target), "normal"

        word_peak_tot = float(np.percentile(core_tot, 90))
        word_peak_sus = float(np.percentile(core_sus, 90))
        local_fl_tot = float(np.median(self.local_floor_total[f_w_start:max(f_w_start + 1, f_target)]))
        local_fl_sus = float(np.median(self.local_floor_sustain[f_w_start:max(f_w_start + 1, f_target)]))

        dyn_tot = max(word_peak_tot - local_fl_tot, self.global_peak_total - self.global_floor_total, 0.20)
        dyn_sus = max(word_peak_sus - local_fl_sus, self.global_peak_sustain - self.global_floor_sustain, 0.20)

        # FASE 1: RICONOSCIMENTO PAUSE & TRONCAMENTO CODA SILENZIOSA
        active_tot_thresh = local_fl_tot + (0.24 if is_short_func else 0.18) * dyn_tot
        active_sus_thresh = local_fl_sus + (0.24 if is_short_func else 0.17) * dyn_sus

        span_tot = self.env_total[f_w_start:f_target]
        span_sus = self.env_sustain[f_w_start:f_target]
        is_active = (span_tot >= active_tot_thresh) | (span_sus >= active_sus_thresh)

        max_geminate_gap = 55 if has_pause_punct else (45 if is_short_func else 75)
        n_span = len(is_active)
        idx = 0
        while idx < n_span:
            if not is_active[idx]:
                gap_s = idx
                while idx < n_span and not is_active[idx]:
                    idx += 1
                gap_len = idx - gap_s
                if gap_s > 0 and idx < n_span:
                    after_window = span_sus[idx:min(n_span, idx + 80)]
                    after_peak = float(np.max(after_window)) if len(after_window) > 0 else 0.0
                    effective_bridge = max_geminate_gap + (35 if (len(clean_w) >= 5 and after_peak >= local_fl_sus + 0.42 * dyn_sus) else 0)
                    if gap_len <= effective_bridge:
                        is_active[gap_s:idx] = True
            else:
                idx += 1

        first_active = 0
        while first_active < n_span and not is_active[first_active]:
            first_active += 1

        if first_active < n_span:
            cluster_end = first_active
            while cluster_end < n_span and is_active[cluster_end]:
                cluster_end += 1

            if not is_short_func and len(clean_w) >= 4 and cluster_end < n_span:
                rem_indices = np.where(is_active[cluster_end:])[0]
                if len(rem_indices) > 0:
                    last_act_rel = cluster_end + int(rem_indices[-1]) + 1
                    gap_between = int(rem_indices[0])
                    rem_energy = float(np.percentile(span_sus[cluster_end + int(rem_indices[0]):last_act_rel], 85))
                    if gap_between <= 115 and rem_energy >= (local_fl_sus + 0.36 * dyn_sus):
                        cluster_end = last_act_rel

            trailing_silence_ms = n_span - cluster_end
            min_pause_to_cut = 50 if (has_pause_punct or is_short_func) else 72

            if trailing_silence_ms >= min_pause_to_cut:
                cut_f = max(f_min_end, min(f_target, f_w_start + cluster_end + 18))
                return self._frame_to_time(cut_f), "pause_cut"

        if is_short_func and (f_target - f_w_start) > 550:
            max_func_f = min(f_target, f_w_start + 420)
            refined_short = self.refine_offset(self._frame_to_time(max_func_f), min_end, self._frame_to_time(max_func_f + 40))
            return max(min_end, refined_short), "pause_cut"

        # FASE 2: TRACCIAMENTO VOCALE PAROLE ALLUNGATE (SUSTAIN & MELISMA)
        tail_check_start = max(f_w_start, f_target - 45)
        tail_sus_level = float(np.mean(self.env_sustain[tail_check_start:f_target + 1])) if f_target >= tail_check_start else 0.0
        sustain_active_at_end = tail_sus_level >= (local_fl_sus + 0.24 * dyn_sus)

        max_sustain_cap_f = min(f_max_end, f_w_start + (5500 if (is_line_end or has_elongated_spelling) else 3600))

        if sustain_active_at_end and not is_short_func and (max_sustain_cap_f > f_target + 35):
            hold_thresh = local_fl_sus + (0.17 if (is_line_end or has_elongated_spelling) else 0.21) * dyn_sus
            max_vibrato_dip_ms = 75 if (is_line_end or has_elongated_spelling) else 50

            sustain_f = f_target
            dip_count = 0

            for f in range(f_target, max_sustain_cap_f):
                sus_val = float(self.env_sustain[f])
                odf_val = float(self.odf[f]) if f < len(self.odf) else 0.0

                if dip_count >= 12 and odf_val >= 0.48:
                    break

                if sus_val >= hold_thresh:
                    sustain_f = f
                    dip_count = 0
                else:
                    dip_count += 1
                    if dip_count >= max_vibrato_dip_ms:
                        break

            if sustain_f > f_target + 30:
                ext_f = min(max_sustain_cap_f, sustain_f + 15)
                return self._frame_to_time(ext_f), "sustain_ext"

        # FASE 3: MICRO-RAFFINAMENTO LOCALE DEL RILASCIO
        refined_t = self.refine_offset(self._frame_to_time(f_target), min_end, max_end)
        return refined_t, "normal"

    def detect_vocal_islands(self, merge_gap_ms: int = 260, min_dur_ms: int = 70) -> list:
        """Individua tutte le isole di energia vocale attiva sul vocal stem."""
        if self.num_frames == 0:
            return []
        dyn_tot = max(self.global_peak_total - self.global_floor_total, 0.20)
        dyn_sus = max(self.global_peak_sustain - self.global_floor_sustain, 0.20)
        th_tot = self.local_floor_total + max(0.08, 0.14 * dyn_tot)
        th_sus = self.local_floor_sustain + max(0.06, 0.12 * dyn_sus)
        active = (self.env_total >= th_tot) | (self.env_sustain >= th_sus)

        islands = []
        i = 0
        n = self.num_frames
        while i < n:
            if active[i]:
                s = i
                while i < n and active[i]:
                    i += 1
                islands.append([s, i])
            else:
                i += 1

        merged = []
        for s, e in islands:
            if merged and (s - merged[-1][1]) <= merge_gap_ms:
                merged[-1][1] = e
            else:
                merged.append([s, e])

        out = []
        for s, e in merged:
            if (e - s) >= min_dur_ms:
                pk = float(np.max(self.env_total[s:e]))
                if pk >= self.global_floor_total + 0.18 * dyn_tot:
                    out.append((self._frame_to_time(s), self._frame_to_time(e)))
        return out

    def snap_onset_micro(
        self,
        ctc_t: float,
        min_t: float,
        max_t: float,
        max_shift: float = 0.040,
        max_shift_fwd: float = None,
    ) -> float:
        """Aggancia con precisione il piede acustico del transiente vocale, pesato per prossimità a ctc_t."""
        if self.num_frames == 0:
            return round(ctc_t, 3)
        fwd_r = float(max_shift if max_shift_fwd is None else max_shift_fwd)
        w_s = max(min_t, ctc_t - max_shift)
        w_e = min(max_t, ctc_t + fwd_r)
        if w_e <= w_s:
            return round(max(min_t, min(max_t, ctc_t)), 3)
        fs, fe = self._time_to_frame(w_s), self._time_to_frame(w_e)
        if fe <= fs:
            return round(max(min_t, min(max_t, ctc_t)), 3)
        sub_odf = self.odf[fs:fe + 1]
        if len(sub_odf) == 0:
            return round(max(min_t, min(max_t, ctc_t)), 3)

        norm_r = max(max_shift, fwd_r, 0.015)
        best_rel_i = -1
        best_weighted = -1.0
        for rel_i, ov in enumerate(sub_odf):
            ft = self._frame_to_time(fs + rel_i)
            dt = ft - ctc_t
            weight = max(0.35, 1.0 - 0.55 * (abs(dt) / norm_r))
            sc = float(ov) * weight
            if sc > best_weighted:
                best_weighted = sc
                best_rel_i = rel_i

        pk_v = float(sub_odf[best_rel_i]) if best_rel_i >= 0 else 0.0
        if pk_v < 0.09:
            return round(max(min_t, min(max_t, ctc_t)), 3)
        best_f = fs + best_rel_i
        dyn_tot = max(self.global_peak_total - self.global_floor_total, 0.20)
        post_e = float(np.mean(self.env_total[best_f:min(self.num_frames, best_f + 45)]))
        if post_e < float(self.local_floor_total[best_f]) + 0.08 * dyn_tot:
            return round(max(min_t, min(max_t, ctc_t)), 3)
        foot_f = best_f
        thresh = pk_v * 0.22
        for sb in range(1, min(22, best_f - fs + 1)):
            cf = best_f - sb
            if self.odf[cf] < thresh:
                foot_f = cf + 1
                break
        return max(round(min_t, 3), min(round(max_t, 3), self._frame_to_time(foot_f)))

    def trim_word_leading_silence(self, w_start: float, w_end: float, min_dur: float = 0.045) -> float:
        """Avanza l'attacco di parole lunghe che iniziano dentro una pausa silenziosa."""
        if self.num_frames == 0 or (w_end - w_start) < 0.38:
            return round(w_start, 3)
        skipped = self.detect_leading_silence_end(w_start, w_end)
        if skipped > w_start + 0.045 and skipped < w_end - min_dur:
            return self.snap_onset_micro(skipped, w_start, w_end - min_dur, max_shift=0.015, max_shift_fwd=0.008)
        return round(w_start, 3)

    def find_onset_after_internal_dip(self, w_start: float, w_end: float, min_tail_dur: float = 0.14, min_dip_ms: int = 18, min_span: float = 0.58) -> float:
        """Se una parola breve/stirata include un avanzo vocale precedente separato da un calo/pausa interna, aggancia l'attacco reale dopo la pausa."""
        if self.num_frames == 0 or (w_end - w_start) < min_span:
            return round(w_start, 3)
        f_s = self._time_to_frame(w_start + 0.08)
        f_e = self._time_to_frame(w_end - min_tail_dur)
        if f_e <= f_s + min_dip_ms:
            return round(w_start, 3)
        f_w0 = self._time_to_frame(w_start)
        f_w1 = self._time_to_frame(w_end)
        seg_e = self.env_total[f_w0:max(f_w0 + 20, f_w1)]
        seg_fl = self.local_floor_total[f_w0:max(f_w0 + 20, f_w1)]
        if len(seg_e) == 0:
            return round(w_start, 3)
        loc_pk = float(np.percentile(seg_e, 90))
        loc_fl = float(np.median(seg_fl))
        loc_dyn = max(loc_pk - loc_fl, 0.14)
        dyn_tot = max(self.global_peak_total - self.global_floor_total, 0.20)
        eff_dyn = min(dyn_tot, loc_dyn)
        dip_run = 0
        for cf in range(f_e, f_s - 1, -1):
            dip_th = float(self.local_floor_total[cf]) + 0.10 * eff_dyn
            if self.env_total[cf] <= dip_th:
                dip_run += 1
                if dip_run >= min_dip_ms:
                    last_silent = cf + dip_run
                    f_wake = min(f_e, last_silent)
                    while f_wake < f_e and self.env_total[f_wake] <= dip_th:
                        f_wake += 1
                    cand_t = self._frame_to_time(f_wake)
                    if cand_t > w_start + 0.08 and cand_t <= w_end - min_tail_dur:
                        return self.snap_onset_micro(cand_t, max(w_start, cand_t - 0.015), w_end - 0.05, max_shift=0.015, max_shift_fwd=0.008)
                    break
            else:
                dip_run = 0
        return round(w_start, 3)

    def expand_phrase_onset(
        self,
        ctc_t: float,
        min_t: float,
        max_lookback: float = 0.50,
        max_t: float = None,
        stop_at_valley: bool = False,
    ) -> float:
        """Recupera eventuali respiri vocali o sillabe morbide all'attacco di una frase ed elimina il silenzio iniziale."""
        if self.num_frames == 0:
            return round(ctc_t, 3)
        end_cap = self._frame_to_time(max(0, self.num_frames - 1)) if max_t is None else float(max_t)
        anchor_t = ctc_t
        if end_cap - anchor_t > 0.24:
            skipped = self.detect_leading_silence_end(anchor_t, min(end_cap, anchor_t + 8.0))
            if skipped > anchor_t + 0.045 and skipped < end_cap - 0.04:
                return self.snap_onset_micro(
                    skipped, max(min_t, anchor_t), end_cap - 0.035, max_shift=0.015, max_shift_fwd=0.008
                )

        fc = self._time_to_frame(anchor_t)
        fmin = self._time_to_frame(max(min_t, anchor_t - max_lookback))
        if fc <= fmin:
            return self.snap_onset_micro(
                anchor_t, min_t, max(anchor_t + 0.040, end_cap - 0.035), max_shift=0.015, max_shift_fwd=0.008
            )
        dyn_tot = max(self.global_peak_total - self.global_floor_total, 0.20)
        dyn_sus = max(self.global_peak_sustain - self.global_floor_sustain, 0.20)
        f_ref_e = min(self.num_frames, fc + 250)
        loc_pk_tot = float(np.percentile(self.env_total[fc:max(fc + 20, f_ref_e)], 85))
        loc_fl_tot = float(np.median(self.local_floor_total[fmin:fc + 1]))
        loc_dyn = max(loc_pk_tot - loc_fl_tot, 0.15 * dyn_tot)
        active_th = loc_fl_tot + 0.16 * loc_dyn
        e_atk = float(np.percentile(self.env_total[fc:min(self.num_frames, fc + 45)], 90)) if stop_at_valley else 0.0
        min_e = float(self.env_total[fc])
        min_f = fc

        last_act = fc
        silent_run = 0
        for cf in range(fc, fmin - 1, -1):
            cur_e = float(self.env_total[cf])
            if stop_at_valley:
                if cur_e < min_e:
                    min_e = cur_e
                    min_f = cf
                elif (fc - cf) >= 12 and min_e <= max(loc_fl_tot + 0.14 * loc_dyn, 0.48 * e_atk) and cur_e > max(min_e * 1.45, active_th):
                    last_act = max(last_act, min_f)
                    break
            is_act = (cur_e >= active_th) or (self.env_sustain[cf] >= self.local_floor_sustain[cf] + 0.14 * dyn_sus)
            if is_act:
                last_act = cf
                silent_run = 0
            else:
                silent_run += 1
                if silent_run >= 22:
                    break
        cand_t = self._frame_to_time(last_act)
        hi_t = min(end_cap - 0.035, anchor_t + 0.012) if end_cap > anchor_t + 0.020 else (anchor_t + 0.012)
        return self.snap_onset_micro(
            cand_t,
            max(min_t, self._frame_to_time(fmin)),
            max(cand_t + 0.008, hi_t),
            max_shift=0.015,
            max_shift_fwd=0.008,
        )

    def refine_phrase_offset(
        self,
        w_start: float,
        ctc_end: float,
        min_end: float,
        max_end: float,
        extend_sustain: bool = True,
    ) -> float:
        """Estende note tenute/vibrato fino alla fine reale ed elimina le code di silenzio."""
        if self.num_frames == 0:
            return round(max(min_end, min(max_end, ctc_end)), 3)
        fs = self._time_to_frame(w_start)
        f_min = self._time_to_frame(min_end)
        f_ctc = self._time_to_frame(max(min_end, min(max_end, ctc_end)))
        f_max = self._time_to_frame(max_end)
        if f_max <= f_min or f_ctc <= fs + 15:
            return round(max(min_end, min(max_end, ctc_end)), 3)

        dyn_sus = max(self.global_peak_sustain - self.global_floor_sustain, 0.20)
        dyn_tot = max(self.global_peak_total - self.global_floor_total, 0.20)
        core_tot = self.env_total[fs:max(fs + 30, min(f_ctc, fs + 900))]
        core_sus = self.env_sustain[fs:max(fs + 30, min(f_ctc, fs + 900))]
        if len(core_tot) == 0:
            return round(max(min_end, min(max_end, ctc_end)), 3)

        pk_tot = float(np.percentile(core_tot, 90))
        pk_sus = float(np.percentile(core_sus, 90))
        fl_tot = float(np.median(self.local_floor_total[fs:f_ctc + 1]))
        fl_sus = float(np.median(self.local_floor_sustain[fs:f_ctc + 1]))
        loc_dyn_tot = max(pk_tot - fl_tot, 0.35 * dyn_tot, 0.14)
        loc_dyn_sus = max(pk_sus - fl_sus, 0.35 * dyn_sus, 0.14)

        hold_th_sus = fl_sus + 0.18 * loc_dyn_sus
        hold_th_tot = fl_tot + 0.18 * loc_dyn_tot

        tail_s = max(fs, f_ctc - 35)
        tail_sus = float(np.mean(self.env_sustain[tail_s:f_ctc + 1]))
        tail_tot = float(np.mean(self.env_total[tail_s:f_ctc + 1]))
        if extend_sustain and (tail_sus >= hold_th_sus or tail_tot >= fl_tot + 0.22 * loc_dyn_tot) and f_max > f_ctc + 15:
            last_f = f_ctc
            dip = 0
            for cf in range(f_ctc, f_max):
                is_on = (self.env_sustain[cf] >= hold_th_sus) or (self.env_total[cf] >= fl_tot + 0.22 * loc_dyn_tot)
                if cf >= f_ctc and self.odf[cf] >= 0.45:
                    if dip >= 15:
                        break
                    pre_min = float(np.min(self.env_total[max(f_ctc, cf - 14):cf + 1]))
                    post_pk = float(np.max(self.env_total[cf:min(self.num_frames, cf + 18)]))
                    if post_pk >= 1.55 * max(0.20, pre_min):
                        last_f = min(last_f, max(f_ctc, cf - 12))
                        break
                if is_on:
                    last_f = cf
                    dip = 0
                else:
                    dip += 1
                    if dip >= 45:
                        break
            if last_f > f_ctc + 15:
                return min(round(max_end, 3), self._frame_to_time(last_f + 12))

        span_act = (self.env_total[fs:f_ctc] >= hold_th_tot) | (self.env_sustain[fs:f_ctc] >= hold_th_sus)
        # Trova l'ultima finestra attiva reale (>= 22ms) ignorando micro-picchi isolati nella coda di silenzio
        n_sp = len(span_act)
        if n_sp >= 30:
            win_sum = np.convolve(span_act.astype(np.int32), np.ones(22, dtype=np.int32), mode="same")
            sustained_idx = np.flatnonzero(win_sum >= 14)
            if len(sustained_idx) > 0:
                last_act = fs + int(sustained_idx[-1]) + 1
                if f_ctc - last_act >= 70:
                    return max(round(min_end, 3), self._frame_to_time(last_act + 15))
        return round(max(min_end, min(max_end, ctc_end)), 3)

    def has_active_energy(self, t0: float, t1: float, min_ratio: float = 0.55) -> bool:
        """Verifica se l'intervallo tra due parole contiene voce continua attiva."""
        if self.num_frames == 0:
            return True
        f0, f1 = self._time_to_frame(t0), self._time_to_frame(t1)
        if f1 <= f0:
            return True
        dyn_tot = max(self.global_peak_total - self.global_floor_total, 0.20)
        th = self.local_floor_total[f0:f1] + 0.16 * dyn_tot
        return float(np.mean(self.env_total[f0:f1] >= th)) >= min_ratio

    def find_strong_burst_in_gap(self, t0: float, t1: float, min_dur: float = 0.15):
        """Trova eventuali sillabe isolate cantate prima/dopo una pausa intraparola."""
        if self.num_frames == 0:
            return None
        f0, f1 = self._time_to_frame(t0), self._time_to_frame(t1)
        min_f = int(min_dur * 1000)
        if f1 - f0 < min_f:
            return None
        dyn_tot = max(self.global_peak_total - self.global_floor_total, 0.20)
        th = self.local_floor_total[f0:f1] + 0.20 * dyn_tot
        act = self.env_total[f0:f1] >= th
        i = 0
        n = len(act)
        while i < n:
            if act[i]:
                s = i
                while i < n and (act[i] or (i + 60 < n and np.any(act[i:i + 60]))):
                    i += 1
                if (i - s) >= min_f:
                    pk = float(np.max(self.env_total[f0 + s:f0 + i]))
                    if pk >= self.global_floor_total + 0.28 * dyn_tot:
                        return (
                            self.snap_onset_micro(self._frame_to_time(f0 + s), t0, t1, 0.030),
                            self._frame_to_time(f0 + i),
                        )
            else:
                i += 1
        return None

    def find_gap_adlib_span(self, t0: float, t1: float):
        """Individua ad-lib / eco in background collocati nella pausa tra due righe, escludendo code di decadimento."""
        if self.num_frames == 0:
            return None
        f0, f1 = self._time_to_frame(t0), self._time_to_frame(t1)
        if f1 - f0 < 120:
            return None
        dyn_tot = max(self.global_peak_total - self.global_floor_total, 0.20)
        dyn_sus = max(self.global_peak_sustain - self.global_floor_sustain, 0.20)
        th = self.local_floor_total[f0:f1] + 0.14 * dyn_tot
        th_s = self.local_floor_sustain[f0:f1] + 0.12 * dyn_sus
        act = (self.env_total[f0:f1] >= th) | (self.env_sustain[f0:f1] >= th_s)

        # Estrai isole contigue (gap interno max 140ms) invece di unire eventuale coda a t0 col vero ad-lib
        spans = []
        n_act = len(act)
        i = 0
        while i < n_act:
            if act[i]:
                s = i
                while i < n_act:
                    if act[i]:
                        i += 1
                    else:
                        nxt_rel = np.flatnonzero(act[i:min(n_act, i + 140)])
                        if len(nxt_rel) > 0:
                            i += int(nxt_rel[0])
                        else:
                            break
                spans.append((s, i))
            else:
                i += 1

        # Scarta il decadimento residuo attaccato al confine sinistro t0 (< 110ms) o troppo corto (< 90ms)
        valid_spans = [
            (s, e) for s, e in spans
            if (e - s) >= 90 and not (s <= 12 and e < 120 and len(spans) >= 2)
        ]
        if not valid_spans:
            return None

        best_s, best_e = max(valid_spans, key=lambda pair: float(np.sum(self.env_total[f0 + pair[0]:f0 + pair[1]])))
        pk = float(np.max(self.env_total[f0 + best_s:f0 + best_e]))
        if pk < self.global_floor_total + 0.15 * dyn_tot:
            return None
        return (
            self.snap_onset_micro(self._frame_to_time(f0 + best_s), t0, t1, max_shift=0.015, max_shift_fwd=0.008),
            min(round(t1, 3), self._frame_to_time(f0 + best_e + 15)),
        )

    def find_all_gap_bursts(self, t0: float, t1: float, min_dur_ms: int = 50, merge_gap_ms: int = 45) -> list:
        """Restituisce tutte le raffiche vocali/sibilanti distinte nella pausa [t0, t1], escludendo la coda iniziale a t0."""
        if self.num_frames == 0:
            return []
        f0, f1 = self._time_to_frame(t0), self._time_to_frame(t1)
        if f1 - f0 < max(70, min_dur_ms + 15):
            return []
        dyn_tot = max(self.global_peak_total - self.global_floor_total, 0.20)
        dyn_sus = max(self.global_peak_sustain - self.global_floor_sustain, 0.20)
        th_tot = self.local_floor_total[f0:f1] + max(0.030, 0.055 * dyn_tot)
        th_sus = self.local_floor_sustain[f0:f1] + max(0.025, 0.050 * dyn_sus)
        act = (self.env_total[f0:f1] >= th_tot) | (self.env_sustain[f0:f1] >= th_sus)

        spans = []
        n_act = len(act)
        i = 0
        while i < n_act:
            if act[i]:
                s = i
                while i < n_act:
                    if act[i]:
                        i += 1
                    else:
                        nxt_rel = np.flatnonzero(act[i:min(n_act, i + merge_gap_ms)])
                        if len(nxt_rel) > 0:
                            i += int(nxt_rel[0])
                        else:
                            break
                if (i - s) >= min_dur_ms:
                    spans.append((s, i))
            else:
                i += 1

        if not spans:
            return []

        # Se la prima raffica è incollata a t0 (coda del core precedente) e contiene un calo interno prima del nuovo attacco, taglia la coda
        s0, e0 = spans[0]
        if s0 <= 20 and (e0 - s0) >= 180:
            sub0 = self.env_total[f0 + s0:f0 + e0]
            e_start0 = float(sub0[0])
            # Cerca il minimo locale nei primi 420ms seguito da una nuova salita
            scan_lim = min(len(sub0) - 40, 420)
            if scan_lim > 40 and e_start0 >= 0.25:
                v_rel = int(np.argmin(sub0[:scan_lim]))
                v_val = float(sub0[v_rel])
                post_pk = float(np.max(sub0[v_rel:min(len(sub0), v_rel + 250)]))
                if v_rel >= 15 and v_val <= 0.35 * e_start0 and post_pk >= max(v_val * 2.0, 0.20):
                    spans[0] = (s0 + v_rel + 1, e0)

        valid = []
        for idx_sp, (s, e) in enumerate(spans):
            if (e - s) < min_dur_ms:
                continue
            # Scarta la coda/respiro incollata al confine sinistro t0 quando segue almeno una pausa e/o un vero ad-lib più avanti
            if s <= 20 and e < 420 and (len(spans) >= 2 or float(self.env_total[f0]) >= float(np.max(self.env_total[f0 + s:f0 + e])) * 0.72):
                continue
            pk = float(np.max(self.env_total[f0 + s:f0 + e]))
            loc_fl = float(np.median(self.local_floor_total[f0 + s:f0 + e]))
            if pk < max(self.global_floor_total + 0.055 * dyn_tot, loc_fl * 4.0 + 0.035):
                continue
            bs = self.snap_onset_micro(self._frame_to_time(f0 + s), t0, t1, max_shift=0.015, max_shift_fwd=0.008)
            be = min(round(t1, 3), self._frame_to_time(f0 + e + 15))
            if be >= bs + 0.045:
                valid.append((bs, be))
        return valid


def refine_vocal_onset_transient(
    audio_16k: np.ndarray,
    target_time: float,
    min_time: float = 0.0,
    max_time: float = 9999.0,
) -> float:
    engine = AcousticPrecisionEngine(audio_16k)
    return engine.refine_onset(target_time, min_time, max_time)


def calculate_auto_compensation_offset(
    engine_or_audio,
    aligned_tokens: list,
    min_tokens: int = 8,
    search_window: float = 0.160,
    deadzone_ms: int = 8,
) -> float:
    """Misura lo sfasamento sistematico bidirezionale (anticipo +ms / ritardo -ms) tra token AI e transienti vocali 1ms."""
    if isinstance(engine_or_audio, AcousticPrecisionEngine):
        engine = engine_or_audio
    else:
        if engine_or_audio is None or len(engine_or_audio) == 0:
            return 0.0
        engine = AcousticPrecisionEngine(engine_or_audio)

    if not aligned_tokens or engine.num_frames == 0:
        return 0.0

    candidate_tokens = [
        tok for tok in aligned_tokens
        if 0.10 <= (float(tok["end"]) - float(tok["start"])) <= 1.60
        and len(str(tok.get("clean") or clean_token_str(tok.get("word", "")))) >= 2
    ]
    if not candidate_tokens:
        return 0.0

    if len(candidate_tokens) > 120:
        step = len(candidate_tokens) / 120.0
        sample_tokens = [candidate_tokens[int(i * step)] for i in range(120)]
    else:
        sample_tokens = candidate_tokens

    dyn_tot = max(engine.global_peak_total - engine.global_floor_total, 0.20)
    post_pause_deltas = []
    continuous_deltas = []
    for tok in sample_tokens:
        t_raw = float(tok["start"])
        t_end = float(tok["end"])
        w_s = max(0.0, t_raw - search_window)
        w_e = min(t_end - 0.015, t_raw + search_window)
        if w_e <= w_s + 0.020:
            continue
        fs, fe = engine._time_to_frame(w_s), engine._time_to_frame(w_e)
        fc = engine._time_to_frame(t_raw)
        if fe <= fs + 5:
            continue
        sub_odf = engine.odf[fs:fe + 1]
        if len(sub_odf) == 0:
            continue

        # Valuta se c'è una pausa vocale prima del token (attacco pulito vs sillaba legata)
        pre_s = max(0, fc - 60)
        pre_e_slice = engine.env_total[pre_s:max(pre_s + 1, fc - 8)]
        post_e_slice = engine.env_total[fc:min(engine.num_frames, fc + 60)]
        pre_mean = float(np.mean(pre_e_slice)) if len(pre_e_slice) > 0 else 0.0
        post_mean = float(np.mean(post_e_slice)) if len(post_e_slice) > 0 else 0.0
        has_pre_pause = (pre_mean < float(engine.local_floor_total[fc]) + 0.12 * dyn_tot) and (post_mean > pre_mean + 0.12 * dyn_tot)

        if has_pre_pause:
            pk_i = int(np.argmax(sub_odf))
        else:
            # Nel parlato continuo pesa per prossimità evitando salti su sillabe adiacenti
            best_w = -1.0
            pk_i = 0
            for ri, ov in enumerate(sub_odf):
                dt_r = engine._frame_to_time(fs + ri) - t_raw
                w_sc = float(ov) * max(0.30, 1.0 - 0.60 * (abs(dt_r) / max(search_window, 0.020)))
                if w_sc > best_w:
                    best_w = w_sc
                    pk_i = ri

        pk_v = float(sub_odf[pk_i])
        med_v = float(np.median(sub_odf))
        if pk_v < 0.08 or pk_v < med_v * 2.1:
            continue
        best_f = fs + pk_i
        post_e = float(np.mean(engine.env_total[best_f:min(engine.num_frames, best_f + 55)]))
        if post_e < float(engine.local_floor_total[best_f]) + 0.10 * dyn_tot:
            continue

        foot_f = best_f
        thresh = pk_v * 0.22
        for sb in range(1, min(20, best_f - fs + 1)):
            cf = best_f - sb
            if engine.odf[cf] < thresh:
                foot_f = cf + 1
                break

        acoustic_t = engine._frame_to_time(foot_f)
        delta = acoustic_t - t_raw
        if -search_window <= delta <= search_window:
            if has_pre_pause:
                post_pause_deltas.append(delta)
            else:
                continuous_deltas.append(delta)

    deltas = post_pause_deltas if len(post_pause_deltas) >= max(4, min_tokens // 2) else (post_pause_deltas + continuous_deltas)

    if len(deltas) >= max(3, min_tokens // 2):
        if len(deltas) >= min_tokens:
            q25, q75 = np.percentile(deltas, [25, 75])
            iqr = q75 - q25
            clean_deltas = [d for d in deltas if (q25 - 1.5 * iqr) <= d <= (q75 + 1.5 * iqr)]
            med = float(np.median(clean_deltas if clean_deltas else deltas))
        else:
            med = float(np.median(deltas))
        if abs(med) < (deadzone_ms / 1000.0):
            return 0.0
        return round(max(-0.180, min(0.180, med)), 3)

    return 0.0


# ==============================================================================
# HYBRID CTC + WHISPER MACRO-ANCHOR ALIGNER (MMS_FA + MULTILINGUAL NUMBERS)
# ==============================================================================
NONLEXICAL_INTERJECTIONS = {
    "eh", "oh", "ah", "uh", "mm", "hm", "hmm", "yeah", "yea", "ye", "yah", "ya",
    "ooh", "ohh", "ahh", "ehh", "uhh", "eheh", "eheheh", "ahah", "ohoh",
    "la", "na", "da", "whoa", "woah", "hey", "ay", "ey", "o", "e", "a", "u",
}


def _int_to_it(n: int) -> str:
    if n == 0:
        return "zero"
    units = ["", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove",
             "dieci", "undici", "dodici", "tredici", "quattordici", "quindici", "sedici",
             "diciassette", "diciotto", "diciannove"]
    tens = ["", "", "venti", "trenta", "quaranta", "cinquanta", "sessanta", "settanta", "ottanta", "novanta"]
    if n < 20:
        return units[n]
    if n < 100:
        t, u = divmod(n, 10)
        res = tens[t][:-1] if u in (1, 8) else tens[t]
        return res + units[u]
    if n < 1000:
        h, r = divmod(n, 100)
        res = "cento" if h == 1 else units[h] + "cento"
        return res + (_int_to_it(r) if r else "")
    if n < 10000:
        k, r = divmod(n, 1000)
        res = "mille" if k == 1 else _int_to_it(k) + "mila"
        return res + (_int_to_it(r) if r else "")
    return str(n)


def _int_to_en(n: int) -> str:
    if n == 0:
        return "zero"
    units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
             "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
             "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    if n < 20:
        return units[n]
    if n < 100:
        t, u = divmod(n, 10)
        return tens[t] + units[u]
    if n < 1000:
        h, r = divmod(n, 100)
        return units[h] + "hundred" + (_int_to_en(r) if r else "")
    if n < 10000:
        k, r = divmod(n, 1000)
        return units[k] + "thousand" + (_int_to_en(r) if r else "")
    return str(n)


def _int_to_es(n: int) -> str:
    if n == 0:
        return "cero"
    units = ["", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
             "diez", "once", "doce", "trece", "catorce", "quince", "dieciseis",
             "diecisiete", "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidos",
             "veintitres", "veinticuatro", "veinticinco", "veintiseis", "veintisiete",
             "veintiocho", "veintinueve"]
    tens = ["", "", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa"]
    hundreds = ["", "ciento", "doscientos", "trescientos", "cuatrocientos", "quinientos",
                "seiscientos", "setecientos", "ochocientos", "novecientos"]
    if n < 30:
        return units[n]
    if n < 100:
        t, u = divmod(n, 10)
        return tens[t] + ("y" + units[u] if u else "")
    if n == 100:
        return "cien"
    if n < 1000:
        h, r = divmod(n, 100)
        return hundreds[h] + (_int_to_es(r) if r else "")
    if n < 10000:
        k, r = divmod(n, 1000)
        res = "mil" if k == 1 else _int_to_es(k) + "mil"
        return res + (_int_to_es(r) if r else "")
    return str(n)


def _int_to_fr(n: int) -> str:
    if n == 0:
        return "zero"
    units = ["", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
             "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
             "dixsept", "dixhuit", "dixneuf"]
    tens = ["", "dix", "vingt", "trente", "quarante", "cinquante", "soixante", "soixante", "quatrevingt", "quatrevingt"]
    if n < 20:
        return units[n]
    if n < 70:
        t, u = divmod(n, 10)
        return tens[t] + units[u]
    if n < 80:
        return "soixante" + units[n - 60]
    if n < 100:
        return "quatrevingt" + units[n - 80]
    if n < 1000:
        h, r = divmod(n, 100)
        res = "cent" if h == 1 else units[h] + "cent"
        return res + (_int_to_fr(r) if r else "")
    if n < 10000:
        k, r = divmod(n, 1000)
        res = "mille" if k == 1 else _int_to_fr(k) + "mille"
        return res + (_int_to_fr(r) if r else "")
    return str(n)


def _int_to_de(n: int) -> str:
    if n == 0:
        return "null"
    units = ["", "eins", "zwei", "drei", "vier", "funf", "sechs", "sieben", "acht", "neun",
             "zehn", "elf", "zwolf", "dreizehn", "vierzehn", "funfzehn", "sechzehn",
             "siebzehn", "achtzehn", "neunzehn"]
    tens = ["", "", "zwanzig", "dreissig", "vierzig", "funfzig", "sechzig", "siebzig", "achtzig", "neunzig"]
    if n < 20:
        return units[n]
    if n < 100:
        t, u = divmod(n, 10)
        if u == 0:
            return tens[t]
        u_word = "ein" if u == 1 else units[u]
        return u_word + "und" + tens[t]
    if n < 1000:
        h, r = divmod(n, 100)
        u_h = "ein" if h == 1 else units[h]
        return u_h + "hundert" + (_int_to_de(r) if r else "")
    return str(n)


def _int_to_pt(n: int) -> str:
    if n == 0:
        return "zero"
    units = ["", "um", "dois", "tres", "quatro", "cinco", "seis", "sete", "oito", "nove",
             "dez", "onze", "doze", "treze", "catorze", "quinze", "dezesseis",
             "dezessete", "dezoito", "dezenove"]
    tens = ["", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta", "oitenta", "noventa"]
    hundreds = ["", "cento", "duzentos", "trezentos", "quatrocentos", "quinhentos",
                "seiscentos", "setecentos", "oitocentos", "novecentos"]
    if n < 20:
        return units[n]
    if n < 100:
        t, u = divmod(n, 10)
        return tens[t] + ("e" + units[u] if u else "")
    if n == 100:
        return "cem"
    if n < 1000:
        h, r = divmod(n, 100)
        return hundreds[h] + ("e" + _int_to_pt(r) if r else "")
    return str(n)


def _int_to_lang(n: int, lang: str = "it") -> str:
    lc = (lang or "it")[:2].lower()
    if lc == "en":
        return _int_to_en(n)
    if lc == "es":
        return _int_to_es(n)
    if lc == "fr":
        return _int_to_fr(n)
    if lc == "de":
        return _int_to_de(n)
    if lc == "pt":
        return _int_to_pt(n)
    return _int_to_it(n)


def _digits_to_words(s: str, lang: str = "it") -> str:
    dmaps = {
        "it": ["zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove"],
        "en": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
        "es": ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"],
        "fr": ["zero", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"],
        "de": ["null", "eins", "zwei", "drei", "vier", "funf", "sechs", "sieben", "acht", "neun"],
        "pt": ["zero", "um", "dois", "tres", "quatro", "cinco", "seis", "sete", "oito", "nove"],
    }
    dmap = dmaps.get((lang or "it")[:2].lower(), dmaps["en"])
    return "".join(dmap[int(ch)] for ch in s if ch.isdigit())


_HOMOGLYPH_TRANS = str.maketrans({
    "а": "a", "е": "e", "ё": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "А": "A", "Е": "E", "Ё": "E", "О": "O", "Р": "P", "С": "C", "У": "Y", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
    "\u2000": " ", "\u2001": " ", "\u2002": " ", "\u2003": " ", "\u2004": " ", "\u2005": " ",
    "\u2006": " ", "\u2007": " ", "\u2008": " ", "\u2009": " ", "\u200a": " ", "\u200b": " ",
    "\u00a0": " ", "\ufeff": " ",
})


def _strip_accents_lower(s: str) -> str:
    s = str(s).translate(_HOMOGLYPH_TRANS)
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


def _clean_phonetic_ctc(s: str, lang: str = "it", collapse_repeats: bool = True) -> str:
    s = re.sub(r"\d+", lambda m: _int_to_lang(int(m.group(0)), lang) if int(m.group(0)) < 10000 else m.group(0), s)
    s = _strip_accents_lower(s)
    s = re.sub(r"[^a-z]", "", s)
    if collapse_repeats:
        s = re.sub(r"(.)\1+", r"\1", s)
    return s


def is_latin_script_compatible(text: str) -> bool:
    """Verifica che il testo sia prevalentemente in caratteri latini per MMS_FA CTC."""
    alpha_chars = [ch for ch in text if ch.isalpha()]
    if not alpha_chars:
        return True
    latin_cnt = 0
    for ch in alpha_chars:
        base = _strip_accents_lower(ch)
        if any("a" <= c <= "z" for c in base):
            latin_cnt += 1
    return (latin_cnt / float(len(alpha_chars))) >= 0.50


def is_nonlexical_interjection(words: list, lang: str = "it") -> bool:
    ws = [_clean_phonetic_ctc(w, lang=lang, collapse_repeats=True) for w in words if _clean_phonetic_ctc(w, lang=lang)]
    if not ws:
        return True
    if len(ws) <= 4 and all(w in NONLEXICAL_INTERJECTIONS or re.fullmatch(r"(eh|oh|ah|uh|la|na)+", w) for w in ws):
        return True
    return False


def _is_whisper_hallucination(text: str) -> bool:
    """Riconosce note allucinazioni di Whisper su silenzio o code strumentali."""
    clean = re.sub(r"[^\w\s]", "", _strip_accents_lower(text)).strip()
    if not clean:
        return True
    hallucinations = {
        "grazie a tutti", "grazie", "grazie per la visione", "sottotitoli a cura di",
        "sottotitoli creati dalla comunita amaraorg", "amaraorg", "musica", "applausi",
        "iscriviti al canale", "thank you for watching", "thanks for watching",
        "subtitles by", "music", "bye", "ciao",
    }
    if clean in hallucinations:
        return True
    if any(p in clean for p in ("sottotitoli", "amaraorg", "grazie a tutti", "thank you for watching")):
        return len(clean.split()) <= 6
    raw_ws = [re.sub(r"[^\w]", "", w) for w in clean.split() if re.sub(r"[^\w]", "", w)]
    if len(raw_ws) >= 10 and len(set(raw_ws)) == 1:
        return True
    if len(raw_ws) >= 12 and len(set(raw_ws)) <= 2:
        return True
    if len(raw_ws) == 1 and len(raw_ws[0]) >= 8 and len(set(raw_ws[0])) <= 2:
        return True
    if re.search(r"(.)\1{7,}", clean):
        return True
    return False


def _separate_demucs_stems(audio_path: str, device: str = "cuda", demucs_model: str = "htdemucs"):
    """Estrae in un solo passaggio GPU sia lo stem vocals sia lo stem filtered_vocal (vocals + other banda 85-3400Hz)."""
    import demucs.pretrained
    import demucs.apply
    from demucs.audio import AudioFile
    import torchaudio.functional as AF

    model_name = demucs_model if demucs_model in ("htdemucs", "htdemucs_ft") else "htdemucs"
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    d_model = demucs.pretrained.get_model(model_name).to(dev).eval()
    try:
        wav_st = AudioFile(audio_path).read(
            streams=0, samplerate=d_model.samplerate, channels=d_model.audio_channels
        )
        ref = wav_st.mean(0)
        ref_std = float(ref.std())
        wav_st_n = (wav_st - ref.mean()) / (ref_std if ref_std > 1e-8 else 1.0)
        with torch.inference_mode():
            sources = demucs.apply.apply_model(d_model, wav_st_n[None], device=dev, progress=False)[0]
        sources = sources * (ref_std if ref_std > 1e-8 else 1.0) + ref.mean()

        src_names = list(d_model.sources)
        idx_v = src_names.index("vocals")
        idx_o = src_names.index("other") if "other" in src_names else None

        voc_mono = sources[idx_v].mean(0, keepdim=True).cpu()
        voc_16k = AF.resample(voc_mono, d_model.samplerate, 16000).squeeze(0).numpy().astype(np.float32)

        max_v = float(np.max(np.abs(voc_16k)))
        scale_v = max_v if max_v > 1e-4 else 1.0
        voc_norm = (voc_16k / scale_v).astype(np.float32)

        filt_norm = None
        if idx_o is not None:
            oth_mono = sources[idx_o].mean(0, keepdim=True).cpu()
            oth_16k = AF.resample(oth_mono, d_model.samplerate, 16000).squeeze(0).numpy().astype(np.float32)
            oth_rel = (oth_16k / scale_v).astype(np.float32)
            if HAS_SCIPY and len(oth_rel) > 1600:
                try:
                    sos_vp = butter(4, [85, 3400], btype="bandpass", fs=16000, output="sos")
                    oth_vp = sosfiltfilt(sos_vp, oth_rel).astype(np.float32)
                except Exception:
                    oth_vp = oth_rel
            else:
                oth_vp = oth_rel
            comb = voc_norm + oth_vp
            max_c = float(np.max(np.abs(comb)))
            if max_c > 1e-4:
                comb = (comb / max_c).astype(np.float32)
            filt_norm = comb

        return voc_norm, filt_norm
    finally:
        del d_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HybridCTCAligner:
    """Motore di allineamento ibrido: Whisper Macro-Anchors + MMS_FA CTC (20ms) + Acoustic 1ms."""

    @staticmethod
    def can_run_ctc(text: str) -> bool:
        if not HAS_MMS_FA:
            return False
        return is_latin_script_compatible(text)

    @staticmethod
    def compute_ctc_emissions(vocal_audio_16k: np.ndarray, device: str = "cuda", log_fn=None, keep_model: bool = False):
        """Calcola la matrice emissioni CTC log-prob con normalizzazione per-finestra (layer_norm locale)."""
        bundle = torchaudio.pipelines.MMS_FA
        dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        mms_model = bundle.get_model().to(dev).eval()
        tokenizer = bundle.get_tokenizer()
        valid_chars = set(bundle.get_labels()) - {"-", "*"}

        v_wav = np.asarray(vocal_audio_16k, dtype=np.float32)
        chunk_samples = int(8.0 * 16000)
        pad_samples = int(1.5 * 16000)
        em_chunks = []

        with torch.inference_mode():
            for s_i in range(0, len(v_wav), chunk_samples):
                e_i = min(len(v_wav), s_i + chunk_samples)
                ws = max(0, s_i - pad_samples)
                we = min(len(v_wav), e_i + pad_samples)
                seg_t = torch.from_numpy(v_wav[ws:we]).unsqueeze(0).to(dev)
                em, _ = mms_model(seg_t)
                lt = (s_i - ws) // 320
                kf = int(np.ceil((e_i - s_i) / 320))
                em_chunks.append(em[:, lt:lt + kf, :].cpu())

        emission = torch.cat(em_chunks, dim=1)[0].to(dev)
        aligner = bundle.get_aligner()
        if not keep_model:
            del mms_model
            mms_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (emission, tokenizer, aligner, valid_chars, mms_model, dev) if keep_model else (emission, tokenizer, aligner, valid_chars)

    @staticmethod
    def align(
        vocal_audio_16k: np.ndarray,
        clean_lines_data: list,
        whisper_segments: list,
        acoustic_engine: AcousticPrecisionEngine,
        detected_lang: str,
        device: str = "cuda",
        total_audio_sec: float = 0.0,
        progress_cb=None,
        filtered_vocal_16k: np.ndarray = None,
    ) -> list:
        """Esegue Two-Pass CTC Alignment (Lexical Anchors -> Island Gaps -> BG Ad-Libs) + 1ms Refinement."""
        emission, tokenizer, aligner, valid_chars, mms_model, ctc_dev = HybridCTCAligner.compute_ctc_emissions(
            vocal_audio_16k, device=device, keep_model=True
        )
        v_wav_full = np.asarray(vocal_audio_16k, dtype=np.float32)
        f_wav_full = np.asarray(filtered_vocal_16k, dtype=np.float32) if filtered_vocal_16k is not None else None
        try:
            return HybridCTCAligner._align_with_resources(
                v_wav_full=v_wav_full,
                f_wav_full=f_wav_full,
                emission=emission,
                tokenizer=tokenizer,
                aligner=aligner,
                valid_chars=valid_chars,
                mms_model=mms_model,
                ctc_dev=ctc_dev,
                clean_lines_data=clean_lines_data,
                whisper_segments=whisper_segments,
                acoustic_engine=acoustic_engine,
                detected_lang=detected_lang,
                total_audio_sec=total_audio_sec,
                progress_cb=progress_cb,
            )
        finally:
            if mms_model is not None:
                del mms_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _align_with_resources(
        v_wav_full: np.ndarray,
        f_wav_full,
        emission,
        tokenizer,
        aligner,
        valid_chars: set,
        mms_model,
        ctc_dev,
        clean_lines_data: list,
        whisper_segments: list,
        acoustic_engine: AcousticPrecisionEngine,
        detected_lang: str,
        total_audio_sec: float = 0.0,
        progress_cb=None,
    ) -> list:
        if progress_cb:
            progress_cb(78)

        fd = 0.020
        lang = (detected_lang or "it")[:2].lower()
        islands = acoustic_engine.detect_vocal_islands()

        def word_variants(w: str):
            parts = [p for p in re.split(r"[-/]", w) if p.strip()] or [w]
            part_opts = []
            for p in parts:
                wc = re.sub(r"[^\w']", "", p)
                cands = []
                if wc.isdigit():
                    val = int(wc)
                    if val < 10000:
                        s_card = _int_to_lang(val, lang)
                        cands.append(s_card)
                        if len(wc) >= 2:
                            s_dig = _digits_to_words(wc, lang)
                            if s_dig and s_dig != s_card:
                                cands.append(s_dig)
                        if lang == "it" and wc == "1":
                            cands.append("un")
                else:
                    low = _strip_accents_lower(p).strip("'")
                    cln = "".join(c for c in low if c in valid_chars)
                    if cln:
                        cands.append(cln)
                    if wc.lower() == "tv":
                        cands.extend(["tivu", "tivi"])
                    elif wc.lower() == "ok":
                        cands.extend(["okay", "okei"])
                part_opts.append([c for c in cands if c])
            # Verifica se tutte le parti sono vuote (es. solo trattino o punteggiatura isolata)
            if all(len(opt) == 0 for opt in part_opts):
                return None
            clean_opts = [opt if opt else [] for opt in part_opts]
            non_empty = [opt for opt in clean_opts if opt]
            out = []
            for combo in product(*non_empty):
                flat = list(combo)
                if flat not in out:
                    out.append(flat)
                if len(out) >= 3:
                    break
            return out if out else None

        lines_info = []
        for idx, seg_info in enumerate(clean_lines_data):
            text = seg_info["text"]
            words = text.split()
            adlib_mask = [False] * len(words)
            if not seg_info["is_bg"] and "(" in text and words:
                p_depth = 0
                groups = []
                cur_grp = None
                tmp_mask = [False] * len(words)
                for wi_tok, w_tok in enumerate(words):
                    opens = w_tok.count("(")
                    closes = w_tok.count(")")
                    is_ad = (p_depth > 0) or w_tok.lstrip().startswith("(")
                    if is_ad:
                        if cur_grp is None:
                            cur_grp = [wi_tok, wi_tok]
                        else:
                            cur_grp[1] = wi_tok
                    else:
                        if cur_grp is not None:
                            groups.append(tuple(cur_grp))
                            cur_grp = None
                    tmp_mask[wi_tok] = is_ad
                    p_depth = max(0, p_depth + opens - closes)
                if cur_grp is not None:
                    groups.append(tuple(cur_grp))

                first_core_idx = next((idx_w for idx_w, m_v in enumerate(tmp_mask) if not m_v), -1)
                if first_core_idx != -1:
                    # I gruppi tra parentesi iniziali (prima del testo core) o lunghi (>=5 parole) vanno nel core di Pass 1A
                    for g_s, g_e in groups:
                        g_len = g_e - g_s + 1
                        if g_s > first_core_idx and g_len <= 4:
                            for idx_w in range(g_s, g_e + 1):
                                adlib_mask[idx_w] = True

            core_words = [w for w, is_ad in zip(words, adlib_mask) if not is_ad] or words
            core_text = " ".join(core_words).strip()
            phon_full = _clean_phonetic_ctc(text, lang=lang, collapse_repeats=True)
            phon_core = _clean_phonetic_ctc(core_text, lang=lang, collapse_repeats=True) if core_text else phon_full
            lines_info.append({
                "idx": idx,
                "agent": seg_info["agent"],
                "is_bg": seg_info["is_bg"],
                "text": text,
                "words": words,
                "adlib_mask": adlib_mask,
                "has_inline_adlibs": any(adlib_mask),
                "phon": phon_full,
                "phon_core": phon_core or phon_full,
                "has_trailing_parens": bool(re.search(r"\([^)]+\)\s*$", text)),
                "nonlex": is_nonlexical_interjection(core_words, lang=lang),
            })

        # Costruzione stream parole macro-ancora Whisper ritagliato sulle isole vocali reali
        w_words = []
        for seg in whisper_segments:
            st = float(seg.get("start", 0.0))
            en = float(seg.get("end", 0.0))
            txt = str(seg.get("text", "")).strip()
            if _is_whisper_hallucination(txt):
                continue
            raw_toks = [t for t in txt.split() if _clean_phonetic_ctc(t, lang)]
            if not raw_toks:
                continue
            uniq = set(_clean_phonetic_ctc(t, lang) for t in raw_toks)
            if len(raw_toks) >= 12 and len(uniq) <= 2:
                continue

            ov_isls = [
                (max(st, is_s - 0.08), min(en, is_e + 0.08))
                for (is_s, is_e) in islands
                if is_e > st + 0.05 and is_s < en - 0.05
            ]
            if ov_isls:
                st = max(st, ov_isls[0][0])
                en = min(en, ov_isls[-1][1])
            else:
                ov_isls = [(st, en)]
            if en <= st + 0.05:
                continue

            seg_words_raw = seg.get("words") or []
            parsed_sw = []
            for sw in seg_words_raw:
                wt = str(sw.get("word", "")).strip()
                ph = _clean_phonetic_ctc(wt, lang)
                if ph:
                    ws_w = max(st, float(sw.get("start", st)))
                    we_w = min(en, max(ws_w + 0.03, float(sw.get("end", ws_w + 0.03))))
                    parsed_sw.append({"phon": ph, "start": ws_w, "end": we_w})

            if len(parsed_sw) >= max(1, len(raw_toks) - 1):
                # Tronca la prima parola se Whisper DTW l'ha stirata su un lungo silenzio introduttivo (>1.2s)
                if len(parsed_sw) >= 2 and (parsed_sw[0]["end"] - parsed_sw[0]["start"]) > 1.20:
                    tail_dur = parsed_sw[-1]["end"] - parsed_sw[1]["start"]
                    if tail_dur / max(1, len(parsed_sw) - 1) <= 0.55:
                        parsed_sw[0]["start"] = max(parsed_sw[0]["start"], parsed_sw[1]["start"] - 0.60)
                st_eff = parsed_sw[0]["start"]
                en_eff = parsed_sw[-1]["end"]
                for pw in parsed_sw:
                    w_words.append({
                        "phon": pw["phon"],
                        "start": pw["start"],
                        "end": max(pw["start"] + 0.03, pw["end"]),
                        "seg_s": st_eff,
                        "seg_e": en_eff,
                    })
                continue

            act_spans = [(a, b) for a, b in ov_isls if b > a + 0.02] or [(st, en)]
            # Se uno span multi-isola ha un lungo silenzio iniziale rispetto al numero di parole, evita di spalmare sul preambolo
            if len(act_spans) >= 2:
                last_span = act_spans[-1]
                tot_act = sum(b - a for a, b in act_spans)
                if (tot_act / max(1, len(raw_toks))) >= 0.75 and (last_span[1] - last_span[0]) >= len(raw_toks) * 0.22:
                    if (last_span[0] - act_spans[-2][1]) >= 0.55:
                        act_spans = [last_span]
                        st = last_span[0]
            act_tot = sum(b - a for a, b in act_spans)

            def _map_frac(frac: float) -> float:
                tgt = max(0.0, min(act_tot, frac * act_tot))
                acc = 0.0
                for a, b in act_spans:
                    d = b - a
                    if acc + d >= tgt or (a, b) == act_spans[-1]:
                        return a + max(0.0, min(d, tgt - acc))
                    acc += d
                return act_spans[-1][1]

            tot_c = sum(max(1, len(_clean_phonetic_ctc(t, lang, collapse_repeats=False))) for t in raw_toks)
            c_acc = 0
            for t in raw_toks:
                cl = max(1, len(_clean_phonetic_ctc(t, lang, collapse_repeats=False)))
                ws_t = _map_frac(c_acc / tot_c)
                c_acc += cl
                we_t = _map_frac(c_acc / tot_c)
                w_words.append({
                    "phon": _clean_phonetic_ctc(t, lang),
                    "start": ws_t,
                    "end": max(ws_t + 0.03, we_t),
                    "seg_s": st,
                    "seg_e": en,
                })

        def _is_echo_bg(idx_l: int) -> bool:
            li = lines_info[idx_l]
            if not li["is_bg"]:
                return False
            if len(li["words"]) <= 3:
                return True
            for prev_i in range(idx_l - 1, -1, -1):
                if not lines_info[prev_i]["is_bg"]:
                    sim = SequenceMatcher(None, li["phon"], lines_info[prev_i]["phon"]).ratio()
                    return sim >= 0.50
            return False

        seq_indices = [i for i in range(len(lines_info)) if not _is_echo_bg(i)]
        has_any_lexical_seq = any(not lines_info[i]["nonlex"] for i in seq_indices)

        # Anchor DP solo su righe lessicali (evita che interiezioni "Oh/Eh" rubino ancore alle strofe)
        S = len(seq_indices)
        M = len(w_words)
        NEG_INF = -1e9
        dp = np.full((S + 1, M + 1), NEG_INF, dtype=np.float64)
        back = [[None] * (M + 1) for _ in range(S + 1)]
        dp[0, 0] = 0.0

        for si in range(S + 1):
            for j in range(M + 1):
                v = dp[si, j]
                if v <= NEG_INF / 2:
                    continue
                if j < M:
                    if v - 0.12 > dp[si, j + 1]:
                        dp[si, j + 1] = v - 0.12
                        back[si][j + 1] = (si, j, "skip_w")
                if si < S:
                    li = lines_info[seq_indices[si]]
                    skip_pen = 0.25 if (li["nonlex"] and has_any_lexical_seq) else (0.70 if li["is_bg"] else 1.25)
                    if v - skip_pen > dp[si + 1, j]:
                        dp[si + 1, j] = v - skip_pen
                        back[si + 1][j] = (si, j, "skip_l")
                if si < S and j < M:
                    li = lines_info[seq_indices[si]]
                    if li["nonlex"] and has_any_lexical_seq:
                        continue
                    l_ph = li["phon"]
                    l_core = li["phon_core"]
                    nw = max(1, len(li["words"]))
                    acc = ""
                    max_k = min(M - j, max(nw + 6, int(nw * 2.2)))
                    for k in range(1, max_k + 1):
                        if k > 1 and (w_words[j + k - 1]["start"] - w_words[j + k - 2]["end"]) > 3.2:
                            break
                        acc += w_words[j + k - 1]["phon"]
                        if len(acc) > len(l_ph) * 2.2 + 8:
                            break
                        sim_full = SequenceMatcher(None, l_ph, acc).ratio()
                        sim_core = SequenceMatcher(None, l_core, acc).ratio() if l_core != l_ph else sim_full
                        sim = max(sim_full, sim_core)
                        # Impedisce che un frammento di 1 parola (es. "moonlight") ancori una riga lunga (6+ parole)
                        if len(l_ph) >= 12 and len(acc) < 0.48 * len(l_ph) and sim < 0.68:
                            continue
                        if li["is_bg"] and len(acc) < 0.55 * len(l_ph) and sim < 0.62:
                            continue
                        # Impedisce di inglobare un ad-lib isolato prima di una pausa (>= 0.45s) se il suffisso senza esso matcha uguale o meglio
                        if k >= 2 and (w_words[j + 1]["start"] - w_words[j]["end"]) >= 0.45:
                            suf_acc = acc[len(w_words[j]["phon"]):]
                            suf_sim = max(
                                SequenceMatcher(None, l_ph, suf_acc).ratio(),
                                SequenceMatcher(None, l_core, suf_acc).ratio() if l_core != l_ph else 0.0,
                            )
                            if sim <= suf_sim + 0.03:
                                continue
                        # Simmetricamente, impedisce di inglobare parole rumore dopo una pausa (>= 0.50s) se il prefisso pre-pausa matcha già bene
                        if k >= 2:
                            pause_breaks = [
                                q for q in range(j + 1, j + k)
                                if (w_words[q]["start"] - w_words[q - 1]["end"]) >= 0.50
                            ]
                            if pause_breaks:
                                p_brk = pause_breaks[-1]
                                pre_acc = "".join(w_words[t]["phon"] for t in range(j, p_brk))
                                if len(pre_acc) >= 4:
                                    pre_sim = max(
                                        SequenceMatcher(None, l_ph, pre_acc).ratio(),
                                        SequenceMatcher(None, l_core, pre_acc).ratio() if l_core != l_ph else 0.0,
                                    )
                                    if sim <= pre_sim + 0.05:
                                        continue
                        ref_len = len(l_core) if (sim_core > sim_full and len(l_core) >= 8) else len(l_ph)
                        if sim >= 0.48:
                            len_eff = min(len(acc), ref_len, 22)
                            len_bonus = len_eff * 0.14
                            # Lieve preferenza cronologica: evita salti di 10+s e preferisce la prima riga in caso di doppioni identici
                            chrono_tie = 0.025 * (1.0 - (j / max(1, M))) + 0.004 * (1.0 - (si / max(1, S)))
                            score = (sim - 0.40) * (0.4 + 0.6 * sim) * (3.2 + len_bonus) + chrono_tie
                            span_t = w_words[j + k - 1]["end"] - w_words[j]["start"]
                            if span_t > 6.5:
                                score -= (span_t - 6.5) * 0.9
                            if v + score > dp[si + 1, j + k]:
                                dp[si + 1, j + k] = v + score
                                back[si + 1][j + k] = (si, j, "match")

        anchors = {}
        ci, cj = S, M
        while ci > 0 or cj > 0:
            st_b = back[ci][cj]
            if st_b is None:
                break
            pi, pj, op = st_b
            if op == "match":
                li_idx = seq_indices[pi]
                anchors[li_idx] = (
                    w_words[pj]["start"],
                    w_words[cj - 1]["end"],
                    w_words[pj]["seg_s"],
                    w_words[cj - 1]["seg_e"],
                )
            ci, cj = pi, pj

        def _containing_islands(t0: float, t1: float):
            return [isl for isl in islands if isl[1] > t0 + 0.08 and isl[0] < t1 - 0.08]

        def _build_block_cands(block_indices, exclude_inline_adlibs: bool = False):
            cand_lists = [[]]
            word_meta = []
            for bi in block_indices:
                w_flags = []
                mask = lines_info[bi]["adlib_mask"]
                for wi_orig, w in enumerate(lines_info[bi]["words"]):
                    if exclude_inline_adlibs and mask[wi_orig]:
                        continue
                    vars_w = word_variants(w)
                    if vars_w is None:
                        w_flags.append((w, False, wi_orig))
                    else:
                        w_flags.append((w, True, wi_orig))
                        new_cands = []
                        for cl in cand_lists:
                            for vw in vars_w:
                                new_cands.append(cl + [(len(vw), vw)])
                                if len(new_cands) >= 4:
                                    break
                            if len(new_cands) >= 4:
                                break
                        cand_lists = new_cands
                word_meta.append((bi, w_flags))
            return cand_lists, word_meta

        def _run_ctc_on_emission(sub_em, offset_sec: float, win_s_clamp: float, cand_lists: list, word_meta: list):
            best_spans = None
            best_counts = None
            best_score = -1e9
            for cl in cand_lists:
                all_sub = [tok for _, sub_l in cl for tok in sub_l]
                toks = tokenizer(all_sub)
                min_frames = int(sum(len(x) for x in toks) * 1.15) + 2
                if sub_em.shape[0] < min_frames:
                    continue
                try:
                    sp = aligner(sub_em, toks)
                    sc = float(np.mean([x.score for sub in sp for x in sub])) if sp else -1.0
                    if sc > best_score:
                        best_score = sc
                        best_spans = sp
                        best_counts = [cnt for cnt, _ in cl]
                except Exception:
                    continue

            if best_spans is None:
                return None

            out = {}
            cur = 0
            w_global_idx = 0
            for bi, w_flags in word_meta:
                wl = []
                for orig_w, has_ctc, wi_orig in w_flags:
                    if has_ctc:
                        cnt = best_counts[w_global_idx]
                        w_global_idx += 1
                        w_sp = best_spans[cur:cur + cnt]
                        cur += cnt
                        ws_t = offset_sec + (w_sp[0][0].start * fd)
                        we_t = offset_sec + (w_sp[-1][-1].end * fd)
                        sc = float(np.mean([x.score for sub in w_sp for x in sub]))
                        wl.append({"word": orig_w, "start": round(ws_t, 3), "end": round(we_t, 3), "score": sc, "orig_wi": wi_orig})
                    else:
                        wl.append({"word": orig_w, "start": None, "end": None, "score": 0.5, "orig_wi": wi_orig})

                for wi, item in enumerate(wl):
                    if item["start"] is None:
                        if wi > 0 and wl[wi - 1]["end"] is not None:
                            item["start"] = wl[wi - 1]["end"]
                            item["end"] = round(wl[wi - 1]["end"] + 0.04, 3)
                        else:
                            nxt_s = next((x["start"] for x in wl[wi + 1:] if x["start"] is not None), win_s_clamp + 0.04)
                            item["start"] = round(max(win_s_clamp, nxt_s - 0.04), 3)
                            item["end"] = round(max(item["start"] + 0.04, nxt_s), 3)
                out[bi] = wl
            return out

        def _run_ctc_on_waveform_chunks(wav_arr: np.ndarray, win_s: float, win_e: float, cand_lists: list, word_meta: list, chunk_sec: float = 4.5, pad_sec: float = 1.0):
            if mms_model is None or wav_arr is None or len(wav_arr) == 0:
                return None
            s_idx = max(0, int(round(win_s * 16000)))
            e_idx = min(len(wav_arr), int(round(win_e * 16000)))
            if e_idx - s_idx < 1600 or float(np.max(np.abs(wav_arr[s_idx:e_idx]))) <= 1e-4:
                return None
            c_samp = int(chunk_sec * 16000)
            p_samp = int(pad_sec * 16000)
            em_chunks = []
            try:
                with torch.inference_mode():
                    if (e_idx - s_idx) <= int(6.5 * 16000):
                        loc_t = torch.from_numpy(wav_arr[s_idx:e_idx]).unsqueeze(0).to(ctc_dev)
                        loc_em, _ = mms_model(loc_t)
                        em_mat = loc_em[0]
                    else:
                        for cur_s in range(s_idx, e_idx, c_samp):
                            cur_e = min(e_idx, cur_s + c_samp)
                            ws_p = s_idx if cur_s == s_idx else max(s_idx, cur_s - p_samp)
                            we_p = e_idx if cur_e == e_idx else min(e_idx, cur_e + p_samp)
                            seg_t = torch.from_numpy(wav_arr[ws_p:we_p]).unsqueeze(0).to(ctc_dev)
                            em, _ = mms_model(seg_t)
                            lt = (cur_s - ws_p) // 320
                            kf = int(np.ceil((cur_e - cur_s) / 320))
                            em_chunks.append(em[:, lt:lt + kf, :])
                        em_mat = torch.cat(em_chunks, dim=1)[0]
                return _run_ctc_on_emission(em_mat, s_idx / 16000.0, win_s, cand_lists, word_meta)
            except Exception:
                return None

        def _align_block_ctc(block_indices, win_s: float, win_e: float, prefer_local: bool = False, exclude_inline_adlibs: bool = True):
            f0 = max(0, int(np.floor(win_s / fd)))
            f1 = min(emission.shape[0], int(np.ceil(win_e / fd)))
            if f1 <= f0 + 4:
                return None

            cand_lists, word_meta = _build_block_cands(block_indices, exclude_inline_adlibs=exclude_inline_adlibs)
            if not cand_lists or not cand_lists[0]:
                return None

            out = _run_ctc_on_emission(emission[f0:f1], f0 * fd, win_s, cand_lists, word_meta)

            # Ri-valutazione CTC locale isolata se il blocco ha intro/ponte soffice o ad-lib adiacente a cori forti
            need_local = prefer_local or (out is None)
            if out is not None and not need_local:
                all_w = [w for bi in block_indices for w in out[bi]]
                mean_sc = float(np.mean([w["score"] for w in all_w])) if all_w else 0.0
                max_dur = max((w["end"] - w["start"]) for w in all_w) if all_w else 0.0
                if mean_sc < 0.08 or max_dur >= 1.25:
                    need_local = True

            if need_local and mms_model is not None and len(v_wav_full) > 0:
                loc_s = max(0.0, win_s - (0.0 if win_s > 0.15 else 0.12))
                loc_e = min(len(v_wav_full) / 16000.0, win_e + 0.05)
                loc_out = _run_ctc_on_waveform_chunks(v_wav_full, loc_s, loc_e, cand_lists, word_meta)
                if loc_out is not None:
                    if out is None or prefer_local:
                        out = loc_out
                    else:
                        sc_g = float(np.mean([w["score"] for bi in block_indices for w in out[bi]]))
                        sc_l = float(np.mean([w["score"] for bi in block_indices for w in loc_out[bi]]))
                        dur_g = max((w["end"] - w["start"]) for bi in block_indices for w in out[bi])
                        dur_l = max((w["end"] - w["start"]) for bi in block_indices for w in loc_out[bi])
                        if sc_l > sc_g + 0.015 or (dur_g >= 1.35 and dur_l < 1.15 and sc_l >= sc_g * 0.85):
                            out = loc_out

            return out

        def _refine_block_words(res_b, blk, eng_ref, win_s, b_first, prev_end_limit, next_start_limit, has_pending_bg_before, has_pending_bg_after):
            out_blk = {}
            for pos_in_blk, bi in enumerate(blk):
                wl = res_b[bi]
                p_lim = prev_end_limit if pos_in_blk == 0 else out_blk[blk[pos_in_blk - 1]][-1]["end"]
                n_lim = next_start_limit if pos_in_blk == len(blk) - 1 else res_b[blk[pos_in_blk + 1]][0]["start"]

                if (pos_in_blk == 0 and not has_pending_bg_before and len(wl) <= 2
                        and wl[0]["score"] < 0.35 and b_first[1] < wl[0]["start"] - 0.45):
                    gap_burst = eng_ref.find_strong_burst_in_gap(
                        max(p_lim + 0.10, b_first[3] - 0.25), wl[0]["start"] - 0.15, min_dur=0.16
                    )
                    if gap_burst:
                        wl[0]["start"] = gap_burst[0]

                prev_has_inline_adlib = (bi > 0 and lines_info[bi - 1].get("has_inline_adlibs", False))
                lookback = 0.08 if ((pos_in_blk == 0 and has_pending_bg_before) or prev_has_inline_adlib) else 0.50
                w0_sc = float(wl[0].get("score", 0.5))
                if wl[0]["start"] - p_lim >= 0.12:
                    wl[0]["start"] = eng_ref.expand_phrase_onset(
                        wl[0]["start"], max(p_lim + 0.015, win_s), max_lookback=lookback, max_t=wl[0]["end"]
                    )
                else:
                    sh_back = 0.020 if w0_sc >= 0.70 else 0.032
                    sh_fwd = 0.012 if w0_sc >= 0.70 else 0.025
                    wl[0]["start"] = eng_ref.snap_onset_micro(
                        wl[0]["start"], max(p_lim + 0.005, wl[0]["start"] - sh_back), wl[0]["end"] - 0.025,
                        max_shift=sh_back, max_shift_fwd=sh_fwd,
                    )
                if (wl[0]["end"] - wl[0]["start"]) >= 0.45:
                    wl[0]["start"] = max(
                        round(p_lim + 0.005, 3) if p_lim > 0 else round(win_s, 3),
                        eng_ref.trim_word_leading_silence(wl[0]["start"], wl[0]["end"]),
                    )
                w0_clean_str = clean_token_str(wl[0].get("word", ""))
                has_bg_or_adlib_before = (prev_has_inline_adlib or (pos_in_blk == 0 and has_pending_bg_before))
                if (wl[0]["end"] - wl[0]["start"]) >= 0.62 and (has_bg_or_adlib_before or (len(w0_clean_str) <= 5 and (pos_in_blk == 0 or w0_sc < 0.85))):
                    wl[0]["start"] = max(
                        wl[0]["start"],
                        eng_ref.find_onset_after_internal_dip(wl[0]["start"], wl[0]["end"], min_tail_dur=0.14, min_dip_ms=16),
                    )
                # Zero-drift: se la coda della riga precedente invade l'attacco di wl[0], rifila la riga precedente anziche' spingere avanti questa!
                if pos_in_blk > 0:
                    prev_wl = out_blk[blk[pos_in_blk - 1]]
                    if p_lim > wl[0]["start"] - 0.005:
                        if wl[0]["start"] > prev_wl[-1]["start"] + 0.045:
                            trim_t = round(max(prev_wl[-1]["start"] + 0.040, wl[0]["start"] - 0.005), 3)
                            prev_wl[-1]["end"] = trim_t
                        else:
                            min_ls = round(p_lim + 0.010, 3)
                            if wl[0]["start"] < min_ls and wl[-1]["end"] > p_lim + 0.05:
                                wl[0]["start"] = min_ls
                                for k_sh in range(len(wl)):
                                    if k_sh > 0 and wl[k_sh]["start"] < wl[k_sh - 1]["end"]:
                                        wl[k_sh]["start"] = wl[k_sh - 1]["end"]
                                    if wl[k_sh]["end"] < wl[k_sh]["start"] + 0.035:
                                        wl[k_sh]["end"] = round(wl[k_sh]["start"] + 0.035, 3)

                for k in range(1, len(wl)):
                    igap = wl[k]["start"] - wl[k - 1]["end"]
                    wk_sc = float(wl[k].get("score", 0.5))
                    if igap >= 0.22:
                        wl[k]["start"] = eng_ref.expand_phrase_onset(
                            wl[k]["start"],
                            wl[k - 1]["end"] + 0.02,
                            max_lookback=min(0.18, 0.55 * igap),
                            max_t=wl[k]["end"],
                            stop_at_valley=True,
                        )
                    else:
                        sh_b = 0.018 if wk_sc >= 0.70 else 0.030
                        sh_f = 0.010 if wk_sc >= 0.70 else 0.024
                        wl[k]["start"] = eng_ref.snap_onset_micro(
                            wl[k]["start"], max(wl[k - 1]["start"] + 0.045, wl[k]["start"] - sh_b), wl[k]["end"] - 0.025,
                            max_shift=sh_b, max_shift_fwd=sh_f,
                        )
                    if (wl[k]["end"] - wl[k]["start"]) >= 0.45:
                        wl[k]["start"] = max(
                            round(wl[k - 1]["end"] + 0.015, 3),
                            eng_ref.trim_word_leading_silence(wl[k]["start"], wl[k]["end"]),
                        )
                    if (wl[k - 1]["end"] - wl[k - 1]["start"]) >= 0.50 and wl[k - 1]["end"] <= wl[k]["start"]:
                        wl[k - 1]["end"] = eng_ref.refine_phrase_offset(
                            wl[k - 1]["start"], wl[k - 1]["end"], wl[k - 1]["start"] + 0.04, wl[k]["start"] - 0.01, extend_sustain=False
                        )
                    ng = wl[k]["start"] - wl[k - 1]["end"]
                    if 0 < ng <= 0.11 and eng_ref.has_active_energy(wl[k - 1]["end"], wl[k]["start"]):
                        wl[k - 1]["end"] = round(wl[k]["start"], 3)
                    elif ng < 0:
                        wl[k - 1]["end"] = round(wl[k]["start"], 3)
                    elif 0.025 <= ng <= 0.18 and (wl[k - 1]["end"] - wl[k - 1]["start"]) < 0.14 and eng_ref.num_frames > 0:
                        # Estendi vocali/parole brevi lungo la coda sonora reale fino al calo o all'attacco di wl[k]
                        f_cur = eng_ref._time_to_frame(wl[k - 1]["end"])
                        f_lim = min(eng_ref.num_frames, eng_ref._time_to_frame(wl[k]["start"] - 0.012))
                        dyn_tot_w = max(eng_ref.global_peak_total - eng_ref.global_floor_total, 0.20)
                        ext_f = f_cur
                        for cf_w in range(f_cur, f_lim):
                            if float(eng_ref.odf[cf_w]) >= 0.45 and cf_w > f_cur + 12:
                                break
                            if float(eng_ref.env_total[cf_w]) >= float(eng_ref.local_floor_total[cf_w]) + 0.16 * dyn_tot_w:
                                ext_f = cf_w + 1
                            else:
                                break
                        if ext_f > f_cur + 12:
                            wl[k - 1]["end"] = min(round(wl[k]["start"], 3), eng_ref._frame_to_time(ext_f))
                    elif ng >= 0.12:
                        has_mid_adlib_after = (
                            wl[k - 1].get("orig_wi") is not None
                            and wl[k].get("orig_wi") is not None
                            and (wl[k]["orig_wi"] - wl[k - 1]["orig_wi"]) > 1
                        )
                        wl[k - 1]["end"] = eng_ref.refine_phrase_offset(
                            wl[k - 1]["start"], wl[k - 1]["end"], wl[k - 1]["start"] + 0.04, wl[k]["start"] - 0.02,
                            extend_sustain=(not has_mid_adlib_after),
                        )

                if (pos_in_blk == len(blk) - 1 and not has_pending_bg_after and len(wl) == 1
                        and wl[0]["score"] < 0.35 and n_lim - wl[-1]["end"] >= 0.65):
                    post_burst = eng_ref.find_strong_burst_in_gap(
                        wl[-1]["end"] + 0.15, n_lim - 0.04, min_dur=0.25
                    )
                    if post_burst and post_burst[1] - post_burst[0] >= 0.30 and post_burst[1] - wl[0]["start"] <= 3.6:
                        wl[-1]["end"] = min(round(n_lim - 0.02, 3), post_burst[1])

                max_end_cap = max(wl[-1]["start"] + 0.045, n_lim - 0.008)
                has_trail_adlib = bool(lines_info[bi].get("adlib_mask") and lines_info[bi]["adlib_mask"][-1])
                wl[-1]["end"] = eng_ref.refine_phrase_offset(
                    wl[-1]["start"],
                    min(wl[-1]["end"], max_end_cap),
                    wl[-1]["start"] + 0.045,
                    max_end_cap,
                    extend_sustain=(not has_trail_adlib),
                )
                out_blk[bi] = wl
            return out_blk

        def _prev_blk_end(prev_wl):
            prev_w = prev_wl[-1]
            pe = prev_w["end"]
            if (pe - prev_w["start"]) >= 0.60:
                pe = acoustic_engine.refine_phrase_offset(
                    prev_w["start"], pe, prev_w["start"] + 0.045, pe, extend_sustain=False
                )
            return pe

        def _align_and_refine_block(blk, win_s, win_e, b_first, prev_end_limit, next_start_limit, has_pending_bg_before, has_pending_bg_after, ext_we_cand: float = 0.0):
            res_b = _align_block_ctc(blk, win_s, win_e)
            # Se singole righe di un blocco multi-riga hanno score debole (<0.12) o parole stirate (>=1.25s) per layer_norm adiacente a cori forti, ri-valuta localmente la singola riga
            if res_b and mms_model is not None and len(v_wav_full) > 0:
                for pos_b, bi in enumerate(blk):
                    wl_b = res_b.get(bi)
                    if not wl_b or len(wl_b) < 2:
                        continue
                    sc_b = float(np.mean([w["score"] for w in wl_b]))
                    max_wd_b = max((w["end"] - w["start"]) for w in wl_b)
                    if sc_b < 0.12 or max_wd_b >= 1.25:
                        p_b = _prev_blk_end(res_b[blk[pos_b - 1]]) + 0.015 if pos_b > 0 else win_s
                        n_b = res_b[blk[pos_b + 1]][0]["start"] - 0.015 if pos_b + 1 < len(blk) else win_e
                        anc_b = anchors.get(bi)
                        if sc_b < 0.08 and pos_b > 0 and (wl_b[0]["start"] - p_b) <= 3.5:
                            t_s = max(win_s, p_b)
                        else:
                            t_s = max(win_s, p_b, (min(wl_b[0]["start"], anc_b[0]) - 0.08) if anc_b else (wl_b[0]["start"] - 0.12))
                        t_e = min(win_e, n_b, (max(wl_b[-1]["end"], anc_b[1]) + 0.25) if anc_b else (wl_b[-1]["end"] + 0.25))
                        if 0.45 <= (t_e - t_s) <= 8.5:
                            c1, m1 = _build_block_cands([bi], exclude_inline_adlibs=True)
                            if c1 and c1[0]:
                                iso_b = _run_ctc_on_waveform_chunks(v_wav_full, t_s, t_e, c1, m1)
                                if iso_b and bi in iso_b:
                                    iso_wl = iso_b[bi]
                                    sc_iso = float(np.mean([w["score"] for w in iso_wl]))
                                    max_wd_iso = max((w["end"] - w["start"]) for w in iso_wl)
                                    accept_iso = (
                                        (sc_iso > sc_b + 0.04 and max_wd_iso <= max_wd_b + 0.15)
                                        or (sc_iso >= max(sc_b + 0.08, sc_b * 2.0) and max_wd_iso < 1.25)
                                        or (max_wd_b >= 1.25 and max_wd_iso < 1.15 and sc_iso >= sc_b)
                                    )
                                    if accept_iso:
                                        res_b[bi] = iso_wl

            # Se un ad-lib o rumore su un'isola precedente precede il vero attacco del blocco e causa basso score o stretching anomalo, prova a saltare isl[0]
            if res_b and (has_pending_bg_before or len(lines_info[blk[0]]["words"]) >= 3):
                blk_isls = [isl for isl in islands if isl[1] > win_s + 0.10 and isl[0] < win_e - 0.10]
                if len(blk_isls) >= 2 and (blk_isls[1][0] - blk_isls[0][1]) >= 0.25 and res_b[blk[0]][0]["start"] < blk_isls[1][0] - 0.10:
                    head_bis = blk[:min(2, len(blk))]
                    max_wd_orig = max((w["end"] - w["start"]) for bi in head_bis for w in res_b[bi])
                    sc_orig = float(np.mean([w["score"] for bi in head_bis for w in res_b[bi]]))
                    anchor_after_isl0 = (b_first[1] > blk_isls[0][1] + 0.05)
                    if has_pending_bg_before or anchor_after_isl0 or max_wd_orig >= 1.45:
                        alt_ws = max(win_s, blk_isls[1][0] - 0.08)
                        res_alt = _align_block_ctc(blk, alt_ws, win_e)
                        if res_alt:
                            max_wd_alt = max((w["end"] - w["start"]) for bi in head_bis for w in res_alt[bi])
                            sc_alt = float(np.mean([w["score"] for bi in head_bis for w in res_alt[bi]]))
                            alt_span = res_alt[blk[0]][-1]["end"] - res_alt[blk[0]][0]["start"]
                            anchor_inside_isl0 = (b_first[0] == "confirmed" and b_first[1] < blk_isls[0][1] - 0.20)
                            not_cramped = not (anchor_inside_isl0 and len(res_alt[blk[0]]) >= 5 and alt_span < len(res_alt[blk[0]]) * 0.17)
                            if not_cramped and (sc_alt > sc_orig + 0.04 or (max_wd_orig >= 1.45 and max_wd_alt < 1.25 and sc_alt >= sc_orig - 0.02)):
                                res_b = res_alt
                                win_s = alt_ws

            # Se la prima parola breve/anomala è stirata dal CTC su bleed/silenzio iniziale, ri-allinea vicino a w0["end"] - 0.50s
            if res_b and blk[0] in res_b and len(res_b[blk[0]]) >= 2:
                w0_b, w1_b = res_b[blk[0]][0], res_b[blk[0]][1]
                w0_clean = re.sub(r"[^\w]", "", w0_b["word"])
                w0_dur = w0_b["end"] - w0_b["start"]
                tail_med = float(np.median([w["end"] - w["start"] for w in res_b[blk[0]][1:]]))
                is_stretched_head = (
                    (len(w0_clean) <= (7 if has_pending_bg_before else 5) and w0_dur >= (0.78 if has_pending_bg_before else 0.85))
                    or (len(res_b[blk[0]]) >= 3 and len(w0_clean) <= 8 and w0_dur >= max(0.75 if has_pending_bg_before else 0.95, 2.2 * tail_med))
                )
                if is_stretched_head:
                    alt_ws2 = max(win_s + 0.15, w0_b["end"] - 0.50)
                    if alt_ws2 < w1_b["start"] - 0.08 and alt_ws2 < win_e - 0.50:
                        res_alt2 = _align_block_ctc(blk, alt_ws2, win_e)
                        if res_alt2 and blk[0] in res_alt2:
                            alt_w0 = res_alt2[blk[0]][0]
                            sc_orig2 = float(np.mean([w["score"] for w in res_b[blk[0]]]))
                            sc_alt2 = float(np.mean([w["score"] for w in res_alt2[blk[0]]]))
                            if (alt_w0["end"] - alt_w0["start"]) < (w0_b["end"] - w0_b["start"]) * 0.65 and sc_alt2 >= sc_orig2 * 0.85:
                                res_b = res_alt2
                                win_s = alt_ws2

            if res_b and ext_we_cand > win_e + 0.30 and blk[-1] in res_b and len(res_b[blk[-1]]) >= 3:
                li_b = blk[-1]
                wl_orig_l = res_b[li_b]
                p_tail_s = _prev_blk_end(res_b[blk[-2]]) + 0.015 if len(blk) >= 2 else win_s
                tail_k = min(2, len(wl_orig_l))
                sc_tail_orig = float(np.mean([w["score"] for w in wl_orig_l[-tail_k:]]))
                sc_line_orig = float(np.mean([w["score"] for w in wl_orig_l]))
                res_tail = _align_block_ctc([li_b], max(win_s, p_tail_s), ext_we_cand)
                if res_tail and li_b in res_tail and len(res_tail[li_b]) >= 3:
                    wl_ext_l = res_tail[li_b]
                    sc_tail_ext = float(np.mean([w["score"] for w in wl_ext_l[-tail_k:]]))
                    sc_line_ext = float(np.mean([w["score"] for w in wl_ext_l]))
                    max_wd_ext = max((w["end"] - w["start"]) for w in wl_ext_l)
                    if (wl_ext_l[-1]["end"] > win_e + 0.15 and max_wd_ext < 1.35
                            and sc_line_ext >= sc_line_orig - 0.02
                            and (sc_tail_ext > sc_tail_orig + 0.04
                                 or (wl_ext_l[-1]["end"] > win_e + 0.25 and float(wl_ext_l[-1]["score"]) >= max(0.18, float(wl_orig_l[-1]["score"]) + 0.03)))):
                        res_b[li_b] = wl_ext_l
                        win_e = ext_we_cand

            if not res_b:
                dur = max(0.3, win_e - win_s - 0.1)
                tot_w = sum(len(lines_info[bi]["words"]) for bi in blk)
                cur_t = win_s + 0.05
                dt = dur / max(1, tot_w)
                res_b = {}
                for bi in blk:
                    res_b[bi] = []
                    for w in lines_info[bi]["words"]:
                        res_b[bi].append({"word": w, "start": round(cur_t, 3), "end": round(cur_t + dt, 3), "score": 0.0})
                        cur_t += dt

            return _refine_block_words(
                res_b, blk, acoustic_engine, win_s, b_first, prev_end_limit, next_start_limit,
                has_pending_bg_before, has_pending_bg_after,
            )

        # Blocchi continui per Pass 1A (evitando split forzati intra-segmento < 16 righe)
        blocks_1a = []
        cur_blk = []
        for k in range(S):
            si = seq_indices[k]
            if si in anchors:
                if not cur_blk:
                    cur_blk = [si]
                else:
                    prev_i = cur_blk[-1]
                    a_p = anchors[prev_i]
                    a_c = anchors[si]
                    same_seg = (a_p[2] == a_c[2] and a_p[3] == a_c[3])
                    gap_est = a_c[0] - a_p[1]
                    contiguous_seq = (seq_indices[k - 1] == prev_i)
                    split = (
                        not contiguous_seq
                        or len(cur_blk) >= 16
                        or (not same_seg and gap_est >= 0.40)
                        or (not same_seg and (lines_info[prev_i]["is_bg"] != lines_info[si]["is_bg"] or lines_info[prev_i]["agent"] != lines_info[si]["agent"]))
                    )
                    if split:
                        blocks_1a.append(cur_blk)
                        cur_blk = [si]
                    else:
                        cur_blk.append(si)
            else:
                if cur_blk:
                    blocks_1a.append(cur_blk)
                    cur_blk = []
        if cur_blk:
            blocks_1a.append(cur_blk)

        final_lines = [None] * len(lines_info)

        # PASS 1A: Allineamento di precisione sui blocchi lessicali ancorati
        for b_idx, blk in enumerate(blocks_1a):
            first_i = blk[0]
            last_i = blk[-1]
            a_first = anchors[first_i]
            a_last = anchors[last_i]

            prev_seq_idx = -1
            prev_end_limit = 0.0
            if b_idx > 0:
                prev_blk_last = blocks_1a[b_idx - 1][-1]
                prev_seq_idx = prev_blk_last
                if final_lines[prev_blk_last]:
                    k_between = sum(1 for x in seq_indices if prev_blk_last < x < first_i)
                    prev_end_limit = final_lines[prev_blk_last][-1]["end"] + 0.35 * k_between

            next_seq_idx = len(lines_info)
            next_start_limit = total_audio_sec
            same_seg_with_next = False
            k_between_next = 0
            if b_idx + 1 < len(blocks_1a):
                nxt_first = blocks_1a[b_idx + 1][0]
                next_seq_idx = nxt_first
                k_between_next = sum(1 for x in seq_indices if last_i < x < nxt_first)
                next_start_limit = anchors[nxt_first][0] - 0.70 * k_between_next
                a_nxt = anchors[nxt_first]
                same_seg_with_next = (a_last[2] == a_nxt[2] and a_last[3] == a_nxt[3] and k_between_next == 0)

            has_pending_bg_before = any(
                lines_info[x]["is_bg"] and final_lines[x] is None
                for x in range(max(0, prev_seq_idx + 1), first_i)
            )
            has_pending_bg_after = any(
                lines_info[x]["is_bg"] and final_lines[x] is None
                for x in range(last_i + 1, min(len(lines_info), next_seq_idx))
            )

            if has_pending_bg_before:
                raw_ws = (a_first[0] - 0.25) if (a_first[0] > a_first[2] + 0.15) else (min(a_first[0], a_first[2]) - 0.30)
            else:
                raw_ws = min(a_first[0], a_first[2]) - 0.65
            raw_we = max(a_last[1], a_last[3]) + 0.65
            if len(blk) == 1:
                isls = _containing_islands(a_first[0] - 0.5, a_first[1] + 0.5)
                if len(isls) == 1:
                    raw_ws = min(raw_ws, isls[0][0] - 0.25)
                    raw_we = max(raw_we, isls[0][1] + 0.25)

            win_s = max(0.0, prev_end_limit - 0.05 if prev_end_limit > 0 else 0.0, raw_ws)
            end_margin = 0.85 if same_seg_with_next else (0.0 if k_between_next > 0 else 0.25)
            win_e = min(total_audio_sec, next_start_limit + end_margin if next_start_limit < total_audio_sec else total_audio_sec, raw_we)
            if win_e <= win_s + 0.2:
                win_e = min(total_audio_sec, win_s + 2.5)

            ext_we_cand = 0.0
            if (b_idx + 1 < len(blocks_1a) and k_between_next == 0 and not has_pending_bg_after
                    and not lines_info[last_i]["is_bg"]
                    and len(lines_info[last_i]["words"]) >= 3 and (next_start_limit - win_e) >= 0.30):
                a_nxt_bound = min(total_audio_sec, next_start_limit - 0.05)
                tail_anchor_e = max(a_last[1], a_last[3])
                ext_isls = [
                    isl for isl in islands
                    if isl[0] >= (win_e - 0.20) and isl[1] <= (a_nxt_bound - 0.15)
                    and (isl[0] - tail_anchor_e) <= 2.50
                ]
                if ext_isls:
                    ext_we_cand = min(a_nxt_bound, ext_isls[0][1] + 1.25)
                else:
                    gap_b = acoustic_engine.find_strong_burst_in_gap(
                        max(win_s + 0.25, win_e - 0.25), a_nxt_bound, min_dur=0.14
                    )
                    if gap_b and (gap_b[0] - tail_anchor_e) <= 2.50:
                        ext_we_cand = min(a_nxt_bound, max(win_e + 0.50, gap_b[1] + 0.20))

            b_first_tup = ("confirmed", a_first[0], a_first[1], a_first[2], a_first[3])
            out_blk = _align_and_refine_block(
                blk, win_s, win_e, b_first_tup, prev_end_limit, next_start_limit,
                has_pending_bg_before, has_pending_bg_after, ext_we_cand=ext_we_cand,
            )
            for bi, wl in out_blk.items():
                final_lines[bi] = wl

        # PASS 1B: Assegnazione isole vocali / interiezioni / gap sui confini esatti del Pass 1A
        def _try_filtered_vocal_gap(sub_blk, g_s: float, g_e: float):
            if f_wav_full is None or mms_model is None or (g_e - g_s) < 1.2:
                return None
            # Salta solo l'eventuale silenzio digitale puro iniziale (es. 0.00..0.10s)
            scan_s = g_s
            s_i0 = max(0, int(g_s * 16000))
            s_i1 = min(len(f_wav_full), int(g_e * 16000))
            if s_i1 - s_i0 < 1600:
                return None
            step_s = 800  # 50ms
            while s_i0 + step_s < s_i1:
                rms_blk = float(np.sqrt(np.mean(f_wav_full[s_i0:s_i0 + step_s] ** 2)))
                if rms_blk >= 0.004:
                    break
                s_i0 += step_s
            while s_i1 - step_s > s_i0:
                rms_blk = float(np.sqrt(np.mean(f_wav_full[s_i1 - step_s:s_i1] ** 2)))
                if rms_blk >= 0.004:
                    break
                s_i1 -= step_s
            scan_s = s_i0 / 16000.0
            scan_e = s_i1 / 16000.0
            if scan_e - scan_s < 0.8:
                return None
            cand_lists, word_meta = _build_block_cands(sub_blk, exclude_inline_adlibs=True)
            if not cand_lists or not cand_lists[0]:
                return None
            filt_res = _run_ctc_on_waveform_chunks(f_wav_full, scan_s, scan_e, cand_lists, word_meta, chunk_sec=4.5, pad_sec=1.0)
            if not filt_res:
                return None
            # Raffinamento acustico leggero sullo stem filtrato locale
            pad_f = max(0, int(scan_s * 16000))
            end_f = min(len(f_wav_full), int(scan_e * 16000))
            loc_slice = f_wav_full[pad_f:end_f]
            pk_loc = float(np.max(np.abs(loc_slice)))
            if pk_loc <= 1e-4:
                return None
            loc_eng = AcousticPrecisionEngine((loc_slice / pk_loc).astype(np.float32))
            shifted = {}
            for bi in sub_blk:
                shifted[bi] = [
                    {
                        "word": w["word"],
                        "start": max(0.0, round(w["start"] - scan_s, 3)),
                        "end": max(0.04, round(w["end"] - scan_s, 3)),
                        "score": w["score"],
                        "orig_wi": w.get("orig_wi"),
                    }
                    for w in filt_res[bi]
                ]
            ref_loc = _refine_block_words(
                shifted, sub_blk, loc_eng, 0.0, ("filt", 0.0, scan_e - scan_s, 0.0, scan_e - scan_s),
                0.0, scan_e - scan_s, False, False
            )
            out_global = {}
            for bi in sub_blk:
                out_global[bi] = [
                    {
                        "word": w["word"],
                        "start": round(scan_s + w["start"], 3),
                        "end": round(scan_s + w["end"], 3),
                        "score": w["score"],
                        "orig_wi": w.get("orig_wi"),
                    }
                    for w in ref_loc[bi]
                ]
            return out_global

        p_idx = 0
        while p_idx < S:
            li_idx = seq_indices[p_idx]
            if final_lines[li_idx] is not None:
                p_idx += 1
                continue
            run_end = p_idx
            while run_end < S and final_lines[seq_indices[run_end]] is None:
                run_end += 1
            prev_t = final_lines[seq_indices[p_idx - 1]][-1]["end"] if p_idx > 0 and final_lines[seq_indices[p_idx - 1]] else 0.0
            next_t = final_lines[seq_indices[run_end]][0]["start"] if run_end < S and final_lines[seq_indices[run_end]] else total_audio_sec

            k_lines = run_end - p_idx
            sub_blk = [seq_indices[p_idx + k_off] for k_off in range(k_lines)]
            # Ritaglia ogni isola sui confini [prev_t, next_t] anziché escludere isole che attraversano il confine
            cand_isls = []
            for isl_s, isl_e in islands:
                cs = max(prev_t + 0.01, isl_s)
                ce = min(next_t - 0.01, isl_e)
                if ce - cs >= 0.14:
                    cand_isls.append((round(cs, 3), round(ce, 3)))
            work_isls = list(cand_isls)
            while len(work_isls) > k_lines:
                gaps = [(work_isls[g + 1][0] - work_isls[g][1], g) for g in range(len(work_isls) - 1)]
                gaps.sort()
                if not gaps or gaps[0][0] > 0.75:
                    break
                gi = gaps[0][1]
                work_isls[gi] = (work_isls[gi][0], work_isls[gi + 1][1])
                work_isls.pop(gi + 1)
            while len(work_isls) > k_lines:
                durs = [(e - s, idx_i) for idx_i, (s, e) in enumerate(work_isls)]
                durs.sort()
                if durs[0][0] < 0.25:
                    work_isls.pop(durs[0][1])
                else:
                    break
            # Riserva eventuali isole adiacenti ai margini che appartengono a righe (BG) pendenti prima o dopo il run
            if len(work_isls) > k_lines and k_lines > 0:
                prev_seq_u = seq_indices[p_idx - 1] + 1 if p_idx > 0 else 0
                next_seq_u = seq_indices[run_end] if run_end < S else len(lines_info)
                n_bg_before = sum(1 for x in range(prev_seq_u, seq_indices[p_idx]) if lines_info[x]["is_bg"] and final_lines[x] is None)
                n_bg_after = sum(1 for x in range(seq_indices[run_end - 1] + 1, next_seq_u) if lines_info[x]["is_bg"] and final_lines[x] is None)
                if n_bg_after > 0:
                    drop_r = min(n_bg_after, len(work_isls) - k_lines)
                    work_isls = work_isls[:-drop_r] if drop_r > 0 else work_isls
                if n_bg_before > 0 and len(work_isls) > k_lines:
                    drop_l = min(n_bg_before, len(work_isls) - k_lines)
                    work_isls = work_isls[drop_l:] if drop_l > 0 else work_isls

            # Se c'e' un'isola chiara per la prima riga e le successive sono in zona beat/filtered (es. intro tag), proteggi la prima isola!
            if k_lines >= 2 and len(work_isls) == 1 and (work_isls[0][1] - work_isls[0][0]) <= 1.50 and (next_t - work_isls[0][1]) >= 1.20:
                is_s0, is_e0 = work_isls[0]
                ui0 = seq_indices[p_idx]
                c_words0 = [w for w, is_ad in zip(lines_info[ui0]["words"], lines_info[ui0]["adlib_mask"]) if not is_ad]
                orig_idx0 = [wi_o for wi_o, is_ad in enumerate(lines_info[ui0]["adlib_mask"]) if not is_ad]
                res0 = _align_block_ctc([ui0], max(prev_t, is_s0 - 0.08), min(next_t, is_e0 + 0.08))
                if res0 and ui0 in res0 and res0[ui0]:
                    ref0 = _refine_block_words(res0, [ui0], acoustic_engine, max(prev_t, is_s0 - 0.08), ("isl0", is_s0, is_e0, is_s0, is_e0), prev_t, min(next_t, is_e0 + 0.08), False, False)
                    final_lines[ui0] = ref0[ui0]
                else:
                    dt0 = (is_e0 - is_s0) / max(1, len(c_words0))
                    final_lines[ui0] = [{"word": w, "start": round(is_s0 + kk * dt0, 3), "end": round(is_s0 + (kk + 1) * dt0, 3), "score": 0.5, "orig_wi": orig_idx0[kk]} for kk, w in enumerate(c_words0)]
                p_idx += 1
                continue

            voc_cand_out = None
            if len(work_isls) == k_lines and k_lines > 0:
                voc_cand_out = {}
                for k_off in range(k_lines):
                    ui = seq_indices[p_idx + k_off]
                    is_s, is_e = work_isls[k_off]
                    words_l = [w for w, is_ad in zip(lines_info[ui]["words"], lines_info[ui]["adlib_mask"]) if not is_ad]
                    orig_wis = [wi_o for wi_o, is_ad in enumerate(lines_info[ui]["adlib_mask"]) if not is_ad]
                    if lines_info[ui]["nonlex"] and len(words_l) == 1:
                        voc_cand_out[ui] = [{"word": words_l[0], "start": round(is_s, 3), "end": round(is_e, 3), "score": 0.99, "orig_wi": orig_wis[0]}]
                    elif lines_info[ui]["nonlex"]:
                        res_b = _align_block_ctc([ui], max(0.0, is_s - 0.08), min(total_audio_sec, is_e + 0.08))
                        if res_b and ui in res_b:
                            wl = res_b[ui]
                            wl[0]["start"] = round(is_s, 3)
                            wl[-1]["end"] = round(is_e, 3)
                            for kk in range(1, len(wl)):
                                if wl[kk]["start"] < wl[kk - 1]["start"] + 0.05:
                                    wl[kk]["start"] = round(wl[kk - 1]["start"] + 0.05, 3)
                                wl[kk - 1]["end"] = round(min(wl[kk]["start"], max(wl[kk - 1]["end"], wl[kk]["start"])), 3)
                            voc_cand_out[ui] = wl
                        else:
                            dt = (is_e - is_s) / max(1, len(words_l))
                            voc_cand_out[ui] = [{"word": w, "start": round(is_s + kk * dt, 3), "end": round(is_s + (kk + 1) * dt, 3), "score": 0.0, "orig_wi": orig_wis[kk]} for kk, w in enumerate(words_l)]
                    else:
                        b_tup = ("island", is_s, is_e, is_s, is_e)
                        out_b = _align_and_refine_block([ui], max(prev_t, is_s - 0.15), min(next_t, is_e + 0.15), b_tup, prev_t, next_t, False, False)
                        voc_cand_out[ui] = out_b[ui]
            elif len(work_isls) >= 1:
                ws_u = max(prev_t, work_isls[0][0] - 0.10)
                we_u = min(next_t, work_isls[-1][1] + 0.10)
                b_tup = ("isl_span", ws_u, we_u, ws_u, we_u)
                voc_cand_out = _align_and_refine_block(sub_blk, ws_u, we_u, b_tup, prev_t, next_t, False, False)

            # Verifica se lo stem vocals ha perso voci filtrate (intro/ponte low-pass, telefono, ecc.)
            need_filt_check = (voc_cand_out is None)
            voc_cramped = False
            voc_sc = 0.0
            max_v_dur = 0.0
            if voc_cand_out is not None:
                all_vw = [w for bi in sub_blk for w in voc_cand_out[bi]]
                voc_sc = float(np.mean([w.get("score", 0.0) for w in all_vw])) if all_vw else 0.0
                max_v_dur = max((w["end"] - w["start"]) for w in all_vw) if all_vw else 0.0
                voc_cramped = any(
                    len(voc_cand_out[bi]) >= 3 and (voc_cand_out[bi][-1]["end"] - voc_cand_out[bi][0]["start"]) < 0.72
                    for bi in sub_blk
                )
                intro_missing_head = (prev_t <= 0.05 and k_lines >= 2 and work_isls and work_isls[0][0] >= 2.0)
                if voc_cramped or voc_sc < 0.13 or intro_missing_head:
                    need_filt_check = True

            chosen_out = voc_cand_out
            if need_filt_check and f_wav_full is not None and (next_t - prev_t) >= max(1.2, k_lines * 1.4):
                filt_out = _try_filtered_vocal_gap(sub_blk, prev_t, next_t)
                ext_applied = None
                if filt_out is not None and prev_t <= 0.05 and run_end < S and final_lines[seq_indices[run_end]]:
                    nxt_wl = final_lines[seq_indices[run_end]]
                    if len(nxt_wl) >= 2 and len(nxt_wl[0]["word"]) <= 5 and (nxt_wl[0]["end"] - nxt_wl[0]["start"]) >= 0.65:
                        ext_ge = min(next_t + 4.50, nxt_wl[1]["start"] - 0.08)
                        if ext_ge > next_t + 0.20:
                            ext_out = _try_filtered_vocal_gap(sub_blk, prev_t, ext_ge)
                            if ext_out is not None:
                                sc_base = float(np.mean([w.get("score", 0.0) for bi in sub_blk for w in filt_out[bi]]))
                                sc_ext = float(np.mean([w.get("score", 0.0) for bi in sub_blk for w in ext_out[bi]]))
                                if sc_ext >= sc_base + 0.015 and ext_out[sub_blk[-1]][-1]["end"] > next_t + 0.10:
                                    filt_out = ext_out
                                    ext_applied = min(round(nxt_wl[0]["end"] - 0.04, 3), round(ext_out[sub_blk[-1]][-1]["end"] + 0.02, 3))
                if filt_out is not None:
                    all_fw = [w for bi in sub_blk for w in filt_out[bi]]
                    filt_sc = float(np.mean([w.get("score", 0.0) for w in all_fw])) if all_fw else 0.0
                    max_f_dur = max((w["end"] - w["start"]) for w in all_fw) if all_fw else 0.0
                    filt_cramped = any(
                        len(filt_out[bi]) >= 3 and (filt_out[bi][-1]["end"] - filt_out[bi][0]["start"]) < 0.72
                        for bi in sub_blk
                    )
                    filt_stretched = (max_f_dur >= 1.55 and (voc_cand_out is not None and max_v_dur < 1.15))
                    filt_wins = False
                    if voc_cand_out is None:
                        filt_wins = (filt_sc >= 0.06)
                    elif not filt_stretched:
                        covers_missing_head = bool(
                            intro_missing_head and not filt_cramped and filt_sc >= voc_sc - 0.01
                            and filt_out[sub_blk[0]][0]["start"] <= work_isls[0][0] - 0.8
                        )
                        filt_wins = (
                            (voc_cramped and not filt_cramped and filt_sc >= voc_sc - 0.01)
                            or (filt_sc >= voc_sc + 0.025)
                            or covers_missing_head
                        )
                    if filt_wins:
                        chosen_out = filt_out
                        if ext_applied is not None and ext_applied > final_lines[seq_indices[run_end]][0]["start"]:
                            final_lines[seq_indices[run_end]][0]["start"] = ext_applied

            if chosen_out is not None:
                for bi, wl in chosen_out.items():
                    final_lines[bi] = wl
            else:
                step_t = max(0.20, (next_t - prev_t) / max(1, k_lines))
                for k_off in range(k_lines):
                    ui = seq_indices[p_idx + k_off]
                    gs = prev_t + k_off * step_t
                    ge = min(next_t, prev_t + (k_off + 1) * step_t)
                    wl_words = [w for w, is_ad in zip(lines_info[ui]["words"], lines_info[ui]["adlib_mask"]) if not is_ad]
                    orig_wis = [wi_o for wi_o, is_ad in enumerate(lines_info[ui]["adlib_mask"]) if not is_ad]
                    dt = max(0.05, (ge - gs) / max(1, len(wl_words)))
                    final_lines[ui] = [{"word": w, "start": round(gs + kk * dt, 3), "end": round(gs + (kk + 1) * dt, 3), "score": 0.0, "orig_wi": orig_wis[kk]} for kk, w in enumerate(wl_words)]
            p_idx = run_end

        # Recupero post-Pass-1A/1B: se una riga ha parole finali ripetute/posticipate dentro un vuoto >=0.55s scoperto dopo l'allineamento della riga successiva
        for bi in range(len(lines_info) - 1):
            wl_cur = final_lines[bi]
            wl_nxt = final_lines[bi + 1]
            if (not wl_cur or not wl_nxt or len(wl_cur) < 3
                    or lines_info[bi]["is_bg"] or lines_info[bi + 1]["is_bg"]
                    or lines_info[bi].get("has_trailing_parens")):
                continue
            cur_e = float(wl_cur[-1]["end"])
            nxt_s = float(wl_nxt[0]["start"])
            if nxt_s - cur_e < 0.55:
                continue
            gap_burst = acoustic_engine.find_strong_burst_in_gap(cur_e + 0.12, nxt_s - 0.08, min_dur=0.14)
            if not gap_burst or (gap_burst[0] - cur_e) > 2.20:
                continue
            w_s_rec = max(0.0, float(wl_cur[0]["start"]) - 0.06)
            w_e_rec = min(total_audio_sec, nxt_s - 0.03)
            res_rec = _align_block_ctc([bi], w_s_rec, w_e_rec)
            if not res_rec or bi not in res_rec or len(res_rec[bi]) != len(wl_cur):
                continue
            wl_rec = res_rec[bi]
            sc_orig_m = float(np.mean([float(x.get("score", 0.0)) for x in wl_cur]))
            sc_rec_m = float(np.mean([float(x.get("score", 0.0)) for x in wl_rec]))
            prev_orig_s = float(wl_cur[-2].get("score", 0.0))
            prev_rec_s = float(wl_rec[-2].get("score", 0.0))
            last_orig_s = float(wl_cur[-1].get("score", 0.0))
            last_rec_s = float(wl_rec[-1].get("score", 0.0))
            if (wl_rec[-1]["start"] >= gap_burst[0] - 0.12
                    and wl_rec[-1]["end"] <= nxt_s - 0.015
                    and (wl_rec[-1]["end"] - wl_rec[-1]["start"]) <= 1.25
                    and wl_rec[-2]["end"] <= gap_burst[0] - 0.08
                    and prev_rec_s >= prev_orig_s - 0.05
                    and last_rec_s >= max(0.20, last_orig_s + 0.03)
                    and sc_rec_m >= sc_orig_m - 0.01):
                p_rec_lim = max((final_lines[p][-1]["end"] for p in range(bi - 1, -1, -1) if final_lines[p]), default=0.0)
                ref_rec = _refine_block_words(
                    {bi: wl_rec}, [bi], acoustic_engine, w_s_rec,
                    ("rec", wl_cur[0]["start"], wl_rec[-1]["end"], w_s_rec, w_e_rec),
                    p_rec_lim, nxt_s, False, False,
                )
                if ref_rec and bi in ref_rec:
                    final_lines[bi] = ref_rec[bi]

        # PASS 2: Allineamento Background / Echo / Gap Ad-Libs (gestendo anche run consecutivi di {bg})
        li_idx = 0
        while li_idx < len(lines_info):
            if final_lines[li_idx] is not None:
                li_idx += 1
                continue
            bg_run_end = li_idx + 1
            while bg_run_end < len(lines_info) and final_lines[bg_run_end] is None:
                bg_run_end += 1
            bg_run = list(range(li_idx, bg_run_end))

            p_end = 0.0
            p_words = []
            for b in range(li_idx - 1, -1, -1):
                if final_lines[b]:
                    p_end = final_lines[b][-1]["end"]
                    p_words = final_lines[b]
                    break
            n_start = total_audio_sec
            for f_i in range(bg_run_end, len(lines_info)):
                if final_lines[f_i]:
                    n_start = final_lines[f_i][0]["start"]
                    break

            if len(bg_run) >= 2:
                gap_s = max(0.0, p_end - 0.02)
                gap_e = max(gap_s, n_start - 0.03)
                bg_isls = []
                for isl_s, isl_e in islands:
                    cs = max(gap_s, isl_s)
                    ce = min(gap_e, isl_e)
                    if ce - cs >= 0.14:
                        bg_isls.append((round(cs, 3), round(ce, 3)))
                w_in_gap = [
                    s for s in whisper_segments
                    if not _is_whisper_hallucination(str(s.get("text", "")))
                    and float(s.get("end", 0.0)) > gap_s + 0.1
                    and float(s.get("start", 0.0)) < gap_e - 0.1
                ]
                if w_in_gap:
                    w_end_lim = float(w_in_gap[-1].get("end", gap_e))
                    w_start_lim = float(w_in_gap[0].get("start", gap_s))
                    bg_isls_f = [
                        isl for isl in bg_isls
                        if isl[0] <= w_end_lim + 1.50 or (isl[1] - isl[0]) >= 1.25
                    ]
                    if bg_isls_f:
                        bg_isls = bg_isls_f
                    ws_run = max(gap_s, min(bg_isls[0][0] - 0.10 if bg_isls else w_start_lim - 0.10, w_start_lim - 0.10))
                    we_run = min(gap_e, max(bg_isls[-1][1] + 0.12 if bg_isls else w_end_lim + 0.15, w_end_lim + 0.15))
                else:
                    while len(bg_isls) > len(bg_run) and (bg_isls[-1][1] - bg_isls[-1][0]) < 1.0:
                        if (bg_isls[-1][0] - bg_isls[-2][1]) >= 2.2:
                            bg_isls.pop()
                        else:
                            break
                    ws_run = max(gap_s, bg_isls[0][0] - 0.10) if bg_isls else gap_s
                    we_run = min(gap_e, bg_isls[-1][1] + 0.12) if bg_isls else gap_e
                if we_run <= ws_run + 0.25:
                    we_run = min(total_audio_sec, max(ws_run + 0.60, gap_e))

                res_run = _align_block_ctc(bg_run, ws_run, we_run, prefer_local=True)
                if res_run and all(bi in res_run for bi in bg_run):
                    ref_bg = _refine_block_words(
                        res_run, bg_run, acoustic_engine, ws_run,
                        ("bg_run", ws_run, we_run, ws_run, we_run),
                        p_end, min(n_start - 0.02, we_run + 0.15), False, False,
                    )
                    for bi, wl in ref_bg.items():
                        final_lines[bi] = wl
                    li_idx = bg_run_end
                    continue

            # Singola riga {bg} (o fallback sequenziale)
            for pos_r, cur_bi in enumerate(bg_run):
                li = lines_info[cur_bi]
                if pos_r > 0 and final_lines[bg_run[pos_r - 1]]:
                    p_end = final_lines[bg_run[pos_r - 1]][-1]["end"]
                    p_words = final_lines[bg_run[pos_r - 1]]
                bg_phons = set(_clean_phonetic_ctc(w, lang) for w in li["words"] if _clean_phonetic_ctc(w, lang))
                echo_after = max(0.0, p_end - 0.10)
                if p_words and len(bg_run) == 1:
                    for pw in p_words:
                        if _clean_phonetic_ctc(pw["word"], lang) in bg_phons:
                            echo_after = max(echo_after, pw["end"] - 0.02)

                gap_s = max(0.0, p_end - 0.02) if len(bg_run) == 1 else max(0.0, p_end + 0.01)
                gap_e = max(gap_s, n_start - 0.04)
                gap_adlib = acoustic_engine.find_gap_adlib_span(gap_s, gap_e) if (gap_e - gap_s >= 0.25 and (not li["nonlex"] or (gap_e - gap_s) >= 0.80)) else None
                if gap_adlib and len(li["words"]) == 1:
                    final_lines[cur_bi] = [{"word": li["words"][0], "start": gap_adlib[0], "end": gap_adlib[1], "score": 0.90}]
                    continue

                if gap_adlib and (gap_adlib[1] - gap_adlib[0]) >= 0.30:
                    ws = max(gap_s, gap_adlib[0] - 0.12)
                    we = min(n_start - 0.03, max(ws + 0.35, gap_adlib[1] + 0.05))
                    if we <= ws + 0.30:
                        we = min(total_audio_sec, max(ws + 0.35, n_start - 0.02))
                    res_bg = _align_block_ctc([cur_bi], ws, we, prefer_local=True)
                    bg_max_end = min(n_start - 0.02, we)
                else:
                    ws = echo_after if len(bg_run) == 1 else max(p_end + 0.01, echo_after)
                    if (n_start - p_end) >= 1.50:
                        we = min(total_audio_sec, max(ws + 0.50, n_start - 0.35))
                        bg_max_end = we
                    elif (n_start - p_end) >= 0.65:
                        we = min(total_audio_sec, max(ws + 0.50, n_start - 0.03))
                        bg_max_end = we
                    else:
                        we = min(total_audio_sec, max(ws + 0.70, n_start + 0.35))
                        bg_max_end = max(we, n_start + 0.35)
                    res_bg = _align_block_ctc([cur_bi], ws, we)
                if not res_bg or cur_bi not in res_bg:
                    ws_f = min(total_audio_sec - 0.4, max(ws, p_end + 0.02))
                    we_f = max(ws_f + 0.35, min(total_audio_sec, max(ws_f + 0.35, n_start - 0.02)))
                    dt = (we_f - ws_f) / max(1, len(li["words"]))
                    final_lines[cur_bi] = [{"word": w, "start": round(ws_f + k * dt, 3), "end": round(ws_f + (k + 1) * dt, 3), "score": 0.0} for k, w in enumerate(li["words"])]
                else:
                    min_bg_s = max(p_end + 0.01, ws) if (len(bg_run) > 1 or ws >= p_end - 0.02) else max(0.0, ws)
                    ref_s = _refine_block_words(
                        res_bg, [cur_bi], acoustic_engine, ws,
                        ("bg_single", ws, we, ws, we),
                        max(0.0, min_bg_s - 0.015), bg_max_end, False, False,
                    )
                    final_lines[cur_bi] = ref_s[cur_bi]
            li_idx = bg_run_end

        # PASS 3: Ricostruzione & allineamento degli ad-lib in linea (...) senza MAI spostare l'attacco della riga successiva
        def _align_inline_adlib_span(ad_words: list, ws_a: float, we_a: float):
            if not ad_words:
                return []
            cand_l = [[]]
            w_f = []
            for wi_a, aw in enumerate(ad_words):
                vars_a = word_variants(aw)
                if vars_a is None:
                    w_f.append((aw, False, wi_a))
                else:
                    w_f.append((aw, True, wi_a))
                    nxt_c = []
                    for cl in cand_l:
                        for vw in vars_a:
                            nxt_c.append(cl + [(len(vw), vw)])
                            if len(nxt_c) >= 4:
                                break
                        if len(nxt_c) >= 4:
                            break
                    cand_l = nxt_c
            if not cand_l or not cand_l[0]:
                return None
            wm = [(0, w_f)]
            f0_a = max(0, int(np.floor(max(0.0, ws_a - 0.04) / fd)))
            f1_a = min(emission.shape[0], int(np.ceil(we_a / fd)))
            out_a = _run_ctc_on_emission(emission[f0_a:f1_a], f0_a * fd, ws_a, cand_l, wm) if f1_a > f0_a + 4 else None
            sc_a = float(np.mean([x["score"] for x in out_a[0]])) if (out_a and 0 in out_a) else -1.0
            if sc_a < 0.16 and mms_model is not None and len(v_wav_full) > 0 and (we_a - ws_a) >= 0.25:
                loc_a = _run_ctc_on_waveform_chunks(v_wav_full, max(0.0, ws_a - 0.04), we_a, cand_l, wm)
                if loc_a and 0 in loc_a:
                    sc_l = float(np.mean([x["score"] for x in loc_a[0]]))
                    if sc_l > sc_a:
                        out_a, sc_a = loc_a, sc_l
            if out_a and 0 in out_a and sc_a >= 0.18:
                wl_a = out_a[0]
                max_d = max((x["end"] - x["start"]) for x in wl_a)
                if len(ad_words) >= 2 and max_d > 1.65:
                    return None
                if wl_a[0]["start"] < ws_a:
                    wl_a[0]["start"] = round(ws_a, 3)
                if wl_a[-1]["end"] > we_a:
                    wl_a[-1]["end"] = round(we_a, 3)
                w0_cap = max(ws_a + 0.025, wl_a[0]["end"] - 0.040) if wl_a[0]["end"] > ws_a + 0.045 else wl_a[0]["start"]
                wl_a[0]["start"] = acoustic_engine.snap_onset_micro(
                    wl_a[0]["start"], ws_a, max(ws_a + 0.025, min(w0_cap, wl_a[-1]["end"] - 0.035)), max_shift=0.025, max_shift_fwd=0.020
                )
                out_smooth = smooth_and_clamp_words(
                    [{"word": x["word"], "start": x["start"], "end": x["end"]} for x in wl_a],
                    min_word_dur=0.035,
                )
                if len(out_smooth) >= 2 and (out_smooth[0]["end"] - out_smooth[0]["start"]) < 0.038:
                    need_d = 0.040 - (out_smooth[0]["end"] - out_smooth[0]["start"])
                    back_room = max(0.0, out_smooth[0]["start"] - ws_a)
                    take_pre = min(back_room, need_d)
                    if take_pre > 0:
                        out_smooth[0]["start"] = round(out_smooth[0]["start"] - take_pre, 3)
                        need_d -= take_pre
                    if need_d > 0 and (out_smooth[1]["end"] - out_smooth[1]["start"]) > 0.075:
                        steal = min(need_d, (out_smooth[1]["end"] - out_smooth[1]["start"]) - 0.055)
                        out_smooth[0]["end"] = round(out_smooth[0]["end"] + steal, 3)
                        out_smooth[1]["start"] = out_smooth[0]["end"]
                return out_smooth
            return None

        for bi, li in enumerate(lines_info):
            if not li.get("has_inline_adlibs") or not final_lines[bi]:
                continue
            full_words = li["words"]
            mask = li["adlib_mask"]
            core_map = {}
            for cw in final_lines[bi]:
                owi = cw.get("orig_wi")
                if owi is not None:
                    core_map[owi] = cw
            if not core_map:
                continue

            prev_line_end = max((final_lines[p][-1]["end"] for p in range(bi - 1, -1, -1) if final_lines[p]), default=0.0)
            next_line_start = next((final_lines[n][0]["start"] for n in range(bi + 1, len(lines_info)) if final_lines[n]), total_audio_sec)

            rebuilt = [None] * len(full_words)
            for owi, cw in core_map.items():
                if 0 <= owi < len(full_words):
                    rebuilt[owi] = {"word": full_words[owi], "start": float(cw["start"]), "end": float(cw["end"])}

            wi = 0
            while wi < len(full_words):
                if rebuilt[wi] is not None:
                    wi += 1
                    continue
                wj = wi + 1
                while wj < len(full_words) and rebuilt[wj] is None:
                    wj += 1
                ad_slice = full_words[wi:wj]
                n_ad = len(ad_slice)

                prev_core_idx = next((p for p in range(wi - 1, -1, -1) if rebuilt[p] is not None), -1)
                next_core_idx = next((n for n in range(wj, len(full_words)) if rebuilt[n] is not None), -1)

                left_t = rebuilt[prev_core_idx]["end"] if prev_core_idx >= 0 else prev_line_end
                right_t = rebuilt[next_core_idx]["start"] if next_core_idx >= 0 else next_line_start
                right_t = max(right_t, left_t + 0.04)

                # Se l'ultima parola core è allungata perché ha assorbito l'ad-lib finale (es. "ehi (Ahah)"), recupera la separazione sul calo interno
                ad_hi_pre = min(right_t, next_line_start - 0.008)
                if prev_core_idx >= 0 and next_core_idx < 0:
                    prev_s_c = rebuilt[prev_core_idx]["start"]
                    prev_e_cur = rebuilt[prev_core_idx]["end"]
                    post_bursts = acoustic_engine.find_all_gap_bursts(left_t, ad_hi_pre, min_dur_ms=50, merge_gap_ms=45) if (ad_hi_pre - left_t) >= 0.16 else []
                    if (prev_e_cur - prev_s_c) >= 0.42 and ((ad_hi_pre - left_t) < 0.22 * n_ad or not post_bursts):
                        # Prima controlla se una raffica distinta alla fine di prev_core è già separata da silenzio/calo (es. L39 "ehi (Ahah)")
                        split_t = -1.0
                        in_bursts = acoustic_engine.find_all_gap_bursts(
                            prev_s_c + 0.14, min(ad_hi_pre, prev_e_cur + 0.15), min_dur_ms=50, merge_gap_ms=45
                        )
                        if in_bursts and in_bursts[-1][0] > prev_s_c + 0.18 and in_bursts[-1][0] < left_t - 0.08:
                            split_t = in_bursts[-1][0]
                            prev_e_cur = max(prev_e_cur, in_bursts[-1][1])
                        else:
                            # Raffica continua senza pausa lunga: ritaglia eventuale coda silenziosa e cerca calo interno nella parte attiva
                            active_cap = min(prev_e_cur, in_bursts[-1][1]) if in_bursts else prev_e_cur
                            cand_sp = acoustic_engine.find_onset_after_internal_dip(
                                prev_s_c, active_cap, min_tail_dur=0.07, min_dip_ms=12, min_span=0.38
                            )
                            if cand_sp > prev_s_c + 0.14 and cand_sp <= active_cap - 0.05:
                                split_t = cand_sp
                                prev_e_cur = max(prev_e_cur, active_cap)
                        if split_t > prev_s_c + 0.14 and split_t <= prev_e_cur - 0.05:
                            rebuilt[prev_core_idx]["end"] = round(split_t - 0.010, 3)
                            left_t = rebuilt[prev_core_idx]["end"]
                            right_t = max(prev_e_cur, left_t + 0.06)
                            ad_hi_pre = min(right_t, next_line_start - 0.008)

                # Garantisci almeno 50ms/parola senza mai invadere la riga successiva o il core precedente
                min_ad_span = min(0.28, 0.055 * n_ad)
                if right_t - left_t < min_ad_span and prev_core_idx >= 0:
                    room_back = rebuilt[prev_core_idx]["end"] - (rebuilt[prev_core_idx]["start"] + 0.055)
                    take_back = min(max(0.0, room_back), min_ad_span - (right_t - left_t))
                    if take_back > 0:
                        rebuilt[prev_core_idx]["end"] = round(rebuilt[prev_core_idx]["end"] - take_back, 3)
                        left_t = rebuilt[prev_core_idx]["end"]

                ctc_ad = None
                ad_hi_bound = min(right_t, next_line_start - 0.008)
                if (ad_hi_bound - left_t) >= 0.30 and left_t < next_line_start - 0.15:
                    ctc_ad = _align_inline_adlib_span(ad_slice, left_t, ad_hi_bound)

                if ctc_ad and len(ctc_ad) == n_ad:
                    placed = ctc_ad
                else:
                    bursts = acoustic_engine.find_all_gap_bursts(left_t, ad_hi_bound, min_dur_ms=50, merge_gap_ms=45) if (ad_hi_bound - left_t) >= 0.16 else []
                    if len(bursts) > n_ad and n_ad >= 2:
                        # Preferisci la finestra contigua di n_ad raffiche più staccata dal respiro di left_t o con span più compatto/posteriore
                        cand_wins = [bursts[idx_b:idx_b + n_ad] for idx_b in range(len(bursts) - n_ad + 1)]
                        best_win = min(
                            cand_wins,
                            key=lambda w: ((w[-1][1] - w[0][0]) + (0.45 if w[0][0] <= left_t + 0.42 and len(cand_wins) >= 2 else 0.0)),
                        )
                        bursts = best_win
                    if len(bursts) == n_ad:
                        placed = [
                            {"word": aw, "start": round(bs, 3), "end": round(be, 3)}
                            for aw, (bs, be) in zip(ad_slice, bursts)
                        ]
                    elif bursts:
                        if len(bursts) > n_ad and n_ad == 1:
                            bs, be = max(bursts, key=lambda b: b[1] - b[0])
                            placed = [{"word": ad_slice[0], "start": round(bs, 3), "end": round(be, 3)}]
                        else:
                            b_s = bursts[0][0]
                            b_e = bursts[-1][1]
                            ctc_in_b = _align_inline_adlib_span(ad_slice, max(left_t, b_s - 0.04), min(ad_hi_bound, b_e + 0.04)) if (b_e - b_s) >= 0.25 else None
                            if ctc_in_b and len(ctc_in_b) == n_ad:
                                placed = ctc_in_b
                            else:
                                # Distribuisci le parole sugli archi attivi delle raffiche rilevate proporzionalmente alla lunghezza
                                tot_w_ch = sum(max(1, len(re.sub(r"[^\w]", "", w))) for w in ad_slice)
                                span_b = max(0.055 * n_ad, b_e - b_s)
                                cur_t_b = b_s
                                placed = []
                                for kk, aw in enumerate(ad_slice):
                                    w_ch = max(1, len(re.sub(r"[^\w]", "", aw)))
                                    d_k = max(0.045, span_b * (w_ch / tot_w_ch)) if kk < n_ad - 1 else max(0.045, b_e - cur_t_b)
                                    en_k = b_e if kk == n_ad - 1 else min(b_e - 0.045 * (n_ad - 1 - kk), cur_t_b + d_k)
                                    placed.append({"word": aw, "start": round(cur_t_b, 3), "end": round(max(cur_t_b + 0.040, en_k), 3)})
                                    cur_t_b = placed[-1]["end"]
                    else:
                        pack_end = min(right_t, left_t + max(min_ad_span, min(0.36, 0.14 * n_ad)))
                        dt_a = max(0.030, (pack_end - left_t) / n_ad)
                        placed = [
                            {
                                "word": aw,
                                "start": round(left_t + kk * dt_a, 3),
                                "end": round(min(right_t, left_t + (kk + 1) * dt_a), 3),
                            }
                            for kk, aw in enumerate(ad_slice)
                        ]
                for kk, p_tok in enumerate(placed):
                    rebuilt[wi + kk] = p_tok
                wi = wj

            final_lines[bi] = smooth_and_clamp_words(
                [x for x in rebuilt if x is not None],
                min_word_dur=0.035,
            )

        if progress_cb:
            progress_cb(92)

        return [
            {
                "text": li["text"],
                "agent": li["agent"],
                "is_bg": li["is_bg"],
                "words": [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in final_lines[i]],
            }
            for i, li in enumerate(lines_info)
        ]


def clean_token_str(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"[^\w]", "", str(s)).lower()


def normalize_elongated_str(s: str) -> str:
    c = clean_token_str(s)
    if not c:
        return ""
    return re.sub(r"(.)\1{2,}", r"\1", c)


def compute_phonetic_similarity(exp_clean: str, tok_clean: str) -> float:
    if not exp_clean or not tok_clean:
        return 0.0
    if exp_clean == tok_clean:
        return 1.0
    exp_norm = normalize_elongated_str(exp_clean)
    tok_norm = normalize_elongated_str(tok_clean)
    if exp_norm and exp_norm == tok_norm:
        return 0.98
    r1 = SequenceMatcher(None, exp_clean, tok_clean).ratio()
    r2 = SequenceMatcher(None, exp_norm, tok_norm).ratio()
    return max(r1, r2 * 0.97)


def reconstruct_line_words(seg_words, line_text, offset_sec=0.0, max_audio_sec=9999.0, return_consumed=False):
    expected_words = line_text.split()
    if not expected_words:
        return ([], 0) if return_consumed else []

    tokens = []
    if seg_words:
        for w in seg_words:
            txt = getattr(w, "word", "")
            if not txt and isinstance(w, dict):
                txt = w.get("word", w.get("text", w.get("orig", "")))
            txt = str(txt).strip()
            if txt:
                raw_s = getattr(w, "start", None) if not isinstance(w, dict) else w.get("start")
                raw_e = getattr(w, "end", None) if not isinstance(w, dict) else w.get("end")
                w_s = max(0.0, min(max_audio_sec - 0.05, float(raw_s) + offset_sec))
                w_e = max(w_s + 0.04, min(max_audio_sec, float(raw_e) + offset_sec))
                tokens.append({
                    "orig": txt,
                    "clean": clean_token_str(txt),
                    "start": w_s,
                    "end": w_e,
                })

    if not tokens:
        return ([], 0) if return_consumed else []

    N = len(expected_words)
    M = len(tokens)
    exp_cleans = [clean_token_str(w) or w.lower() for w in expected_words]

    # Abilita il risolutore DP 1_to_k / 2_to_1 in entrambe le modalità (1:1 e globale)
    NEG_INF = -1e9
    dp = [[NEG_INF] * (M + 1) for _ in range(N + 1)]
    back = [[None] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 0.0

    for i in range(N):
        for j in range(M):
            if dp[i][j] <= NEG_INF / 2:
                continue

            acc_clean = ""
            gap_penalty = 0.0
            for k in range(1, min(6, M - j + 1)):
                t_cur = tokens[j + k - 1]
                acc_clean += t_cur["clean"]
                if k > 1:
                    prev_t = tokens[j + k - 2]
                    inter_gap = max(0.0, t_cur["start"] - prev_t["end"])
                    if inter_gap > 0.25:
                        gap_penalty += inter_gap * 2.2

                sim = compute_phonetic_similarity(exp_cleans[i], acc_clean)
                step_score = sim - 0.04 * (k - 1) - gap_penalty
                cand = dp[i][j] + step_score
                if cand > dp[i + 1][j + k]:
                    dp[i + 1][j + k] = cand
                    back[i + 1][j + k] = (i, j, "1_to_k", k)

            if i + 1 < N:
                combined_exp = exp_cleans[i] + exp_cleans[i + 1]
                sim2 = compute_phonetic_similarity(combined_exp, tokens[j]["clean"])
                cand2 = dp[i][j] + (sim2 * 1.85) - 0.12
                if cand2 > dp[i + 2][j + 1]:
                    dp[i + 2][j + 1] = cand2
                    back[i + 2][j + 1] = (i, j, "2_to_1", 1)

    best_j = M
    best_val = dp[N][M]
    for j_end in range(max(1, M - 2), M + 1):
        val = dp[N][j_end] - (0.0 if return_consumed else 0.08 * (M - j_end))
        if val > best_val:
            best_val = val
            best_j = j_end

    if best_val > NEG_INF / 2:
        ops = []
        curr_i, curr_j = N, best_j
        while curr_i > 0 and curr_j >= 0:
            step = back[curr_i][curr_j]
            if not step:
                break
            prev_i, prev_j, op_type, k_tok = step
            ops.append((prev_i, curr_i, prev_j, curr_j, op_type))
            curr_i, curr_j = prev_i, prev_j

        if curr_i == 0:
            ops.reverse()
            matched = []
            for (i_s, i_e, j_s, j_e, op_type) in ops:
                if op_type == "1_to_k":
                    w_start = tokens[j_s]["start"]
                    w_end = tokens[j_e - 1]["end"]
                    matched.append({
                        "word": expected_words[i_s],
                        "start": round(w_start, 3),
                        "end": round(max(w_start + 0.04, w_end), 3),
                    })
                elif op_type == "2_to_1":
                    t_start = tokens[j_s]["start"]
                    t_end = max(t_start + 0.08, tokens[j_s]["end"])
                    dur = t_end - t_start
                    w1 = expected_words[i_s]
                    w2 = expected_words[i_s + 1]
                    l1 = max(1, len(exp_cleans[i_s]))
                    l2 = max(1, len(exp_cleans[i_s + 1]))
                    ratio = max(0.25, min(0.75, l1 / float(l1 + l2)))
                    mid_t = round(t_start + dur * ratio, 3)
                    matched.append({
                        "word": w1,
                        "start": round(t_start, 3),
                        "end": max(round(t_start + 0.04, 3), mid_t),
                    })
                    matched.append({
                        "word": w2,
                        "start": mid_t,
                        "end": round(max(mid_t + 0.04, t_end), 3),
                    })
            if len(matched) == N:
                return (matched, best_j) if return_consumed else matched

    matched = []
    tok_idx = 0
    num_tok = len(tokens)

    for exp_idx, exp_w in enumerate(expected_words):
        exp_clean = exp_cleans[exp_idx]
        remaining_words = N - exp_idx - 1

        if tok_idx >= num_tok:
            last_end = matched[-1]["end"] if matched else tokens[-1]["end"]
            e_calc = min(max_audio_sec, last_end + 0.18)
            matched.append({
                "word": exp_w,
                "start": round(last_end, 3),
                "end": round(e_calc, 3),
            })
            continue

        best_sim = -1.0
        best_end_idx = tok_idx
        acc_text = ""

        max_look = min(tok_idx + 5, num_tok)
        if not return_consumed and (num_tok - tok_idx) > remaining_words:
            max_look = min(max_look, num_tok - remaining_words)
            max_look = max(tok_idx + 1, max_look)

        for search_end in range(tok_idx, max_look):
            if search_end > tok_idx:
                gap = tokens[search_end]["start"] - tokens[search_end - 1]["end"]
                if gap > 0.35:
                    break
            acc_text += tokens[search_end]["clean"]
            sim = compute_phonetic_similarity(exp_clean, acc_text)
            if sim > best_sim:
                best_sim = sim
                best_end_idx = search_end
            if acc_text == exp_clean or (len(acc_text) >= len(exp_clean) and sim > 0.88):
                break

        w_start = tokens[tok_idx]["start"]
        w_end = tokens[best_end_idx]["end"]
        tok_idx = best_end_idx + 1

        matched.append({
            "word": exp_w,
            "start": round(w_start, 3),
            "end": round(max(w_start + 0.04, w_end), 3),
        })

    if return_consumed:
        return matched, tok_idx
    return matched


def smooth_and_clamp_words(words_list, min_word_dur=0.04):
    if not words_list:
        return words_list

    for k in range(len(words_list)):
        if k > 0:
            gap = words_list[k]["start"] - words_list[k - 1]["end"]
            has_punct = bool(re.search(r"[,.;:!?…\)]$", str(words_list[k - 1].get("word", "")).strip()))
            if gap < 0:
                words_list[k - 1]["end"] = words_list[k]["start"]
                if words_list[k - 1]["end"] <= words_list[k - 1]["start"]:
                    words_list[k - 1]["end"] = round(words_list[k - 1]["start"] + min_word_dur, 3)
                    words_list[k]["start"] = words_list[k - 1]["end"]
            elif gap < (0.018 if has_punct else 0.032):
                words_list[k - 1]["end"] = words_list[k]["start"]

        if words_list[k]["end"] <= words_list[k]["start"]:
            words_list[k]["end"] = round(words_list[k]["start"] + min_word_dur, 3)

    return words_list


# ==============================================================================
# FORMATTAZIONE TEMPI & OUTPUT
# ==============================================================================
def format_lrc_time(seconds: float) -> str:
    total_cs = int(round(max(0.0, float(seconds)) * 100.0))
    minutes, rem_cs = divmod(total_cs, 6000)
    secs, cs = divmod(rem_cs, 100)
    return f"[{minutes:02d}:{secs:02d}.{cs:02d}]"


def format_ttml_time(seconds: float) -> str:
    total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
    minutes, rem_ms = divmod(total_ms, 60000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def format_word_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    rounded = round(seconds, 3)
    s = f"{rounded:.3f}".rstrip("0").rstrip(".")
    return s if "." in s else f"{s}.0"


def format_lrc_for_lrclib(raw_lrc: str) -> str:
    clean_lines = []
    for line in raw_lrc.splitlines():
        line = line.strip()
        if not line or (line.startswith("<") and line.endswith(">")) or LRC_META_TAG_RE.match(line):
            continue

        m = re.match(r"^(\[\d{1,3}:\d{2}(?:\.\d+)?\])\s*(.*)$", line)
        if m:
            ts = m.group(1)
            content = m.group(2)
            content = re.sub(r"\{(?:agent:)?v\d+\}", "", content, flags=re.IGNORECASE)
            is_bg = "{bg}" in content.lower()
            content = re.sub(r"\{bg\}", "", content, flags=re.IGNORECASE).strip()

            if is_bg and not (content.startswith("(") and content.endswith(")")):
                content = f"({content})"

            if content:
                clean_lines.append(f"{ts} {content}")
            else:
                clean_lines.append(ts)
        else:
            if line.startswith("[") and "]" in line:
                clean_lines.append(line)

    return "\n".join(clean_lines)


def format_plain_for_lrclib(plain_lyrics: str, synced_lrc: str = "") -> str:
    source = plain_lyrics if plain_lyrics and plain_lyrics.strip() else synced_lrc
    clean_lines = []
    for line in source.splitlines():
        line = line.strip()
        if not line or (line.startswith("<") and line.endswith(">")) or LRC_META_TAG_RE.match(line):
            continue
        line = re.sub(r"^\[\d{1,3}:\d{2}(?:\.\d+)?\]\s*", "", line)
        line = re.sub(r"[\[\{](?:agent:)?v\d+[\]\}]", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\{bg\}", "", line, flags=re.IGNORECASE).strip()
        if line:
            clean_lines.append(line)
    return "\n".join(clean_lines)


def parse_lrc_to_structured_lines(lrc_text: str):
    lines = lrc_text.splitlines()
    structured = []
    inline_ts_re = re.compile(r"<(\d{1,3}:\d{2}(?:\.\d+)?)>")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("<"):
            continue

        m = re.match(r"^\[(\d{1,3}):(\d{2}(?:\.\d+)?)\]\s*(.*)$", line)
        if not m:
            continue

        min_val = int(m.group(1))
        sec_val = float(m.group(2))
        start_ms = int(round((min_val * 60 + sec_val) * 1000))
        rest = m.group(3).strip()

        agent_m = re.search(r"\{agent:(v\d+)\}", rest, re.IGNORECASE)
        agent = agent_m.group(1).lower() if agent_m else "v1"

        is_bg = "{bg}" in rest.lower() or (rest.startswith("(") and rest.endswith(")"))
        clean_text = re.sub(r"\{(?:agent:)?v\d+\}", "", rest, flags=re.IGNORECASE)
        clean_text = re.sub(r"\{bg\}", "", clean_text, flags=re.IGNORECASE).strip()
        if not clean_text:
            continue

        words = []
        end_ms = start_ms + 2500
        has_inline_elrc = False
        if i < len(lines) and lines[i].strip().startswith("<") and lines[i].strip().endswith(">"):
            w_line = lines[i].strip()[1:-1]
            i += 1
            tokens = w_line.split("|")
            for tok in tokens:
                parts = tok.split(":")
                if len(parts) >= 3:
                    w_txt = ":".join(parts[:-2]).strip()
                    try:
                        w_start_s = float(parts[-2])
                        w_end_s = float(parts[-1])
                        words.append({
                            "word": w_txt,
                            "start_ms": int(round(w_start_s * 1000)),
                            "end_ms": int(round(w_end_s * 1000)),
                        })
                    except Exception:
                        pass

            for k in range(len(words)):
                if k > 0 and words[k]["start_ms"] < words[k - 1]["end_ms"]:
                    words[k - 1]["end_ms"] = words[k]["start_ms"]
                    if words[k - 1]["end_ms"] <= words[k - 1]["start_ms"]:
                        words[k - 1]["end_ms"] = words[k]["start_ms"] + 70
                        words[k]["start_ms"] = words[k - 1]["end_ms"]
                if words[k]["end_ms"] <= words[k]["start_ms"]:
                    words[k]["end_ms"] = words[k]["start_ms"] + 70

            if words:
                end_ms = words[-1]["end_ms"]
        elif inline_ts_re.search(clean_text):
            # Enhanced LRC inline tags: <mm:ss.xx>word1 <mm:ss.xx>word2 ...
            matches = list(inline_ts_re.finditer(clean_text))
            chunks = []
            if matches[0].start() > 0:
                prefix_txt = clean_text[:matches[0].start()].strip()
                if prefix_txt:
                    chunks.append((start_ms, prefix_txt))
            for idx_m, mm in enumerate(matches):
                try:
                    t_ms = int(round(parse_time_to_seconds(mm.group(1)) * 1000))
                except Exception:
                    t_ms = start_ms
                seg_start_idx = mm.end()
                seg_end_idx = matches[idx_m + 1].start() if idx_m + 1 < len(matches) else len(clean_text)
                seg_txt = clean_text[seg_start_idx:seg_end_idx].strip()
                if seg_txt:
                    chunks.append((max(0, t_ms), seg_txt))

            for c_idx, (c_start, c_txt) in enumerate(chunks):
                c_next = chunks[c_idx + 1][0] if c_idx + 1 < len(chunks) else (c_start + 700)
                c_next = max(c_start + 60, c_next)
                sub_words = c_txt.split()
                if len(sub_words) == 1:
                    words.append({"word": sub_words[0], "start_ms": c_start, "end_ms": c_next})
                elif sub_words:
                    span = c_next - c_start
                    weights = [max(1, len(sw)) for sw in sub_words]
                    tot_w = float(sum(weights) or 1)
                    cur = c_start
                    for sw_idx, sw in enumerate(sub_words):
                        if sw_idx == len(sub_words) - 1:
                            w_e = c_next
                        else:
                            w_e = min(c_next - (len(sub_words) - 1 - sw_idx) * 40, max(cur + 40, cur + int(round(span * weights[sw_idx] / tot_w))))
                        words.append({"word": sw, "start_ms": cur, "end_ms": max(cur + 40, w_e)})
                        cur = words[-1]["end_ms"]

            if words:
                has_inline_elrc = True
                clean_text = " ".join(w["word"] for w in words).strip()
                start_ms = min(start_ms, words[0]["start_ms"])
                end_ms = words[-1]["end_ms"]

        structured.append({
            "text": clean_text,
            "agent": agent,
            "is_bg": is_bg,
            "start_ms": start_ms,
            "end_ms": max(start_ms + 80, end_ms),
            "words": words,
            "_inline_elrc": has_inline_elrc,
        })

    # Post-pass: evita sovrapposizioni su LRC solo-riga o ultimo token Enhanced LRC
    for idx, ln in enumerate(structured):
        next_s = structured[idx + 1]["start_ms"] if idx + 1 < len(structured) else None
        is_inline = ln.pop("_inline_elrc", False)
        if not ln["words"]:
            if next_s is not None and next_s > ln["start_ms"] + 120:
                ln["end_ms"] = max(ln["start_ms"] + 100, min(next_s - 20, ln["start_ms"] + 6000))
            else:
                ln["end_ms"] = ln["start_ms"] + 3500
        elif is_inline and ln["words"]:
            last_w = ln["words"][-1]
            if next_s is not None and next_s > last_w["start_ms"] + 80:
                last_w["end_ms"] = min(next_s - 20, last_w["start_ms"] + 1200)
                ln["end_ms"] = max(ln["start_ms"] + 80, last_w["end_ms"])

    return structured


def parse_ttml_to_structured_lines(ttml_text: str):
    structured = []
    p_pattern = re.compile(r"<p\b([^>]*)>(.*?)</p>", re.DOTALL | re.IGNORECASE)
    span_pattern = re.compile(r"<span\b([^>]*)>(.*?)</span>", re.DOTALL | re.IGNORECASE)

    matches = p_pattern.findall(ttml_text)
    for attrs, inner_html in matches:
        b_m = re.search(r'begin="([^"]+)"', attrs, re.IGNORECASE)
        e_m = re.search(r'end="([^"]+)"', attrs, re.IGNORECASE)
        if not b_m or not e_m:
            continue

        try:
            start_ms = int(round(parse_time_to_seconds(b_m.group(1)) * 1000))
            end_ms = int(round(parse_time_to_seconds(e_m.group(1)) * 1000))
        except Exception:
            continue

        ag_m = re.search(r'ttm:agent="([^"]+)"', attrs, re.IGNORECASE)
        agent = ag_m.group(1) if ag_m else "v1"

        ro_m = re.search(r'ttm:role="([^"]+)"', attrs, re.IGNORECASE)
        is_bg = bool(ro_m and "x-bg" in ro_m.group(1).lower())

        words = []
        span_matches = span_pattern.findall(inner_html)
        line_tokens = []

        for s_attrs, w_raw in span_matches:
            sb_m = re.search(r'begin="([^"]+)"', s_attrs, re.IGNORECASE)
            se_m = re.search(r'end="([^"]+)"', s_attrs, re.IGNORECASE)
            if not sb_m or not se_m:
                continue
            try:
                w_start = int(round(parse_time_to_seconds(sb_m.group(1)) * 1000))
                w_end = int(round(parse_time_to_seconds(se_m.group(1)) * 1000))
                clean_w = saxutils.unescape(w_raw).strip()
                if clean_w:
                    words.append({"word": clean_w, "start_ms": w_start, "end_ms": w_end})
                    line_tokens.append(clean_w)
            except Exception:
                continue

        for k in range(len(words)):
            if k > 0 and words[k]["start_ms"] < words[k - 1]["end_ms"]:
                words[k - 1]["end_ms"] = words[k]["start_ms"]
                if words[k - 1]["end_ms"] <= words[k - 1]["start_ms"]:
                    words[k - 1]["end_ms"] = words[k]["start_ms"] + 70
                    words[k]["start_ms"] = words[k - 1]["end_ms"]
            if words[k]["end_ms"] <= words[k]["start_ms"]:
                words[k]["end_ms"] = words[k]["start_ms"] + 70

        line_text = " ".join(line_tokens) if line_tokens else re.sub(r"<[^>]+>", "", inner_html).strip()
        if not line_text:
            continue

        if is_bg and not (line_text.startswith("(") and line_text.endswith(")")):
            line_text = f"({line_text})"

        structured.append({
            "text": line_text,
            "agent": agent,
            "is_bg": is_bg,
            "start_ms": start_ms,
            "end_ms": max(start_ms + 80, end_ms),
            "words": words,
        })

    return structured


def build_lyricsfile_yaml(title, artist, album, duration_ms, language, structured_lines, plain_text, songwriters="", notes=""):
    yaml_lines = [
        'version: "1.0"',
        'metadata:',
        f'  title: {json.dumps(title.strip(), ensure_ascii=False)}',
        f'  artist: {json.dumps(artist.strip(), ensure_ascii=False)}',
    ]
    if album and album.strip():
        yaml_lines.append(f'  album: {json.dumps(album.strip(), ensure_ascii=False)}')
    if duration_ms > 0:
        yaml_lines.append(f'  duration_ms: {int(round(duration_ms))}')
    if language and language not in ("auto", "🌐 Auto / Multilingua (Bilingue)"):
        yaml_lines.append(f'  language: {json.dumps(language, ensure_ascii=False)}')
    sw_list = _split_songwriters(songwriters)
    if sw_list:
        yaml_lines.append('  songwriters:')
        for sw in sw_list:
            yaml_lines.append(f'    - {json.dumps(sw, ensure_ascii=False)}')
    if notes and notes.strip():
        yaml_lines.append(f'  notes: {json.dumps(notes.strip(), ensure_ascii=False)}')

    yaml_lines.append('lines:')
    for line in structured_lines:
        line_text = line["text"].strip()
        yaml_lines.append(f'  - text: {json.dumps(line_text, ensure_ascii=False)}')
        yaml_lines.append(f'    start_ms: {int(line["start_ms"])}')
        yaml_lines.append(f'    end_ms: {int(line["end_ms"])}')

        words = line.get("words", [])
        if words:
            yaml_lines.append('    words:')
            for idx, w in enumerate(words):
                w_str = w["word"].strip()
                if idx < len(words) - 1:
                    w_str += " "
                yaml_lines.append(f'      - text: {json.dumps(w_str, ensure_ascii=False)}')
                yaml_lines.append(f'        start_ms: {int(w["start_ms"])}')
                yaml_lines.append(f'        end_ms: {int(w["end_ms"])}')

    yaml_lines.append('plain: |')
    for p_line in plain_text.splitlines():
        if p_line.strip():
            yaml_lines.append(f'  {p_line.strip()}')
        else:
            yaml_lines.append('')

    return "\n".join(yaml_lines) + "\n"


def build_ttml(title, artist, album, duration_seconds, language, structured_lines, songwriters="", notes=""):
    lang_code = _normalize_lang_code(language) or "it"
    dur_str = format_ttml_time(duration_seconds)
    has_word_timing = any(bool(ln.get("words")) for ln in (structured_lines or []))
    ttml_timing = "Word" if has_word_timing else "Line"

    xml_lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<tt xmlns="http://www.w3.org/ns/ttml"',
        '    xmlns:ttm="http://www.w3.org/ns/ttml#metadata"',
        '    xmlns:itunes="http://music.apple.com/lyric-ttml-internal"',
        f'    itunes:timing="{ttml_timing}"',
        f'    xml:lang="{saxutils.escape(lang_code)}">',
        '  <head>',
        '    <metadata>',
        f'      <ttm:title>{saxutils.escape(title.strip())}</ttm:title>',
    ]
    if notes and notes.strip():
        xml_lines.append(f'      <ttm:desc>{saxutils.escape(notes.strip())}</ttm:desc>')
    xml_lines.extend([
        '      <ttm:agent type="person" xml:id="v1"/>',
        '      <ttm:agent type="other" xml:id="v2"/>',
    ])
    sw_list = _split_songwriters(songwriters)
    if sw_list:
        xml_lines.append('      <iTunesMetadata xmlns="http://music.apple.com/lyric-ttml-internal">')
        xml_lines.append('        <songwriters>')
        for sw in sw_list:
            xml_lines.append(f'          <songwriter>{saxutils.escape(sw)}</songwriter>')
        xml_lines.append('        </songwriters>')
        xml_lines.append('      </iTunesMetadata>')

    xml_lines.extend([
        '    </metadata>',
        '  </head>',
        f'  <body dur="{dur_str}">',
        '    <div>',
    ])

    for line in structured_lines:
        start_t = format_ttml_time(line["start_ms"] / 1000.0)
        end_t = format_ttml_time(line["end_ms"] / 1000.0)
        agent = line.get("agent", "v1")
        is_bg = line.get("is_bg", False)
        bg_attr = ' ttm:role="x-bg"' if is_bg else ''

        words = line.get("words", [])
        if words:
            spans = []
            for w in words:
                w_start = format_ttml_time(w["start_ms"] / 1000.0)
                w_end = format_ttml_time(w["end_ms"] / 1000.0)
                w_text = saxutils.escape(w["word"].strip())
                if w_text:
                    spans.append(f'<span begin="{w_start}" end="{w_end}">{w_text}</span>')

            spans_str = " ".join(spans)
            xml_lines.append(f'      <p begin="{start_t}" end="{end_t}" ttm:agent="{agent}"{bg_attr}>{spans_str}</p>')
        else:
            txt = saxutils.escape(line["text"].strip())
            xml_lines.append(f'      <p begin="{start_t}" end="{end_t}" ttm:agent="{agent}"{bg_attr}>{txt}</p>')

    xml_lines.append('    </div>')
    xml_lines.append('  </body>')
    xml_lines.append('</tt>')

    return "\n".join(xml_lines) + "\n"


def format_lrc_metadata_headers(title="", artist="", album="", duration_seconds=0, songwriters="", notes="") -> list:
    headers = []
    if title and title.strip():
        headers.append(f"[ti:{title.strip()}]")
    if artist and artist.strip():
        headers.append(f"[ar:{artist.strip()}]")
    if album and album.strip():
        headers.append(f"[al:{album.strip()}]")
    sw_list = _split_songwriters(songwriters)
    if sw_list:
        headers.append(f"[au:{', '.join(sw_list)}]")
    if duration_seconds and int(round(duration_seconds)) > 0:
        mins, secs = divmod(int(round(duration_seconds)), 60)
        headers.append(f"[length:{mins:02d}:{secs:02d}]")
    headers.append("[re:MetroSync]")
    if notes and notes.strip():
        clean_note = " | ".join(l.strip() for l in notes.splitlines() if l.strip())
        if clean_note:
            headers.append(f"[# Note: {clean_note}]")
    return headers


OUTRO_CREDIT_TEXT_RE = re.compile(r"^\s*Testo:\s+\S+", re.IGNORECASE)
_OUTRO_CREDIT_LRC_RE = re.compile(
    r"^(?:\[\d{1,3}:\d{2}(?:\.\d+)?\])?\s*(?:\{(?:agent:)?v\d+\}|\{bg\})?\s*(.+)$",
    re.IGNORECASE,
)


def _normalize_outro_candidates(extra_texts=None) -> set:
    out = set()
    for t in (extra_texts or ()):
        if t:
            norm = " ".join(str(t).split()).strip().lower()
            if norm:
                out.add(norm)
    return out


def _is_outro_credit_text(text: str, known_set: set = None) -> bool:
    norm = " ".join(str(text or "").split()).strip()
    if not norm:
        return False
    if OUTRO_CREDIT_TEXT_RE.match(norm):
        return True
    return bool(known_set and norm.lower() in known_set)


def _format_structured_line_to_lrc_pair(line: dict) -> list:
    start_ts = format_lrc_time(line["start_ms"] / 1000.0)
    tag_prefix = "{bg}" if line.get("is_bg") else f"{{agent:{line.get('agent', 'v1')}}}"
    pair = [f"{start_ts}{tag_prefix}{line['text']}"]
    words = line.get("words", [])
    if words:
        w_tokens = [
            f"{w['word']}:{format_word_time(w['start_ms'] / 1000.0)}:{format_word_time(w['end_ms'] / 1000.0)}"
            for w in words
        ]
        pair.append("<" + "|".join(w_tokens) + ">")
    return pair


def strip_outro_credits_from_lrc_lines(lines: list, extra_texts=None) -> list:
    known_set = _normalize_outro_candidates(extra_texts)

    def _is_credit_lrc_line(raw_line: str) -> bool:
        m = _OUTRO_CREDIT_LRC_RE.match(raw_line.strip())
        if not m:
            return False
        body = m.group(1).strip()
        if raw_line.strip().startswith("[") and re.fullmatch(r"\[\d{1,3}:\d{2}(?:\.\d+)?\]", raw_line.strip()):
            return False
        return _is_outro_credit_text(body, known_set)

    cleaned = list(lines)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    while cleaned:
        last_s = cleaned[-1].strip()
        if (
            last_s.startswith("<")
            and last_s.endswith(">")
            and len(cleaned) >= 2
            and _is_credit_lrc_line(cleaned[-2])
        ):
            cleaned.pop()
            cleaned.pop()
            while cleaned and not cleaned[-1].strip():
                cleaned.pop()
            continue
        if _is_credit_lrc_line(last_s):
            cleaned.pop()
            while cleaned and not cleaned[-1].strip():
                cleaned.pop()
            continue
        break
    return cleaned


def strip_outro_credits_from_plain(plain_text: str, extra_texts=None) -> str:
    if not plain_text:
        return ""
    known_set = _normalize_outro_candidates(extra_texts)
    lines = plain_text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and _is_outro_credit_text(lines[-1], known_set):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return "\n".join(lines).strip()


def apply_outro_credits_to_structured(
    structured_lines: list,
    songwriters: str,
    duration_sec: float,
    enabled: bool = True,
    outro_text: str = "",
    notes: str = "",
    known_outro_texts=None,
):
    """Appende (o rimuove in modo idempotente) una riga finale personalizzata o 'Testo: <Autori>' entro la durata audio."""
    known_set = _normalize_outro_candidates(list(known_outro_texts or ()) + [outro_text])
    clean_lines = list(structured_lines or [])
    while clean_lines and (
        clean_lines[-1].get("is_outro_credits")
        or _is_outro_credit_text(str(clean_lines[-1].get("text", "")), known_set)
    ):
        clean_lines.pop()

    if not enabled or not clean_lines or duration_sec <= 0:
        return clean_lines, None

    custom_clean = " ".join(str(outro_text or "").splitlines()).strip()
    if custom_clean:
        credit_text = custom_clean
    else:
        sw_list = _split_songwriters(songwriters)
        if sw_list:
            credit_text = f"Testo: {', '.join(sw_list)}"
        elif (notes or "").strip():
            credit_text = " ".join(str(notes).splitlines()).strip()
        else:
            credit_text = ""

    if not credit_text:
        return clean_lines, None

    last_end_ms = max(int(ln.get("end_ms") or ln.get("start_ms") or 0) for ln in clean_lines)
    limit_ms = int(round(float(duration_sec) * 1000.0)) - 120
    avail_ms = limit_ms - last_end_ms
    if avail_ms < 900:
        return clean_lines, None

    gap_ms = min(700, max(250, avail_ms // 4))
    start_ms = last_end_ms + gap_ms
    end_ms = min(limit_ms, start_ms + 7500)
    total_span_ms = max(600, end_ms - start_ms)

    tokens = credit_text.split()
    weights = [max(2, len(t)) for t in tokens]
    weight_sum = float(sum(weights) or 1)

    words = []
    cursor_ms = start_ms
    for idx, tok in enumerate(tokens):
        if idx == len(tokens) - 1:
            w_end = end_ms
        else:
            share = int(round(total_span_ms * (weights[idx] / weight_sum)))
            w_end = min(end_ms - (len(tokens) - 1 - idx) * 60, max(cursor_ms + 60, cursor_ms + share))
        words.append({
            "word": tok,
            "start_ms": int(cursor_ms),
            "end_ms": int(max(cursor_ms + 40, w_end)),
        })
        cursor_ms = words[-1]["end_ms"]

    credit_entry = {
        "text": credit_text,
        "agent": "v1",
        "is_bg": False,
        "is_outro_credits": True,
        "start_ms": int(start_ms),
        "end_ms": int(words[-1]["end_ms"]) if words else int(end_ms),
        "words": words,
    }
    return clean_lines + [credit_entry], credit_entry


def convert_structured_to_lrc(structured_lines, title="", artist="", album="", duration_seconds=0, songwriters="", notes="", include_headers=True, formato="metrolist"):
    output_lines = []
    if include_headers:
        output_lines.extend(format_lrc_metadata_headers(title, artist, album, duration_seconds, songwriters, notes))
    for i, line in enumerate(structured_lines):
        output_lines.extend(_format_structured_line_to_lrc_pair(line))
        if formato == "normale" and not line.get("is_bg") and (i + 1 < len(structured_lines)):
            next_ln = structured_lines[i + 1]
            if not next_ln.get("is_bg") and (next_ln["start_ms"] - line["end_ms"]) >= 1400:
                break_sec = line["end_ms"] / 1000.0 + 0.12
                if duration_seconds <= 0 or break_sec < float(duration_seconds):
                    output_lines.append(format_lrc_time(break_sec))
    return "\n".join(output_lines)


def parse_yaml_to_structured_lines(yaml_text: str):
    """Parsing leggero per file .lyricsfile.yaml senza dipendenze esterne."""
    def _unq(s: str) -> str:
        s = s.strip()
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            try:
                return str(json.loads(s)) if s.startswith('"') else s[1:-1]
            except Exception:
                return s[1:-1]
        return s

    structured = []
    in_lines = False
    cur_line = None
    cur_word = None
    for raw_l in yaml_text.splitlines():
        if re.match(r"^\s*lines\s*:\s*$", raw_l):
            in_lines = True
            continue
        if in_lines and raw_l and not raw_l.startswith(" "):
            break
        if not in_lines:
            continue
        m_lt = re.match(r"^\s{2}-\s*text\s*:\s*(.+)$", raw_l)
        if m_lt:
            if cur_word and cur_line is not None:
                cur_line["words"].append(cur_word)
                cur_word = None
            if cur_line is not None:
                structured.append(cur_line)
            txt = _unq(m_lt.group(1)).strip()
            is_bg = txt.startswith("(") and txt.endswith(")")
            cur_line = {"text": txt, "agent": "v1", "is_bg": is_bg, "start_ms": 0, "end_ms": 0, "words": []}
            continue
        if cur_line is None:
            continue
        m_ls = re.match(r"^\s{4}start_ms\s*:\s*(\d+)", raw_l)
        if m_ls:
            cur_line["start_ms"] = int(m_ls.group(1))
            continue
        m_le = re.match(r"^\s{4}end_ms\s*:\s*(\d+)", raw_l)
        if m_le:
            cur_line["end_ms"] = int(m_le.group(1))
            continue
        m_wt = re.match(r"^\s{6}-\s*text\s*:\s*(.+)$", raw_l)
        if m_wt:
            if cur_word is not None:
                cur_line["words"].append(cur_word)
            cur_word = {"word": _unq(m_wt.group(1)).strip(), "start_ms": 0, "end_ms": 0}
            continue
        if cur_word is not None:
            m_ws = re.match(r"^\s{8}start_ms\s*:\s*(\d+)", raw_l)
            if m_ws:
                cur_word["start_ms"] = int(m_ws.group(1))
                continue
            m_we = re.match(r"^\s{8}end_ms\s*:\s*(\d+)", raw_l)
            if m_we:
                cur_word["end_ms"] = int(m_we.group(1))
                continue
    if cur_word and cur_line is not None:
        cur_line["words"].append(cur_word)
    if cur_line is not None:
        structured.append(cur_line)
    return [ln for ln in structured if ln["text"]]


def solve_lrclib_challenge(prefix: str, target_hex: str, log_func) -> str:
    prefix_bytes = prefix.encode("utf-8")
    target_int = int(target_hex, 16)
    nonce = 0
    sha256 = hashlib.sha256

    while True:
        candidate_digest = sha256(prefix_bytes + str(nonce).encode("ascii")).digest()
        if int.from_bytes(candidate_digest, "big") <= target_int:
            break
        nonce += 1
        if nonce % 250000 == 0:
            log_func(f"⏳ PoW LRCLIB in elaborazione ({nonce:,} iterazioni)...")

    return f"{prefix}:{nonce}"


def publish_and_verify_lrclib(
    track_name, artist_name, album_name, duration, plain_lyrics,
    synced_lyrics, lyricsfile_yaml, cartella_output, nome_base, log_func,
):
    try:
        pub_cfg = get_publishing_config()
        base_url = pub_cfg["lrclib_url"]
        auth_mode = pub_cfg["lrclib_auth_mode"]
        custom_token = pub_cfg["lrclib_token"]
        client_id = pub_cfg["lrclib_client_id"]

        log_func(f"🌐 Connessione ai server di LRCLIB ({base_url})...")
        api_headers = {
            "User-Agent": client_id,
            "X-User-Agent": client_id,
            "Lrclib-Client": client_id,
        }
        if auth_mode == "bearer_auth" and custom_token:
            api_headers["Authorization"] = f"Bearer {custom_token}"

        publish_token = ""
        if auth_mode == "manual_token" and custom_token:
            publish_token = custom_token
            log_func("🔑 Uso X-Publish-Token configurato manualmente.")
        else:
            try:
                req_challenge = urllib.request.Request(
                    f"{base_url}/api/request-challenge",
                    method="POST",
                    headers=api_headers,
                )
                with urllib.request.urlopen(req_challenge, timeout=15) as resp:
                    challenge_data = json.loads(resp.read().decode("utf-8"))

                prefix = challenge_data.get("prefix")
                target = challenge_data.get("target")

                if not prefix or not target:
                    raise Exception("Risposta challenge non valida da LRCLIB.")

                log_func("🔐 Risoluzione PoW LRCLIB in corso...")
                publish_token = solve_lrclib_challenge(prefix, target, log_func)
                log_func("✅ PoW LRCLIB risolto!")
            except urllib.error.HTTPError as he_ch:
                if auth_mode == "bearer_auth" and custom_token and he_ch.code in (404, 405):
                    log_func("🔑 Istanza privata senza challenge PoW: uso Bearer Token.")
                else:
                    raise

        clean_synced = format_lrc_for_lrclib(synced_lyrics)
        clean_plain = format_plain_for_lrclib(plain_lyrics, synced_lyrics)
        safe_album = album_name.strip() if album_name else ""
        safe_duration = max(1, int(round(duration)))

        payload = {
            "trackName": track_name.strip(),
            "artistName": artist_name.strip(),
            "albumName": safe_album,
            "duration": safe_duration,
            "plainLyrics": clean_plain,
            "syncedLyrics": clean_synced,
            "lyricsfile": lyricsfile_yaml,
        }

        publish_headers = dict(api_headers)
        publish_headers["Content-Type"] = "application/json"
        if publish_token:
            publish_headers["X-Publish-Token"] = publish_token

        log_func("📦 Invio a LRCLIB (Lyricsfile 1.0 Word-by-Word + LRC)...")
        req_publish = urllib.request.Request(
            f"{base_url}/api/publish",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers=publish_headers,
        )

        with urllib.request.urlopen(req_publish, timeout=20) as pub_resp:
            pub_status = pub_resp.status
            pub_body = pub_resp.read().decode("utf-8", errors="ignore")

        if pub_status not in (200, 201):
            raise Exception(f"Rifiutato da LRCLIB ({pub_status}): {pub_body}")

        log_func("📤 LRCLIB: Testo caricato con successo (Status 201 Created)!")

        server_id = None
        try:
            pub_data = json.loads(pub_body) if pub_body else {}
            server_id = pub_data.get("id")
        except Exception:
            pass

        return True, server_id or "Online", None, "Pubblicato su LRCLIB con successo."

    except urllib.error.HTTPError as he:
        err_body = he.read().decode("utf-8", errors="ignore")
        if he.code == 429:
            return False, None, None, "LRCLIB: Rate limit superato. Attendi un momento."
        return False, None, None, f"Errore LRCLIB {he.code}: {err_body}"
    except Exception as e:
        return False, None, None, str(e)


def publish_and_verify_betterlyrics(
    video_id, track_name, artist_name, album_name, duration,
    ttml_lyrics, cartella_output, nome_base, language, log_func,
):
    try:
        clean_vid = extract_youtube_id(video_id)
        if not clean_vid or len(clean_vid) != 11:
            raise Exception(f"YouTube Video ID non valido: '{video_id}'. Inserisci un link YouTube valido.")

        if not ttml_lyrics or not ttml_lyrics.strip():
            raise Exception("Il testo TTML è vuoto! Esegui prima 'AVVIA SYNC' o carica un file sincronizzato.")

        pub_cfg = get_publishing_config()
        unison_url = pub_cfg["unison_url"]
        unison_auth_mode = pub_cfg["unison_auth_mode"]
        unison_bearer = pub_cfg["unison_bearer_token"]
        unison_client_id = pub_cfg["unison_client_id"]

        log_func(f"🎵 Preparazione invio a BetterLyrics / Unison ({unison_url}, Video ID: {clean_vid})...")
        private_key, key_id, jwk = get_or_create_unison_identity()
        log_func(f"🔑 Identità Unison attiva: {key_id[:12]}...")

        lang_code = language if language and language not in ("auto", "🌐 Auto / Multilingua (Bilingue)") else "it"

        payload = {
            "keyId": key_id,
            "timestamp": int(time.time() * 1000),
            "nonce": secrets.token_hex(16),
            "videoId": clean_vid,
            "song": track_name.strip(),
            "artist": artist_name.strip(),
            "duration": int(round(duration)),
            "lyrics": ttml_lyrics,
            "format": "ttml",
            "syncType": "richsync",
            "language": lang_code,
        }

        if album_name and album_name.strip():
            payload["album"] = album_name.strip()

        canonical_json = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

        der_sig = private_key.sign(canonical_json, ec.ECDSA(hashes.SHA256()))

        r, s = decode_dss_signature(der_sig)
        raw_sig = r.to_bytes(32, byteorder="big") + s.to_bytes(32, byteorder="big")
        sig_p1363 = base64.b64encode(raw_sig).decode("ascii")
        sig_der = base64.b64encode(der_sig).decode("ascii")

        api_headers = {
            "User-Agent": unison_client_id,
            "Content-Type": "application/json",
        }
        if unison_auth_mode == "ecdsa_plus_bearer" and unison_bearer:
            api_headers["Authorization"] = f"Bearer {unison_bearer}"

        log_func("🔐 Invio pacchetto TTML conforme a Unison...")
        body_data = {
            "payload": payload,
            "signature": sig_p1363,
            "publicKey": jwk,
        }
        json_bytes = json.dumps(body_data, ensure_ascii=False).encode("utf-8")
        submit_url = f"{unison_url}/lyrics/submit"
        req = urllib.request.Request(
            submit_url,
            data=json_bytes,
            method="POST",
            headers=api_headers,
        )

        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                resp_code = resp.status
                resp_text = resp.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as he1:
            err_body = he1.read().decode("utf-8", errors="ignore")
            if he1.code == 401 and "INVALID_SIGNATURE" in err_body:
                log_func("🔄 Tentativo con codifica firma DER...")
                body_data["signature"] = sig_der
                json_bytes_der = json.dumps(body_data, ensure_ascii=False).encode("utf-8")
                req_der = urllib.request.Request(
                    submit_url,
                    data=json_bytes_der,
                    method="POST",
                    headers=api_headers,
                )
                with urllib.request.urlopen(req_der, timeout=20) as resp:
                    resp_code = resp.status
                    resp_text = resp.read().decode("utf-8", errors="ignore")
            else:
                raise urllib.error.HTTPError(he1.url, he1.code, err_body, he1.hdrs, None)

        if resp_code not in (200, 201):
            raise Exception(f"Rifiutato da Unison ({resp_code}): {resp_text}")

        log_func(f"🎉 SUCCESSO! Testo TTML Word-by-Word pubblicato su BetterLyrics per il video '{clean_vid}'!")
        log_func("✨ Ora chiunque riproduca il brano su Metrolist visualizzerà l'animazione parola per parola!")

        return True, clean_vid, os.path.join(cartella_output, f"{nome_base}.ttml"), "Online su BetterLyrics! Word-by-word attivo su Metrolist."

    except urllib.error.HTTPError as he:
        body = he.msg if hasattr(he, "msg") else str(he)
        return False, None, None, f"Errore BetterLyrics {he.code}: {body}"
    except Exception as e:
        return False, None, None, str(e)


# ==============================================================================
# RICERCA & DOWNLOAD TESTI ONLINE (UNISON / BETTERLYRICS + LRCLIB)
# ==============================================================================
def _http_json_get(url: str, headers: dict = None, timeout: int = 12):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise Exception("URL non valido (solo HTTP/HTTPS).")
    req = urllib.request.Request(url, method="GET", headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _classify_lrclib_item_sync(item: dict) -> str:
    if item.get("lyricsfile") or item.get("hasWordSync"):
        return "word"
    synced = item.get("syncedLyrics") or ""
    if synced.strip():
        if re.search(r"<\d{1,3}:\d{2}(?:\.\d+)?>", synced):
            return "word"
        return "line"
    return "plain"


def _classify_unison_item_sync(item: dict) -> str:
    st = str(item.get("syncType") or "").strip().lower()
    fmt = str(item.get("format") or "").strip().lower()
    if st in ("richsync", "word", "syllable") or fmt == "ttml":
        if st in ("synced", "line"):
            return "line"
        if st in ("plain", "unsynced"):
            return "plain"
        return "word"
    if st in ("synced", "line") or fmt == "lrc":
        return "line"
    return "plain"


def search_lrclib_lyrics(query: str, title: str = "", artist: str = "", album: str = "", pub_cfg: dict = None):
    pub_cfg = pub_cfg or get_publishing_config()
    base_url = pub_cfg["lrclib_url"]
    headers = {
        "User-Agent": pub_cfg["lrclib_client_id"],
        "X-User-Agent": pub_cfg["lrclib_client_id"],
        "Lrclib-Client": pub_cfg["lrclib_client_id"],
    }
    if pub_cfg.get("lrclib_auth_mode") == "bearer_auth" and pub_cfg.get("lrclib_token"):
        headers["Authorization"] = f"Bearer {pub_cfg['lrclib_token']}"

    results_map = {}
    queries_to_try = []
    if query.strip():
        queries_to_try.append({"q": query.strip()})
    if title.strip() and artist.strip():
        qs = {"track_name": title.strip(), "artist_name": artist.strip()}
        if album.strip():
            qs["album_name"] = album.strip()
        queries_to_try.append(qs)
    elif title.strip() and title.strip().lower() != query.strip().lower():
        queries_to_try.append({"q": title.strip()})

    for qp in queries_to_try:
        url = f"{base_url}/api/search?{urllib.parse.urlencode(qp)}"
        try:
            data = _http_json_get(url, headers=headers, timeout=12)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict) or item.get("instrumental"):
                        continue
                    has_any = bool((item.get("syncedLyrics") or "").strip() or (item.get("plainLyrics") or "").strip() or (item.get("lyricsfile") or "").strip())
                    if not has_any:
                        continue
                    iid = item.get("id")
                    if iid is not None and iid not in results_map:
                        results_map[iid] = item
        except Exception:
            if not results_map and qp == queries_to_try[-1]:
                raise
    return list(results_map.values())


def search_unison_lyrics(query: str, title: str = "", artist: str = "", video_id: str = "", pub_cfg: dict = None):
    pub_cfg = pub_cfg or get_publishing_config()
    base_url = pub_cfg["unison_url"]
    headers = {"User-Agent": pub_cfg["unison_client_id"]}
    if pub_cfg.get("unison_auth_mode") == "ecdsa_plus_bearer" and pub_cfg.get("unison_bearer_token"):
        headers["Authorization"] = f"Bearer {pub_cfg['unison_bearer_token']}"

    items_map = {}
    clean_vid = extract_youtube_id(video_id) if video_id else ""
    if clean_vid and len(clean_vid) == 11:
        try:
            resp = _http_json_get(f"{base_url}/lyrics?{urllib.parse.urlencode({'v': clean_vid})}", headers=headers, timeout=12)
            if isinstance(resp, dict) and resp.get("success") and isinstance(resp.get("data"), dict):
                d = resp["data"]
                d["_exact_video_match"] = True
                items_map[d.get("id") or clean_vid] = d
        except urllib.error.HTTPError as he:
            if he.code != 404:
                pass
        except Exception:
            pass

    q_candidates = []
    if query.strip():
        q_candidates.append(query.strip())
    elif title.strip() or artist.strip():
        q_candidates.append(" ".join(p for p in (artist.strip(), title.strip()) if p))

    for q_str in q_candidates:
        url = f"{base_url}/lyrics/search?{urllib.parse.urlencode({'q': q_str})}"
        resp = _http_json_get(url, headers=headers, timeout=12)
        rows = resp.get("data") if isinstance(resp, dict) else resp
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                rid = row.get("id")
                if rid is not None and rid not in items_map:
                    items_map[rid] = row
        break

    return list(items_map.values())


def search_all_online_lyrics(params: dict) -> dict:
    title = str(params.get("title") or "").strip()
    artist = str(params.get("artist") or "").strip()
    album = str(params.get("album") or "").strip()
    youtube = str(params.get("youtube") or "").strip()
    query = str(params.get("query") or "").strip()
    source = str(params.get("source") or "all").strip().lower()
    target_dur = float(params.get("duration") or 0)

    if not query:
        query = " ".join(p for p in (artist, title) if p).strip()
    if not query and not (len(extract_youtube_id(youtube)) == 11):
        return {"success": False, "error": "Inserisci almeno il Titolo/Artista o un Video ID YouTube.", "results": [], "errors": []}

    pub_cfg = get_publishing_config()
    clean_vid = extract_youtube_id(youtube) if youtube else ""
    if len(clean_vid) != 11:
        clean_vid = ""

    combined = []
    errors = []

    if source in ("all", "unison", "betterlyrics"):
        try:
            u_items = search_unison_lyrics(query, title=title, artist=artist, video_id=clean_vid, pub_cfg=pub_cfg)
            for item in u_items:
                sync_lvl = _classify_unison_item_sync(item)
                submitter = item.get("submitter") if isinstance(item.get("submitter"), dict) else {}
                combined.append({
                    "uid": f"unison:{item.get('id')}",
                    "provider": "unison",
                    "provider_label": "BetterLyrics / Unison",
                    "id": item.get("id"),
                    "video_id": item.get("videoId") or "",
                    "title": item.get("song") or item.get("title") or "",
                    "artist": item.get("artist") or "",
                    "album": item.get("album") or "",
                    "duration": float(item.get("duration") or 0),
                    "language": _normalize_lang_code(item.get("language") or ""),
                    "sync_level": sync_lvl,
                    "raw_format": str(item.get("format") or "ttml").lower(),
                    "submitter": submitter.get("displayName") or "",
                    "confidence": item.get("confidence") or "",
                    "exact_video_match": bool(item.get("_exact_video_match") or (clean_vid and item.get("videoId") == clean_vid)),
                    "has_inline_lyrics": bool(item.get("lyrics")),
                    "raw_item": item,
                })
        except Exception as ex:
            errors.append(f"Unison: {ex}")

    if source in ("all", "lrclib"):
        try:
            l_items = search_lrclib_lyrics(query, title=title, artist=artist, album=album, pub_cfg=pub_cfg)
            for item in l_items:
                sync_lvl = _classify_lrclib_item_sync(item)
                snippet = (item.get("plainLyrics") or item.get("syncedLyrics") or "")[:160].replace("\n", " / ")
                combined.append({
                    "uid": f"lrclib:{item.get('id')}",
                    "provider": "lrclib",
                    "provider_label": "LRCLIB",
                    "id": item.get("id"),
                    "video_id": "",
                    "title": item.get("trackName") or item.get("name") or "",
                    "artist": item.get("artistName") or "",
                    "album": item.get("albumName") or "",
                    "duration": float(item.get("duration") or 0),
                    "language": "",
                    "sync_level": sync_lvl,
                    "raw_format": "yaml" if item.get("lyricsfile") else ("lrc" if item.get("syncedLyrics") else "txt"),
                    "submitter": "",
                    "confidence": "",
                    "exact_video_match": False,
                    "has_inline_lyrics": True,
                    "preview_snippet": snippet,
                    "raw_item": item,
                })
        except Exception as ex:
            errors.append(f"LRCLIB: {ex}")

    def _score_result(r: dict) -> float:
        score = 0.0
        if r.get("exact_video_match"):
            score += 100.0
        sync_bonus = {"word": 30.0, "line": 15.0, "plain": 2.0}
        score += sync_bonus.get(r.get("sync_level"), 0.0)
        if r.get("provider") == "unison" and r.get("sync_level") == "word":
            score += 8.0
        if target_dur > 0 and r.get("duration", 0) > 0:
            diff = abs(target_dur - r["duration"])
            if diff <= 2.5:
                score += 20.0
            elif diff <= 6.0:
                score += 10.0
            elif diff > 20.0:
                score -= 10.0
        if title:
            score += 18.0 * SequenceMatcher(None, title.lower(), (r.get("title") or "").lower()).ratio()
        if artist:
            score += 14.0 * SequenceMatcher(None, artist.lower(), (r.get("artist") or "").lower()).ratio()
        return score

    combined.sort(key=_score_result, reverse=True)
    return {"success": True, "results": combined, "errors": errors}


def fetch_online_lyrics_raw(provider: str, item_id, video_id: str = "", raw_item: dict = None) -> dict:
    pub_cfg = get_publishing_config()
    if provider == "unison":
        data = dict(raw_item) if isinstance(raw_item, dict) else {}
        if not data.get("lyrics"):
            headers = {"User-Agent": pub_cfg["unison_client_id"]}
            if pub_cfg.get("unison_auth_mode") == "ecdsa_plus_bearer" and pub_cfg.get("unison_bearer_token"):
                headers["Authorization"] = f"Bearer {pub_cfg['unison_bearer_token']}"
            if item_id is not None and str(item_id).strip():
                url = f"{pub_cfg['unison_url']}/lyrics/{urllib.parse.quote(str(item_id))}"
            elif video_id:
                url = f"{pub_cfg['unison_url']}/lyrics?{urllib.parse.urlencode({'v': video_id})}"
            else:
                raise Exception("ID Unison mancante.")
            resp = _http_json_get(url, headers=headers, timeout=15)
            if isinstance(resp, dict) and isinstance(resp.get("data"), dict):
                data = resp["data"]
            elif isinstance(resp, dict) and resp.get("lyrics"):
                data = resp
        raw_content = str(data.get("lyrics") or "").strip()
        if not raw_content:
            raise Exception("Il server Unison non ha restituito alcun testo per questa voce.")
        return {
            "provider": "unison",
            "title": data.get("song") or data.get("title") or "",
            "artist": data.get("artist") or "",
            "album": data.get("album") or "",
            "duration": float(data.get("duration") or 0),
            "language": _normalize_lang_code(data.get("language") or ""),
            "video_id": data.get("videoId") or video_id or "",
            "format": str(data.get("format") or "").lower(),
            "sync_type": str(data.get("syncType") or "").lower(),
            "lyrics_raw": raw_content,
            "plain_raw": "",
            "lyricsfile_raw": "",
        }

    if provider == "lrclib":
        data = dict(raw_item) if isinstance(raw_item, dict) else {}
        needs_get = (
            not any(data.get(k) for k in ("syncedLyrics", "plainLyrics", "lyricsfile"))
            or (bool(data.get("hasWordSync")) and not data.get("lyricsfile"))
        )
        if needs_get and item_id is not None:
            headers = {
                "User-Agent": pub_cfg["lrclib_client_id"],
                "X-User-Agent": pub_cfg["lrclib_client_id"],
                "Lrclib-Client": pub_cfg["lrclib_client_id"],
            }
            if pub_cfg.get("lrclib_auth_mode") == "bearer_auth" and pub_cfg.get("lrclib_token"):
                headers["Authorization"] = f"Bearer {pub_cfg['lrclib_token']}"
            data = _http_json_get(f"{pub_cfg['lrclib_url']}/api/get/{urllib.parse.quote(str(item_id))}", headers=headers, timeout=15)
        return {
            "provider": "lrclib",
            "title": data.get("trackName") or data.get("name") or "",
            "artist": data.get("artistName") or "",
            "album": data.get("albumName") or "",
            "duration": float(data.get("duration") or 0),
            "language": "",
            "video_id": "",
            "format": "yaml" if data.get("lyricsfile") else ("lrc" if data.get("syncedLyrics") else "txt"),
            "sync_type": "word" if data.get("lyricsfile") or data.get("hasWordSync") else ("line" if data.get("syncedLyrics") else "plain"),
            "lyrics_raw": str(data.get("syncedLyrics") or ""),
            "plain_raw": str(data.get("plainLyrics") or ""),
            "lyricsfile_raw": str(data.get("lyricsfile") or ""),
        }

    raise Exception(f"Provider non supportato: {provider}")


def parse_downloaded_lyrics_bundle(fetched: dict):
    """Parses downloaded Unison/LRCLIB payload into (structured_lines, plain_text, sync_level, extracted_meta)."""
    lyricsfile_raw = (fetched.get("lyricsfile_raw") or "").strip()
    lyrics_raw = (fetched.get("lyrics_raw") or "").strip()
    plain_raw = (fetched.get("plain_raw") or "").strip()

    structured = []
    extracted_meta = {}

    if lyricsfile_raw and re.search(r"^\s*lines\s*:", lyricsfile_raw, re.MULTILINE):
        structured = parse_yaml_to_structured_lines(lyricsfile_raw)
        extracted_meta = extract_lyrics_metadata(lyricsfile_raw)

    if not structured and lyrics_raw:
        low = lyrics_raw.lower()
        if "<tt" in low and "<p" in low:
            structured = parse_ttml_to_structured_lines(lyrics_raw)
            extracted_meta = extract_lyrics_metadata(lyrics_raw)
        elif re.search(r"^\s*version\s*:", lyrics_raw, re.MULTILINE) and re.search(r"^\s*lines\s*:", lyrics_raw, re.MULTILINE):
            structured = parse_yaml_to_structured_lines(lyrics_raw)
            extracted_meta = extract_lyrics_metadata(lyrics_raw)
        elif re.search(r"\[\d{1,3}:\d{2}", lyrics_raw):
            structured = parse_lrc_to_structured_lines(lyrics_raw)
            extracted_meta = extract_lyrics_metadata(lyrics_raw)

    if structured:
        has_words = any(bool(ln.get("words")) for ln in structured)
        sync_level = "word" if has_words else "line"
        if plain_raw:
            plain_out = format_plain_for_lrclib(plain_raw)
        else:
            plain_lines = []
            for ln in structured:
                t = ln.get("text", "").strip()
                if t:
                    plain_lines.append(t)
            plain_out = "\n".join(plain_lines)
        return structured, plain_out, sync_level, extracted_meta

    plain_fallback = format_plain_for_lrclib(plain_raw or lyrics_raw)
    return [], plain_fallback.strip(), "plain", extracted_meta


# ==============================================================================
# BACKEND API & CONTROLLER PRINCIPALE
# ==============================================================================
class BackendAPI:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"
        self.last_audio_path = ""
        self.generated_lrc = ""
        self.generated_lyricsfile = ""
        self.generated_ttml = ""
        self.plain_lyrics = ""
        self.calculated_duration = 0
        self.calculated_duration_exact = 0.0
        self.current_title = ""
        self.current_artist = ""
        self.current_album = ""
        self.current_youtube = ""
        self.current_songwriters = ""
        self.current_notes = ""
        self.current_outro_text = load_config().get("outro_text", DEFAULT_OUTRO_TEXT)
        self.current_outro_credits = load_config().get("outro_credits", True)
        self._last_outro_candidates = set()
        self.last_output_paths = {}
        self._is_yt_temp_audio = False
        self._yt_output_basename = ""
        self.detected_language = "it"
        self.is_alive = True
        self._model_dl_lock = threading.Lock()
        self._model_dl_cancel = threading.Event()
        self._model_dl_active_id = ""
        self._model_dl_active_pct = 0.0

    def _get_window(self):
        if not self.is_alive or not webview.windows:
            return None
        return webview.windows[0]

    def init_info(self):
        cfg = load_config()
        best_model = get_best_available_model_path()
        default_model = cfg.get("model_path")
        if not default_model or not os.path.exists(default_model):
            default_model = best_model

        available_models = get_available_models_list()

        return {
            "device": self.device.upper(),
            "gpu_name": self.gpu_name,
            "default_model": default_model,
            "available_models": available_models,
            "model_catalog": get_whisper_models_status(self._model_dl_active_id, self._model_dl_active_pct),
            "language": cfg.get("language", "🌐 Auto / Multilingua (Bilingue)"),
            "demucs": cfg.get("demucs", True),
            "demucs_model": cfg.get("demucs_model", "htdemucs"),
            "demucs_catalog": get_demucs_models_status(),
            "vad": cfg.get("vad", True),
            "format": cfg.get("format", "metrolist"),
            "offset": cfg.get("offset", 0),
            "auto_offset": cfg.get("auto_offset", True),
            "outro_credits": cfg.get("outro_credits", True),
            "outro_text": cfg.get("outro_text", DEFAULT_OUTRO_TEXT),
            "languages": list(LANGUAGES.keys()),
            "publishing": get_publishing_config(),
            "unison_identity": get_unison_identity_status(),
        }

    def save_user_settings(self, settings):
        merged = load_config()
        if isinstance(settings, dict):
            merged.update(settings)
            if "outro_credits" in settings:
                self.current_outro_credits = bool(settings["outro_credits"])
        return save_config(merged)

    def log(self, text):
        if not self.is_alive:
            return
        win = self._get_window()
        if win:
            try:
                safe = json.dumps(str(text))
                win.evaluate_js(f"window.appendLog({safe});")
            except Exception:
                pass

    def progress(self, pct):
        if not self.is_alive:
            return
        win = self._get_window()
        if win:
            try:
                win.evaluate_js(f"window.updateProgress({pct});")
            except Exception:
                pass

    def _sync_state_metadata(self, meta: dict):
        if not meta:
            return
        if meta.get("title"):
            self.current_title = meta["title"].strip()
        if meta.get("artist"):
            self.current_artist = meta["artist"].strip()
        if meta.get("album"):
            self.current_album = meta["album"].strip()
        if meta.get("youtube"):
            self.current_youtube = meta["youtube"].strip()
        if meta.get("songwriters") is not None:
            self.current_songwriters = meta.get("songwriters", "").strip()
        if meta.get("notes") is not None:
            self.current_notes = meta.get("notes", "").strip()
        if meta.get("outro_text") is not None:
            self.current_outro_text = meta.get("outro_text", "").strip()
        if "outro_credits" in meta and meta.get("outro_credits") is not None:
            self.current_outro_credits = bool(meta.get("outro_credits"))
        if meta.get("language"):
            norm_l = _normalize_lang_code(meta["language"])
            if norm_l:
                self.detected_language = norm_l

    def _write_synced_output_files(self, lrc_path: str, yaml_path: str, ttml_path: str):
        if lrc_path and self.generated_lrc:
            with open(lrc_path, "w", encoding="utf-8") as out_file:
                out_file.write(self.generated_lrc + "\n")
        if yaml_path and self.generated_lyricsfile:
            with open(yaml_path, "w", encoding="utf-8") as out_yaml:
                out_yaml.write(self.generated_lyricsfile)
        if ttml_path and self.generated_ttml:
            with open(ttml_path, "w", encoding="utf-8") as out_ttml:
                out_ttml.write(self.generated_ttml)

    def _rebuild_outputs_from_structured(
        self, structured, title, artist, album, duration_sec, language,
        songwriters="", notes="", plain_text="", preserve_body_lrc=None, include_headers=True,
        outro_credits=None, outro_text=None, exact_duration_sec=None, formato=None,
    ):
        use_credits = self.current_outro_credits if outro_credits is None else bool(outro_credits)
        target_outro_text = self.current_outro_text if outro_text is None else str(outro_text or "").strip()
        target_format = formato or load_config().get("format", "metrolist")
        eff_dur = float(duration_sec or 0)
        if exact_duration_sec and float(exact_duration_sec) > 0:
            eff_dur = min(eff_dur, float(exact_duration_sec)) if eff_dur > 0 else float(exact_duration_sec)

        extra_known = set(self._last_outro_candidates)
        if target_outro_text:
            extra_known.add(target_outro_text)
        if notes:
            extra_known.add(notes)

        structured_out, credit_entry = apply_outro_credits_to_structured(
            structured,
            songwriters=songwriters,
            duration_sec=eff_dur,
            enabled=use_credits,
            outro_text=target_outro_text,
            notes=notes,
            known_outro_texts=extra_known,
        )
        if credit_entry is not None:
            self._last_outro_candidates.add(credit_entry["text"])

        if preserve_body_lrc:
            raw_body_lines = [
                ln for ln in preserve_body_lrc.splitlines()
                if not LRC_META_TAG_RE.match(ln.strip())
            ]
            body_lines = strip_outro_credits_from_lrc_lines(raw_body_lines, extra_texts=extra_known)
            if credit_entry is not None:
                body_lines.extend(_format_structured_line_to_lrc_pair(credit_entry))
            if include_headers:
                hdrs = format_lrc_metadata_headers(title, artist, album, duration_sec, songwriters, notes)
                self.generated_lrc = "\n".join(hdrs + body_lines).strip()
            else:
                self.generated_lrc = "\n".join(body_lines).strip()
        else:
            self.generated_lrc = convert_structured_to_lrc(
                structured_out,
                title=title,
                artist=artist,
                album=album,
                duration_seconds=duration_sec,
                songwriters=songwriters,
                notes=notes,
                include_headers=include_headers,
                formato=target_format,
            )

        base_plain = plain_text.strip() if (plain_text and plain_text.strip()) else format_plain_for_lrclib("", self.generated_lrc)
        base_plain = strip_outro_credits_from_plain(base_plain, extra_texts=extra_known)
        if credit_entry is not None and base_plain:
            self.plain_lyrics = f"{base_plain}\n{credit_entry['text']}"
        elif credit_entry is not None:
            self.plain_lyrics = credit_entry["text"]
        else:
            self.plain_lyrics = base_plain

        self.generated_lyricsfile = build_lyricsfile_yaml(
            title=title,
            artist=artist,
            album=album,
            duration_ms=int(round(duration_sec)) * 1000,
            language=language,
            structured_lines=structured_out,
            plain_text=self.plain_lyrics,
            songwriters=songwriters,
            notes=notes,
        )
        self.generated_ttml = build_ttml(
            title=title,
            artist=artist,
            album=album,
            duration_seconds=duration_sec,
            language=language,
            structured_lines=structured_out,
            songwriters=songwriters,
            notes=notes,
        )

    def select_audio_file(self):
        win = self._get_window()
        if not win:
            return None
        file_types = ("File Audio (*.mp3;*.wav;*.flac;*.m4a;*.ogg)",)
        result = win.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=False,
            file_types=file_types,
        )
        if result and len(result) > 0:
            file_path = result[0]
            self.last_audio_path = file_path
            self._is_yt_temp_audio = False
            self._yt_output_basename = ""
            self.last_output_paths = {}
            self.calculated_duration_exact = 0.0

            meta = extract_audio_metadata(file_path)
            if meta["duration"] < 1:
                try:
                    audio_arr = whisper.load_audio(file_path)
                    self.calculated_duration_exact = float(len(audio_arr) / 16000.0)
                    meta["duration"] = int(round(self.calculated_duration_exact))
                except Exception:
                    pass
            else:
                self.calculated_duration_exact = float(meta["duration"])

            self.calculated_duration = meta["duration"]
            self.current_title = meta["title"]
            self.current_artist = meta["artist"]
            self.current_album = meta["album"]
            self.current_youtube = meta["youtube"]
            self.current_songwriters = meta["songwriters"]
            if meta["language"]:
                self.detected_language = meta["language"]

            return {
                "path": file_path,
                **meta,
            }
        return None

    def auto_fill_metadata_from_files(self, audio_path: str = "", lyrics_path: str = ""):
        """Riestrae tutti i metadati possibili dai file audio e testo attualmente selezionati."""
        combined = {
            "title": "",
            "artist": "",
            "album": "",
            "duration": 0,
            "youtube": "",
            "songwriters": "",
            "notes": "",
            "language": "",
        }
        a_path = (audio_path or self.last_audio_path or "").strip()
        if a_path and os.path.exists(a_path):
            combined.update(extract_audio_metadata(a_path))
        if lyrics_path and os.path.exists(lyrics_path):
            try:
                with open(lyrics_path, "r", encoding="utf-8", errors="ignore") as f:
                    l_meta = extract_lyrics_metadata(f.read())
                for k, v in l_meta.items():
                    if v and not combined.get(k):
                        combined[k] = v
            except Exception:
                pass
        return combined

    def update_outputs_metadata(self, metadata: dict):
        """Rigenera LRC, TTML, Lyricsfile YAML e i file sincronizzati su disco con i metadati aggiornati dalla UI."""
        if not metadata:
            return {"success": False}
        title = (metadata.get("title") or self.current_title or "Senza Titolo").strip()
        artist = (metadata.get("artist") or self.current_artist or "").strip()
        album = (metadata.get("album") or "").strip()
        youtube = (metadata.get("youtube") or "").strip()
        songwriters = (metadata.get("songwriters") or "").strip()
        notes = (metadata.get("notes") or "").strip()
        outro_text = (metadata.get("outro_text") or "").strip() if "outro_text" in metadata else self.current_outro_text
        dur = int(metadata.get("duration") or self.calculated_duration or 0)
        lang_in = _normalize_lang_code(metadata.get("language", "")) or self.detected_language
        outro_credits = metadata.get("outro_credits")
        if outro_credits is not None:
            self.current_outro_credits = bool(outro_credits)

        self.current_title = title
        self.current_artist = artist
        self.current_album = album
        self.current_youtube = youtube
        self.current_songwriters = songwriters
        self.current_notes = notes
        self.current_outro_text = outro_text
        self.detected_language = lang_in

        structured = parse_lrc_to_structured_lines(self.generated_lrc) if self.generated_lrc else []
        if not structured and self.generated_ttml:
            structured = parse_ttml_to_structured_lines(self.generated_ttml)
        if structured:
            self._rebuild_outputs_from_structured(
                structured,
                title=title,
                artist=artist,
                album=album,
                duration_sec=dur,
                language=lang_in,
                songwriters=songwriters,
                notes=notes,
                plain_text=self.plain_lyrics,
                preserve_body_lrc=self.generated_lrc,
                include_headers=True,
                outro_credits=self.current_outro_credits,
                outro_text=outro_text,
                exact_duration_sec=self.calculated_duration_exact,
            )
            if self.last_output_paths:
                try:
                    self._write_synced_output_files(
                        self.last_output_paths.get("lrc", ""),
                        self.last_output_paths.get("yaml", ""),
                        self.last_output_paths.get("ttml", ""),
                    )
                except Exception:
                    pass
            return {
                "success": True,
                "lrc": self.generated_lrc,
                "ttml": self.generated_ttml,
                "lyricsfile": self.generated_lyricsfile,
            }
        return {"success": False}

    def _process_synced_content(self, content: str, file_path: str):
        content = content.strip()
        low = content.lower()
        is_ttml = "<tt" in low and "<p" in low
        is_yaml = (not is_ttml) and bool(re.search(r"^\s*version\s*:", content, re.MULTILINE) and re.search(r"^\s*lines\s*:", content, re.MULTILINE))

        l_meta = extract_lyrics_metadata(content)

        if is_ttml:
            structured = parse_ttml_to_structured_lines(content)
            raw_body = None
        elif is_yaml:
            structured = parse_yaml_to_structured_lines(content)
            raw_body = None
        else:
            structured = parse_lrc_to_structured_lines(content)
            raw_body = content

        if not structured:
            raise Exception("Nessuna riga sincronizzata valida trovata nel file.")

        max_ms = max(line["end_ms"] for line in structured) if structured else 0
        dur_sec = self.calculated_duration if self.calculated_duration > 0 else (int(round(max_ms / 1000.0)) + 2)

        nome_base = os.path.splitext(os.path.basename(file_path))[0]
        title = self.current_title or l_meta.get("title") or nome_base
        artist = self.current_artist or l_meta.get("artist") or ""
        album = self.current_album or l_meta.get("album") or ""
        songwriters = l_meta.get("songwriters") or self.current_songwriters or ""
        notes = l_meta.get("notes") or self.current_notes or ""
        lang_code = l_meta.get("language") or self.detected_language

        self.current_title = title
        self.current_artist = artist
        self.current_album = album
        self.current_songwriters = songwriters
        self.current_notes = notes
        self.detected_language = lang_code
        self.calculated_duration = dur_sec
        self.calculated_duration_exact = float(dur_sec)
        self.last_output_paths = {}

        self._rebuild_outputs_from_structured(
            structured,
            title=title,
            artist=artist,
            album=album,
            duration_sec=dur_sec,
            language=lang_code,
            songwriters=songwriters,
            notes=notes,
            preserve_body_lrc=raw_body,
            include_headers=True,
            outro_text=self.current_outro_text,
            exact_duration_sec=self.calculated_duration_exact,
        )

        return {
            "lrc": self.generated_lrc,
            "ttml": self.generated_ttml,
            "lyricsfile": self.generated_lyricsfile,
            "plain": self.plain_lyrics,
            "duration": dur_sec,
            "line_count": len(structured),
            "meta": {
                "title": l_meta.get("title") or "",
                "artist": l_meta.get("artist") or "",
                "album": l_meta.get("album") or "",
                "songwriters": l_meta.get("songwriters") or "",
                "notes": l_meta.get("notes") or "",
                "language": l_meta.get("language") or "",
            },
        }

    def select_lyrics_file(self):
        win = self._get_window()
        if not win:
            return None
        file_types = ("File di Testo o Sincronizzati (*.txt;*.lrc;*.ttml;*.yaml)",)
        result = win.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=False,
            file_types=file_types,
        )
        if result and len(result) > 0:
            file_path = result[0]
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                is_yaml = bool(re.search(r"^\s*version\s*:", content, re.MULTILINE) and re.search(r"^\s*lines\s*:", content, re.MULTILINE))
                if re.search(r"\[\d{1,3}:\d{2}", content) or "<tt" in content.lower() or is_yaml:
                    parsed = self._process_synced_content(content, file_path)
                    return {"path": file_path, "content": content, "is_synced": True, **parsed}

                l_meta = extract_lyrics_metadata(content)
                return {"path": file_path, "content": content, "is_synced": False, "meta": l_meta}
            except Exception as e:
                return {"error": str(e)}
        return None

    def import_synced_lyrics_file(self):
        win = self._get_window()
        if not win:
            return None
        file_types = ("File Sincronizzati (*.lrc;*.ttml;*.yaml;*.txt)",)
        result = win.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=False,
            file_types=file_types,
        )
        if result and len(result) > 0:
            file_path = result[0]
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                parsed = self._process_synced_content(content, file_path)
                return {"success": True, "path": file_path, **parsed}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return None

    def select_model_file(self):
        win = self._get_window()
        if not win:
            return None
        file_types = ("Pesi Whisper (*.pt)",)
        result = win.create_file_dialog(
            webview.FileDialog.OPEN,
            directory=MODELS_DIR,
            allow_multiple=False,
            file_types=file_types,
        )
        if result and len(result) > 0:
            return {"path": result[0]}
        return None

    def save_project_file(self, data):
        win = self._get_window()
        if not win:
            return None
        title = data.get("title", "Canzone")
        artist = data.get("artist", "")
        def_name = f"{artist} - {title}.metro" if artist else f"{title}.metro"
        def_name = re.sub(r'[\\/*?:"<>|]', "", def_name)

        file_types = ("Progetto MetroSync (*.metro;*.json)",)
        result = win.create_file_dialog(
            webview.FileDialog.SAVE,
            directory=_get_song_output_dir(title=title, artist=artist),
            save_filename=def_name,
            file_types=file_types,
        )
        if result:
            save_path = result[0] if isinstance(result, (list, tuple)) else result
            try:
                data["generated_lrc"] = self.generated_lrc
                data["generated_lyricsfile"] = self.generated_lyricsfile
                data["generated_ttml"] = self.generated_ttml
                data["plain_lyrics"] = self.plain_lyrics
                data["duration"] = self.calculated_duration
                data["audio_path"] = self.last_audio_path
                yt_val = (data.get("youtube") or self.current_youtube or "").strip()
                self.current_youtube = yt_val
                data["youtube_id"] = yt_val
                self.current_songwriters = (data.get("songwriters") or "").strip()
                self.current_notes = (data.get("notes") or "").strip()

                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return {"success": True, "path": save_path}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return None

    def load_project_file(self):
        win = self._get_window()
        if not win:
            return None
        file_types = ("Progetto MetroSync (*.metro;*.json)",)
        result = win.create_file_dialog(
            webview.FileDialog.OPEN,
            directory=OUTPUT_ROOT_DIR,
            allow_multiple=False,
            file_types=file_types,
        )
        if result and len(result) > 0:
            load_path = result[0]
            try:
                with open(load_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.last_audio_path = data.get("audio_path", "")
                self.generated_lrc = data.get("generated_lrc", "")
                self.generated_lyricsfile = data.get("generated_lyricsfile", "")
                self.plain_lyrics = data.get("plain_lyrics", "")
                self.calculated_duration = data.get("duration", 0)
                self.current_title = data.get("title", "")
                self.current_artist = data.get("artist", "")
                self.current_album = data.get("album", "")
                self.current_youtube = data.get("youtube") or data.get("youtube_id", "")
                self.current_songwriters = data.get("songwriters", "")
                self.current_notes = data.get("notes", "")
                self.current_outro_text = data.get("outro_text", "")
                self.last_output_paths = {}
                self.calculated_duration_exact = float(self.calculated_duration or 0)
                if "outro_credits" in data:
                    self.current_outro_credits = bool(data["outro_credits"])
                elif isinstance(data.get("settings"), dict) and "outro_credits" in data["settings"]:
                    self.current_outro_credits = bool(data["settings"]["outro_credits"])
                if data.get("language"):
                    norm_l = _normalize_lang_code(data.get("language", ""))
                    if norm_l:
                        self.detected_language = norm_l

                if self.generated_lrc:
                    structured = parse_lrc_to_structured_lines(self.generated_lrc)
                    if structured:
                        self._rebuild_outputs_from_structured(
                            structured,
                            title=self.current_title or "Senza Titolo",
                            artist=self.current_artist or "Sconosciuto",
                            album=self.current_album or "",
                            duration_sec=self.calculated_duration or 180,
                            language=self.detected_language,
                            songwriters=self.current_songwriters,
                            notes=self.current_notes,
                            plain_text=self.plain_lyrics,
                            preserve_body_lrc=self.generated_lrc,
                            include_headers=True,
                        )

                data["project_file_path"] = load_path
                data["youtube_id"] = self.current_youtube
                data["generated_lrc"] = self.generated_lrc
                data["generated_lyricsfile"] = self.generated_lyricsfile
                data["generated_ttml"] = self.generated_ttml
                return {"success": True, "data": data}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return None

    def get_audio_data_url(self):
        if not self.last_audio_path or not os.path.exists(self.last_audio_path):
            return None
        try:
            mime, _ = mimetypes.guess_type(self.last_audio_path)
            if not mime:
                mime = "audio/mpeg"
            with open(self.last_audio_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime};base64,{b64}"
        except Exception:
            return None

    def _extract_vocal_stems(self, audio_path: str, audio_raw: np.ndarray, usa_demucs: bool, progress_cb=None, demucs_model: str = None):
        """Estrae stem vocals e filtered_vocal (vocals + other banda 85-3400Hz) con fallback sicuro."""
        if not usa_demucs:
            return audio_raw, None
        chosen_demucs = demucs_model or load_config().get("demucs_model", "htdemucs")
        if chosen_demucs not in ("htdemucs", "htdemucs_ft"):
            chosen_demucs = "htdemucs"
        try:
            label_dm = "htdemucs_ft (Massima Precisione Vocale)" if chosen_demucs == "htdemucs_ft" else "htdemucs"
            self.log(f"🎤 Isolamento vocale neurale attivo (Demucs {label_dm})... Estrazione a cappella...")
            if progress_cb:
                progress_cb(15)
            try:
                vocal_audio, filtered_vocal = _separate_demucs_stems(audio_path, self.device, demucs_model=chosen_demucs)
            except Exception:
                import demucs  # noqa: F401
                from stable_whisper.audio import demucs_audio
                vocal_tensor = demucs_audio(
                    audio_path,
                    output_sr=16000,
                    device=self.device,
                    verbose=False,
                )
                vocal_audio = vocal_tensor.cpu().numpy()
                if vocal_audio.ndim > 1:
                    vocal_audio = vocal_audio.squeeze()
                    if vocal_audio.ndim > 1:
                        vocal_audio = vocal_audio.mean(axis=0)
                max_v = np.max(np.abs(vocal_audio))
                if max_v > 1e-4:
                    vocal_audio = vocal_audio / max_v
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                filtered_vocal = None
            self.log("✨ Voce isolata con successo! Eliminati beat, kick drum e strumenti.")
            return vocal_audio, filtered_vocal
        except Exception as ex_demucs:
            self.log(f"⚠️ Demucs non disponibile ({ex_demucs}), uso traccia nativa.")
            return audio_raw, None

    # ==========================================================================
    # MODALITÀ ANALISI E CALIBRAZIONE AUTOMATICA DELL'OFFSET
    # ==========================================================================
    def analyze_auto_compensation(self, params=None):
        if params is None:
            params = {}

        def worker():
            w_model = None
            try:
                audio_path = params.get("audio_path") or self.last_audio_path
                lyrics_text = params.get("lyrics_text") or self.plain_lyrics
                model_path = params.get("model_path", "")
                lang_label = params.get("language")
                usa_demucs = bool(params.get("demucs", load_config().get("demucs", True)))
                demucs_model = params.get("demucs_model") or load_config().get("demucs_model", "htdemucs")

                if not audio_path or not os.path.exists(audio_path):
                    yt_source = (params.get("youtube_audio_url") or params.get("youtube") or "").strip()
                    if yt_source and is_youtube_audio_source(yt_source):
                        yt_res = download_youtube_audio_for_sync(yt_source, log_func=self.log)
                        audio_path = yt_res["audio_path"]
                        self.last_audio_path = audio_path
                        self._is_yt_temp_audio = True
                        self._yt_output_basename = yt_res["base_name"]
                        yt_meta = yt_res.get("meta") or {}
                        self.calculated_duration = yt_meta.get("duration") or 0
                        self.calculated_duration_exact = float(self.calculated_duration or 0)
                        if yt_meta.get("youtube") and not self.current_youtube:
                            self.current_youtube = yt_meta["youtube"]
                        if yt_meta.get("title") and not self.current_title:
                            self.current_title = yt_meta["title"]
                        if yt_meta.get("artist") and not self.current_artist:
                            self.current_artist = yt_meta["artist"]
                    else:
                        self.log("❌ File audio non trovato per l'analisi.")
                        win = self._get_window()
                        if win:
                            win.evaluate_js("if(window.autoOffsetFinished) window.autoOffsetFinished(null);")
                        return

                self.log("🔍 Avvio pre-analisi acustica reale per calibrazione auto-compensazione...")
                audio_raw = whisper.load_audio(audio_path)
                total_sec = float(len(audio_raw) / 16000.0)
                max_amp = np.max(np.abs(audio_raw))
                if max_amp > 1e-4:
                    audio_raw = audio_raw / max_amp

                vocal_audio, filtered_vocal = self._extract_vocal_stems(audio_path, audio_raw, usa_demucs, demucs_model=demucs_model)
                acoustic_engine = AcousticPrecisionEngine(vocal_audio)
                islands = acoustic_engine.detect_vocal_islands()

                # Seleziona una finestra attiva di 32s centrata sui primi versi cantati reali
                win_s = 0.0
                for ia, ib in islands:
                    if (ib - ia) >= 1.0:
                        win_s = max(0.0, ia - 0.5)
                        break
                voc_sec = float(len(vocal_audio) / 16000.0)
                win_e = min(voc_sec, max(min(voc_sec, 35.0), win_s + 32.0))
                if win_e - win_s < 4.0:
                    win_s, win_e = 0.0, min(voc_sec, 35.0)

                load_target = LOCAL_MODEL_TURBO if os.path.exists(LOCAL_MODEL_TURBO) else (
                    model_path if (model_path and os.path.exists(model_path)) else get_best_available_model_path()
                )
                self.log(f"🧠 Modello probe calibrazione: {os.path.basename(load_target)} | Finestra [{win_s:.1f}s→{win_e:.1f}s]")

                s_idx, e_idx = int(win_s * 16000), int(win_e * 16000)
                probe_audio = vocal_audio[s_idx:e_idx]
                filtered_vocal_probe = filtered_vocal[s_idx:e_idx] if filtered_vocal is not None else None
                lang_code = LANGUAGES.get(lang_label, "auto")
                lang_arg = None if lang_code == "auto" else lang_code

                try:
                    w_model = whisper.load_model(load_target, device=self.device)
                except Exception:
                    w_model = stable_ts.load_model(load_target, device=self.device)

                w_res = w_model.transcribe(
                    probe_audio,
                    language=lang_arg,
                    word_timestamps=True,
                    condition_on_previous_text=False,
                    temperature=0.0,
                    beam_size=2,
                    best_of=2,
                    no_speech_threshold=0.75,
                    fp16=(self.device == "cuda"),
                    verbose=False,
                )
                detected_l = (w_res.get("language") or lang_arg or "it")[:2].lower()

                probe_segs = []
                whisper_tokens = []
                for seg in w_res.get("segments", []):
                    txt0 = str(seg.get("text", "")).strip()
                    if not txt0 or _is_whisper_hallucination(txt0):
                        continue
                    s0 = win_s + float(seg.get("start", 0.0))
                    e0 = win_s + float(seg.get("end", 0.0))
                    seg_words = []
                    for w in seg.get("words", []) or []:
                        wt = str(w.get("word", "")).strip()
                        if wt:
                            ws_abs = win_s + float(w.get("start", 0.0))
                            we_abs = win_s + float(w.get("end", 0.0))
                            seg_words.append({"word": wt, "start": ws_abs, "end": we_abs})
                            whisper_tokens.append({
                                "start": ws_abs,
                                "end": we_abs,
                                "clean": clean_token_str(wt),
                            })
                    probe_segs.append({"start": s0, "end": e0, "text": txt0, "words": seg_words})

                del w_model
                w_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                whisper_offset_sec = calculate_auto_compensation_offset(
                    acoustic_engine, whisper_tokens, min_tokens=5, search_window=0.180, deadzone_ms=8
                )
                whisper_offset_ms = int(round(whisper_offset_sec * 1000))

                # Se testo compatibile e MMS_FA presente, misura il residuo effettivo del motore Ibrido CTC+Acustico
                clean_lines = []
                for raw_l in (lyrics_text or "").translate(_HOMOGLYPH_TRANS).splitlines():
                    s = raw_l.strip()
                    if not s or LRC_META_TAG_RE.match(s) or SECTION_HEADER_RE.match(s):
                        continue
                    s = re.sub(r"^\[\d{1,3}:\d{2}(?:\.\d+)?\]\s*", "", s).strip()
                    s = re.sub(r"[\[\{](?:agent:)?v\d+[\]\}]|\{bg\}", "", s, flags=re.IGNORECASE).strip()
                    if s and re.search(r"\w", s):
                        clean_lines.append({"text": s, "agent": "v1", "is_bg": s.startswith("(") and s.endswith(")")})

                offset_ms = whisper_offset_ms
                mode_label = "Whisper"
                if clean_lines and probe_segs and HAS_MMS_FA and is_latin_script_compatible("\n".join(x["text"] for x in clean_lines[:12])):
                    try:
                        # Allinea la sotto-finestra probe in coordinate locali e riporta i token a coordinate globali
                        probe_segs_local = [
                            {
                                "start": ps["start"] - win_s,
                                "end": ps["end"] - win_s,
                                "text": ps["text"],
                                "words": [
                                    {"word": sw["word"], "start": sw["start"] - win_s, "end": sw["end"] - win_s}
                                    for sw in ps.get("words", [])
                                ],
                            }
                            for ps in probe_segs
                        ]
                        ac_sub = AcousticPrecisionEngine(probe_audio)
                        ctc_probe = HybridCTCAligner.align(
                            vocal_audio_16k=probe_audio,
                            clean_lines_data=clean_lines[:24],
                            whisper_segments=probe_segs_local,
                            acoustic_engine=ac_sub,
                            detected_lang=detected_l,
                            device=self.device,
                            total_audio_sec=float(len(probe_audio) / 16000.0),
                            filtered_vocal_16k=filtered_vocal_probe,
                        )
                        ctc_tokens = [
                            {
                                "start": win_s + float(w["start"]),
                                "end": win_s + float(w["end"]),
                                "clean": clean_token_str(w["word"]),
                            }
                            for ln in ctc_probe
                            for w in ln.get("words", [])
                            if float(w["start"]) < (win_e - win_s - 0.5)
                        ]
                        ctc_offset_sec = calculate_auto_compensation_offset(
                            acoustic_engine, ctc_tokens, min_tokens=5, search_window=0.060, deadzone_ms=8
                        )
                        offset_ms = int(round(ctc_offset_sec * 1000))
                        mode_label = "Hybrid CTC + Acustico"
                    except Exception:
                        pass

                self.log(f"🎤 Bias Whisper grezzo: {whisper_offset_ms:+d}ms | 🎯 Calibrazione {mode_label}: {offset_ms:+d}ms")
                self.log(f"✨ ANALISI COMPLETATA: Compensazione ottimale = {offset_ms:+d}ms ({mode_label})")

                win = self._get_window()
                if win:
                    payload = json.dumps({
                        "success": True,
                        "offset_ms": offset_ms,
                        "whisper_offset_ms": whisper_offset_ms,
                        "mode": mode_label,
                        "offset_sec": round(offset_ms / 1000.0, 3),
                    })
                    win.evaluate_js(f"if(window.autoOffsetFinished) window.autoOffsetFinished({payload});")

            except Exception as e:
                traceback.print_exc()
                self.log(f"⚠️ Errore durante l'analisi: {str(e)}")
                win = self._get_window()
                if win:
                    win.evaluate_js("if(window.autoOffsetFinished) window.autoOffsetFinished(null);")
            finally:
                if w_model is not None:
                    del w_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

        threading.Thread(target=worker, daemon=True).start()

    # ==========================================================================
    # CORE ALIGNMENT PIPELINE (HYBRID CTC + ACOUSTIC 1MS + WHISPER FALLBACK)
    # ==========================================================================
    @staticmethod
    def _finalize_processed_lines(
        raw_lines_out: list,
        total_audio_sec: float,
        manual_offset_ms: float = 0.0,
        min_word_duration: float = 0.035,
    ) -> list:
        """Applica offset manuale, monotonicità parole e trim non-cascading tra righe principali."""
        manual_sec = float(manual_offset_ms) / 1000.0
        processed_lines = []
        last_main_line = None

        for item in raw_lines_out:
            words_data = []
            for w in item.get("words", []):
                ws = max(0.0, min(total_audio_sec - 0.05, float(w["start"]) + manual_sec))
                we = max(ws + min_word_duration, min(total_audio_sec - 0.01, float(w["end"]) + manual_sec))
                words_data.append({
                    "word": w["word"],
                    "start": round(ws, 3),
                    "end": round(we, 3),
                })

            if not words_data:
                continue

            words_data = smooth_and_clamp_words(words_data, min_word_dur=min_word_duration)
            line_start = max(0.0, min(total_audio_sec - 0.06, words_data[0]["start"]))
            line_end = max(line_start + min_word_duration, min(total_audio_sec - 0.01, words_data[-1]["end"]))
            words_data[0]["start"] = line_start
            words_data[-1]["end"] = line_end

            is_bg = bool(item.get("is_bg", False))
            if not is_bg and last_main_line is not None and last_main_line["end_s"] > line_start:
                if line_start > last_main_line["start_s"] + min_word_duration:
                    # Rifila la coda della riga precedente invece di spingere avanti la successiva (zero drift!)
                    trimmed_end = round(line_start, 3)
                    last_main_line["end_s"] = trimmed_end
                    prev_words = last_main_line["words"]
                    for pw in prev_words:
                        if pw["end"] > trimmed_end:
                            pw["end"] = trimmed_end
                        if pw["start"] >= pw["end"]:
                            pw["start"] = round(max(last_main_line["start_s"], trimmed_end - min_word_duration), 3)
                    last_main_line["words"] = smooth_and_clamp_words(prev_words, min_word_dur=min_word_duration)
                else:
                    line_start = round(min(total_audio_sec - 0.05, last_main_line["end_s"] + 0.02), 3)
                    words_data[0]["start"] = line_start
                    words_data = smooth_and_clamp_words(words_data, min_word_dur=min_word_duration)
                    line_end = max(line_start + min_word_duration, min(total_audio_sec - 0.01, words_data[-1]["end"]))
                    words_data[-1]["end"] = line_end

            pl = {
                "text": item["text"],
                "agent": item.get("agent", "v1"),
                "is_bg": is_bg,
                "start_s": round(line_start, 3),
                "end_s": round(line_end, 3),
                "words": words_data,
            }
            if not is_bg:
                last_main_line = pl
            processed_lines.append(pl)

        return processed_lines

    def _run_whisper_fallback_pipeline(
        self,
        vocal_audio: np.ndarray,
        acoustic_engine: AcousticPrecisionEngine,
        clean_lines_data: list,
        cleaned_text: str,
        load_target: str,
        lang_code: str,
        usa_vad: bool,
        usa_auto_offset: bool,
        params: dict,
        total_audio_sec: float,
    ) -> tuple:
        """Fallback Whisper DTW + micro-raffinatezza acustica (per lingue non-latine o assenza MMS_FA)."""
        model = stable_ts.load_model(load_target, device=self.device)
        if lang_code == "auto":
            self.log("🌐 Rilevamento automatico lingua principale sul vocal stem...")
            audio_30s = whisper.pad_or_trim(vocal_audio)
            n_mels = getattr(model.dims, "n_mels", 80)
            mel = whisper.log_mel_spectrogram(audio_30s, n_mels=n_mels).to(model.device)
            _, probs = model.detect_language(mel)
            lang_code = max(probs, key=probs.get)
            self.log(f"🔍 Lingua rilevata: {lang_code.upper()} (confidenza: {probs[lang_code]:.1%})")
        self.detected_language = lang_code

        align_kwargs = {
            "language": lang_code,
            "original_split": True,
            "fast_mode": False,
            "max_word_dur": 6.0,
            "suppress_silence": True,
            "suppress_word_ts": True,
            "gap_padding": " ...",
            "only_voice_freq": True,
        }
        if usa_vad:
            align_kwargs["vad"] = True
            align_kwargs["vad_threshold"] = 0.22

        self.log("⚡ Allineamento fonetico Whisper fallback in corso...")
        try:
            result = model.align(vocal_audio, cleaned_text, **align_kwargs)
        except TypeError as ex_type:
            self.log(f"⚠️ Parametro non standard rilevato ({ex_type}). Pulizia opzioni e retry...")
            for bad_key in ["time_precision", "only_voice_freq", "gap_padding", "suppress_word_ts", "fast_mode", "max_word_dur"]:
                align_kwargs.pop(bad_key, None)
            m = re.search(r"unexpected keyword argument(?:\(s\))?:\s*(.+)", str(ex_type))
            if m:
                for k in [x.strip().strip("'\"") for x in m.group(1).split(",")]:
                    align_kwargs.pop(k, None)
            try:
                result = model.align(vocal_audio, cleaned_text, **align_kwargs)
            except Exception:
                align_kwargs["original_split"] = False
                result = model.align(vocal_audio, cleaned_text, **align_kwargs)
        except Exception as ex_align:
            self.log(f"⚠️ Avviso stabilità ({ex_align}). Nuovo tentativo globale standard...")
            align_kwargs["original_split"] = False
            result = model.align(vocal_audio, cleaned_text, **align_kwargs)

        if params.get("neural_refine", False):
            self.log("🎯 Calibrazione neurale (Neural Refinement start + end)...")
            try:
                model.refine(
                    vocal_audio,
                    result,
                    steps="se",
                    precision=0.02,
                    rel_prob_decrease=0.20,
                    abs_prob_decrease=0.05,
                    only_voice_freq=True,
                )
            except Exception as ex_refine:
                self.log(f"⚠️ Neural Refine completato con fallback ({ex_refine})")

        raw_segments = list(result.segments)
        initial_tokens = []
        for seg in raw_segments:
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    w_txt = str(getattr(w, "word", "")).strip()
                    if w_txt:
                        initial_tokens.append({
                            "clean": clean_token_str(w_txt),
                            "start": float(w.start),
                            "end": float(w.end),
                        })

        auto_shift_sec = 0.0
        if usa_auto_offset:
            auto_shift_sec = calculate_auto_compensation_offset(
                acoustic_engine, initial_tokens, min_tokens=5, search_window=0.180, deadzone_ms=8
            )
            self._last_fallback_auto_ms = int(round(auto_shift_sec * 1000))
            self.log(f"🔍 Auto-Compensazione fallback rilevata: {self._last_fallback_auto_ms:+d}ms")
        else:
            self._last_fallback_auto_ms = 0

        can_use_1to1 = (len(raw_segments) == len(clean_lines_data))
        all_aligned_tokens = []
        for seg in raw_segments:
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    w_txt = str(getattr(w, "word", "")).strip()
                    if w_txt:
                        all_aligned_tokens.append({
                            "orig": w_txt,
                            "clean": clean_token_str(w_txt),
                            "start": max(0.0, min(total_audio_sec - 0.05, float(w.start))),
                            "end": max(0.0, min(total_audio_sec, float(w.end))),
                        })

        min_word_duration = 0.04
        draft_lines = []
        tok_cursor = 0
        num_global_tokens = len(all_aligned_tokens)

        for i, seg_info in enumerate(clean_lines_data):
            line_text = seg_info["text"]
            agent = seg_info["agent"]
            is_bg = seg_info["is_bg"]
            expected_words = line_text.split()
            words_data = []

            if can_use_1to1 and i < len(raw_segments) and getattr(raw_segments[i], "words", None):
                words_data = reconstruct_line_words(
                    raw_segments[i].words, line_text, offset_sec=0.0, max_audio_sec=total_audio_sec
                )
            else:
                window_size = max(len(expected_words) * 3, 24)
                tok_slice = all_aligned_tokens[tok_cursor : min(num_global_tokens, tok_cursor + window_size)]
                words_data, consumed = reconstruct_line_words(
                    tok_slice, line_text, offset_sec=0.0, max_audio_sec=total_audio_sec, return_consumed=True
                )
                tok_cursor += max(1, consumed)

            if not words_data:
                continue

            draft_lines.append({
                "text": line_text,
                "agent": agent,
                "is_bg": is_bg,
                "words": words_data,
            })

        prev_raw_limit = 0.0
        for d_line in draft_lines:
            is_bg = d_line["is_bg"]
            w_list = d_line["words"]
            for k in range(len(w_list)):
                raw_k_t0 = w_list[k]["start"]
                raw_k_end = w_list[k]["end"]
                if k == 0:
                    min_k_t = prev_raw_limit if not is_bg else 0.0
                else:
                    min_k_t = w_list[k - 1]["start"] + 0.025
                max_k_t = raw_k_end - 0.015

                if max_k_t > min_k_t:
                    shifted_t0 = max(min_k_t, min(max_k_t, raw_k_t0 + auto_shift_sec))
                    refined_k_t0 = acoustic_engine.refine_onset(
                        target_time=shifted_t0,
                        min_time=min_k_t,
                        max_time=max_k_t,
                        word_end=raw_k_end,
                    )
                    w_list[k]["start"] = round(max(min_k_t, refined_k_t0), 3)

                if w_list[k]["end"] <= w_list[k]["start"]:
                    w_list[k]["end"] = round(w_list[k]["start"] + min_word_duration, 3)

            if not is_bg and w_list:
                prev_raw_limit = max(prev_raw_limit, w_list[-1]["start"] + 0.04)

        pause_cuts_count = 0
        sustain_exts_count = 0
        for idx_l, d_line in enumerate(draft_lines):
            words_data = d_line["words"]
            next_line_first_raw_start = total_audio_sec - 0.05
            for next_idx in range(idx_l + 1, len(draft_lines)):
                cand_l = draft_lines[next_idx]
                if not cand_l["is_bg"] and cand_l["words"]:
                    next_line_first_raw_start = cand_l["words"][0]["start"]
                    break

            for k in range(len(words_data)):
                is_last_w = (k == len(words_data) - 1)
                w_start_raw = words_data[k]["start"]
                raw_k_end = words_data[k]["end"]
                min_k_end = w_start_raw + 0.038

                if not is_last_w:
                    max_k_end = words_data[k + 1]["start"] - 0.012
                else:
                    max_k_end = min(
                        total_audio_sec - 0.03,
                        next_line_first_raw_start - 0.025,
                        max(raw_k_end + 0.15, w_start_raw + 4.8),
                    )

                if max_k_end > min_k_end:
                    refined_k_end, action_tag = acoustic_engine.refine_offset_and_sustain(
                        word_start=w_start_raw,
                        target_end=raw_k_end,
                        min_end=min_k_end,
                        max_end=max_k_end,
                        word_text=words_data[k]["word"],
                        is_line_end=is_last_w,
                    )
                    words_data[k]["end"] = round(refined_k_end, 3)
                    if action_tag == "pause_cut":
                        pause_cuts_count += 1
                    elif action_tag == "sustain_ext":
                        sustain_exts_count += 1

        self.log(
            f"🎵 Analisi Dinamica Vocale (fallback): {pause_cuts_count} pause rifinite • {sustain_exts_count} tenute/melismi estesi."
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return draft_lines, lang_code

    def _execute_synchronization(self, params: dict) -> dict:
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr

        interceptor = RealtimeStreamInterceptor(self, orig_stderr)
        sys.stdout = interceptor
        sys.stderr = interceptor

        try:
            audio_path = params.get("audio_path")
            lyrics_text = params.get("lyrics_text", "")
            model_path = params.get("model_path", "")
            lang_label = params.get("language")
            usa_demucs = params.get("demucs", False)
            usa_vad = params.get("vad", False)
            usa_auto_offset = params.get("auto_offset", True)
            formato = params.get("format", "metrolist")
            manual_offset_ms = float(params.get("offset", 0))

            if not audio_path or not os.path.exists(audio_path):
                yt_source = (params.get("youtube_audio_url") or params.get("youtube") or "").strip()
                if yt_source and is_youtube_audio_source(yt_source):
                    yt_res = download_youtube_audio_for_sync(yt_source, log_func=self.log, progress_cb=self.progress)
                    audio_path = yt_res["audio_path"]
                    self.last_audio_path = audio_path
                    self._is_yt_temp_audio = True
                    self._yt_output_basename = yt_res["base_name"]
                    yt_meta = yt_res.get("meta") or {}
                    for mk in ("title", "artist", "album", "youtube", "songwriters"):
                        if not params.get(mk) and yt_meta.get(mk):
                            params[mk] = yt_meta[mk]
                else:
                    self.log("❌ Errore: File audio non valido!")
                    win = self._get_window()
                    if win:
                        win.evaluate_js("window.syncFinished(false);")
                    return {"success": False, "error": "File audio non valido"}

            if self._is_yt_temp_audio:
                nome_base = self._yt_output_basename or self._safe_base_filename({
                    "title": params.get("title"),
                    "artist": params.get("artist"),
                })
            else:
                nome_base = os.path.splitext(os.path.basename(audio_path))[0]

            lang_code = LANGUAGES.get(lang_label, "auto")

            self.log(
                f"--- AVVIO SINCRONIZZAZIONE ULTRA-PRECISA v4 [{formato.upper()}] ---\nHardware: {self.device.upper()} ({self.gpu_name})"
            )
            self.progress(5)

            audio_raw = whisper.load_audio(audio_path)
            total_audio_sec = float(len(audio_raw) / 16000.0)
            self.calculated_duration = int(round(total_audio_sec))

            max_amp = np.max(np.abs(audio_raw))
            if max_amp > 1e-4:
                audio_raw = audio_raw / max_amp

            load_target = model_path
            if not os.path.exists(load_target) and (not load_target or load_target.endswith(".pt")):
                load_target = get_best_available_model_path()
                self.log(f"⚠️ File modello non trovato, uso modello locale ottimale: {load_target}")

            raw_outro_known = _normalize_outro_candidates([
                params.get("outro_text"),
                self.current_outro_text,
                *self._last_outro_candidates,
            ])
            raw_lines = lyrics_text.translate(_HOMOGLYPH_TRANS).splitlines()
            current_agent = "v1"
            clean_lines_data = []

            for line in raw_lines:
                s = line.strip()
                if not s or (s.startswith("<") and s.endswith(">")) or LRC_META_TAG_RE.match(s):
                    continue

                s = re.sub(r"^\[\d{1,3}:\d{2}(?:\.\d+)?\]\s*", "", s).strip()
                if not s or LRC_META_TAG_RE.match(s):
                    continue

                if SECTION_HEADER_RE.match(s) or _is_outro_credit_text(s, raw_outro_known):
                    continue

                m_sec = re.fullmatch(r"[\[\{](?:agent:)?(v\d+)[\]\}]", s, re.IGNORECASE)
                if m_sec:
                    current_agent = m_sec.group(1).lower()
                    continue

                m_inline = re.match(r"^[\[\{](?:agent:)?(v\d+)[\]\}]\s*(.*)", s, re.IGNORECASE)
                if m_inline:
                    line_agent = m_inline.group(1).lower()
                    text_part = m_inline.group(2).strip()
                else:
                    line_agent = current_agent
                    text_part = s

                is_explicit_bg = "{bg}" in text_part.lower()
                text_part = re.sub(r"\{(?:agent:)?v\d+\}", "", text_part, flags=re.IGNORECASE)
                text_part = re.sub(r"\{bg\}", "", text_part, flags=re.IGNORECASE).strip()
                text_part = " ".join(text_part.split())

                if not text_part or not re.search(r"\w", text_part):
                    continue

                text_part = re.sub(r"[\(\[\{]\s*[xX]?\d+\s*[xX]?\s*[\)\]\}]", "", text_part).strip()
                is_bg = (text_part.startswith("(") and text_part.endswith(")")) or is_explicit_bg

                clean_lines_data.append({
                    "text": text_part,
                    "agent": line_agent,
                    "is_bg": is_bg,
                })

            if not clean_lines_data:
                raise Exception("Nessun testo valido contenente parole cantate trovato.")

            cleaned_text = "\n".join([item["text"] for item in clean_lines_data])
            self.plain_lyrics = cleaned_text

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            vocal_audio, filtered_vocal = self._extract_vocal_stems(
                audio_path, audio_raw, usa_demucs, progress_cb=self.progress,
                demucs_model=params.get("demucs_model") or load_config().get("demucs_model", "htdemucs"),
            )

            self.progress(30)
            self.log("🔬 Inizializzazione motore acustico v4 (Multi-Band Transient, Vocal Islands & Sustain Engine)...")
            acoustic_engine = AcousticPrecisionEngine(vocal_audio)

            min_word_duration = 0.035
            PAUSA_SOGLIA = 1.4
            CODA_VOCE = 0.12

            raw_aligned_lines = None
            ckpt_mms = os.path.join(torch.hub.get_dir(), "checkpoints", "model.pt")
            use_ctc = HybridCTCAligner.can_run_ctc(cleaned_text) and os.path.exists(ckpt_mms)

            if use_ctc:
                try:
                    # Preferisci large-v3-turbo locale per macro-ancoraggio rapido (10-18s vs >3min)
                    anchor_target = load_target
                    if os.path.exists(LOCAL_MODEL_TURBO) and (
                        load_target in (LOCAL_MODEL_V3, LOCAL_MODEL_V2, "large-v3", "large-v2")
                        or os.path.basename(str(load_target)) in ("large-v3.pt", "large-v2.pt")
                    ):
                        anchor_target = LOCAL_MODEL_TURBO
                        self.log("⚡ Modalità Ibrida Neural CTC: Macro-ancoraggio Turbo + Allineatore Fonetico MMS_FA (20ms/1ms)")
                    else:
                        self.log(f"⚡ Modalità Ibrida Neural CTC: Macro-ancoraggio ({os.path.basename(str(anchor_target))}) + Allineatore Fonetico MMS_FA")

                    self.progress(42)
                    try:
                        w_model = whisper.load_model(anchor_target, device=self.device)
                    except Exception:
                        w_model = stable_ts.load_model(anchor_target, device=self.device)

                    w_res = w_model.transcribe(
                        vocal_audio,
                        language=(None if lang_code == "auto" else lang_code),
                        word_timestamps=True,
                        condition_on_previous_text=False,
                        temperature=0.0,
                        beam_size=2,
                        best_of=2,
                        compression_ratio_threshold=2.4,
                        no_speech_threshold=0.65,
                        fp16=(self.device == "cuda"),
                        verbose=False,
                    )
                    if lang_code == "auto":
                        lang_code = (w_res.get("language") or "it")[:2].lower()
                        self.log(f"🔍 Lingua rilevata sul brano completo: {lang_code.upper()}")
                    else:
                        self.log(f"Lingua impostata: {lang_code.upper()}")
                    self.detected_language = lang_code

                    w_segs = list(w_res.get("segments", []))
                    # Recupero automatico finestre saltate da Whisper e suddivisione segmenti sparsi multi-isola
                    try:
                        islands_w = acoustic_engine.detect_vocal_islands()

                        def _has_island_overlap(t0: float, t1: float) -> bool:
                            return any(isl[1] > t0 + 0.05 and isl[0] < t1 - 0.05 for isl in islands_w)

                        def _cluster_islands(isls, gap_thresh: float = 0.28, max_span: float = 6.2):
                            clusters = []
                            for a, b in isls:
                                if not clusters:
                                    clusters.append([a, b])
                                    continue
                                pa, pb = clusters[-1]
                                if (a - pb) < gap_thresh and (b - pa) <= max_span:
                                    clusters[-1][1] = b
                                elif (b - pa) <= max_span and ((pb - pa) < 0.85 or (b - a) < 0.70):
                                    clusters[-1][1] = b
                                else:
                                    clusters.append([a, b])
                            return clusters

                        def _transcribe_slice(t_a: float, t_b: float):
                            if t_b - t_a < 0.6:
                                return []
                            sub_wav = vocal_audio[int(t_a * 16000):int(t_b * 16000)]
                            if len(sub_wav) < 1600:
                                return []
                            sub_peak = float(np.max(np.abs(sub_wav)))
                            if 0.008 < sub_peak < 0.85:
                                gain = min(6.0, 0.85 / sub_peak)
                                sub_wav = (sub_wav * gain).astype(np.float32)
                            rr = w_model.transcribe(
                                sub_wav,
                                language=lang_code,
                                word_timestamps=True,
                                condition_on_previous_text=False,
                                temperature=0.0,
                                beam_size=2,
                                best_of=2,
                                compression_ratio_threshold=2.4,
                                no_speech_threshold=0.85,
                                logprob_threshold=None,
                                fp16=(self.device == "cuda"),
                                verbose=False,
                            )
                            out_ss = []
                            for ss in rr.get("segments", []):
                                txt_s = str(ss.get("text", "")).strip()
                                if not txt_s or _is_whisper_hallucination(txt_s):
                                    continue
                                ns = t_a + float(ss.get("start", 0.0))
                                ne = t_a + float(ss.get("end", 0.0))
                                if _has_island_overlap(ns, ne):
                                    sw_list = []
                                    for sw in ss.get("words", []) or []:
                                        wt = str(sw.get("word", "")).strip()
                                        if wt:
                                            sw_list.append({
                                                "word": wt,
                                                "start": t_a + float(sw.get("start", 0.0)),
                                                "end": t_a + float(sw.get("end", 0.0)),
                                            })
                                    out_ss.append({"start": ns, "end": ne, "text": txt_s, "words": sw_list})
                            return out_ss

                        def _transcribe_clusters(clusters, min_len: float = 0.9):
                            out_all = []
                            for ci, (ca, cb) in enumerate(clusters):
                                pad_l = 0.28 if ci == 0 else min(0.28, max(0.04, (ca - clusters[ci - 1][1]) * 0.45))
                                pad_r = 0.28 if ci == len(clusters) - 1 else min(0.28, max(0.04, (clusters[ci + 1][0] - cb) * 0.45))
                                s0 = max(0.0, ca - pad_l)
                                s1 = min(total_audio_sec, cb + pad_r)
                                if s1 - s0 >= min_len:
                                    out_all.extend(_transcribe_slice(s0, s1))
                            return out_all

                        w_segs = [
                            s for s in w_segs
                            if not _is_whisper_hallucination(str(s.get("text", "")))
                            and _has_island_overlap(float(s.get("start", 0.0)), float(s.get("end", 0.0)))
                        ]
                        for _pass in range(2):
                            w_segs.sort(key=lambda x: float(x.get("start", 0.0)))
                            gap_spans = []
                            if not w_segs:
                                gap_spans.append((0.0, total_audio_sec))
                            else:
                                if float(w_segs[0]["start"]) >= 3.8:
                                    gap_spans.append((0.0, float(w_segs[0]["start"])))
                                for gi in range(len(w_segs) - 1):
                                    g0 = float(w_segs[gi]["end"])
                                    g1 = float(w_segs[gi + 1]["start"])
                                    if g1 - g0 >= 3.8:
                                        gap_spans.append((g0, g1))
                                if total_audio_sec - float(w_segs[-1]["end"]) >= 3.8:
                                    gap_spans.append((float(w_segs[-1]["end"]), total_audio_sec))
                            recovered_any = False
                            for gs, ge in gap_spans:
                                act_isls = [(max(gs, a), min(ge + 0.30, b)) for a, b in islands_w if b > gs + 0.35 and a < ge - 0.35]
                                act_dur = sum(b - a for a, b in act_isls)
                                if act_dur >= 1.6 and act_isls:
                                    new_ss = _transcribe_clusters(_cluster_islands(act_isls), min_len=0.9)
                                    if new_ss:
                                        w_segs.extend(new_ss)
                                        recovered_any = True
                            if not recovered_any:
                                break

                        w_segs.sort(key=lambda x: float(x.get("start", 0.0)))
                        refined_segs = []
                        for s in w_segs:
                            s0, s1 = float(s.get("start", 0.0)), float(s.get("end", 0.0))
                            n_w = len([t for t in str(s.get("text", "")).split() if _clean_phonetic_ctc(t, lang_code)])
                            ov_isls = [
                                (max(s0, a), min(s1 + 0.70, b)) for a, b in islands_w
                                if min(s1, b) - max(s0, a) >= 0.45
                            ]
                            sub_clusters = _cluster_islands(ov_isls, gap_thresh=0.30, max_span=5.5) if (
                                s1 - s0 >= 5.5 and (s1 - s0) / max(1, n_w) >= 0.65 and len(ov_isls) >= 2
                            ) else []
                            if len(sub_clusters) >= 2:
                                split_ok = _transcribe_clusters(sub_clusters, min_len=0.7)
                                if split_ok and sum(len(x["text"].split()) for x in split_ok) >= max(2, int(n_w * 0.60)):
                                    refined_segs.extend(split_ok)
                                    continue
                            refined_segs.append(s)
                        w_segs = refined_segs
                        w_segs.sort(key=lambda x: float(x.get("start", 0.0)))
                    except Exception:
                        pass

                    del w_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

                    self.progress(62)
                    self.log("🎯 Forced Alignment CTC frame-by-frame (20ms) + aggancio transiente acustico (1ms)...")
                    raw_aligned_lines = HybridCTCAligner.align(
                        vocal_audio_16k=vocal_audio,
                        clean_lines_data=clean_lines_data,
                        whisper_segments=w_segs,
                        acoustic_engine=acoustic_engine,
                        detected_lang=lang_code,
                        device=self.device,
                        total_audio_sec=total_audio_sec,
                        progress_cb=self.progress,
                        filtered_vocal_16k=filtered_vocal,
                    )
                    auto_offset_ms = 0
                    if usa_auto_offset and raw_aligned_lines:
                        self.log("🎯 Auto-Compensazione attiva: allineamento neurale CTC + transienti 1ms a fase zero (0ms)")
                    if manual_offset_ms:
                        self.log(f"⚙️ Compensazione manuale applicata: {manual_offset_ms:+0.0f}ms")
                except Exception as ex_ctc:
                    self.log(f"⚠️ Errore motore CTC ({ex_ctc}), passo automaticamente al fallback Whisper...")
                    traceback.print_exc()
                    raw_aligned_lines = None
                    auto_offset_ms = 0

            if raw_aligned_lines is None:
                raw_aligned_lines, lang_code = self._run_whisper_fallback_pipeline(
                    vocal_audio=vocal_audio,
                    acoustic_engine=acoustic_engine,
                    clean_lines_data=clean_lines_data,
                    cleaned_text=cleaned_text,
                    load_target=load_target,
                    lang_code=lang_code,
                    usa_vad=usa_vad,
                    usa_auto_offset=usa_auto_offset,
                    params=params,
                    total_audio_sec=total_audio_sec,
                )
                auto_offset_ms = getattr(self, "_last_fallback_auto_ms", 0)
                if manual_offset_ms:
                    self.log(f"⚙️ Compensazione manuale applicata: {manual_offset_ms:+0.0f}ms")
                effective_offset_ms = float(manual_offset_ms)
            else:
                effective_offset_ms = float(manual_offset_ms) + float(auto_offset_ms)

            processed_lines = self._finalize_processed_lines(
                raw_aligned_lines,
                total_audio_sec=total_audio_sec,
                manual_offset_ms=effective_offset_ms,
                min_word_duration=min_word_duration,
            )

            output_lines = []
            structured_lines = []
            max_detected_time = 0.0

            for i, pl in enumerate(processed_lines):
                line_text = pl["text"]
                agent = pl["agent"]
                is_bg = pl["is_bg"]
                seg_start = pl["start_s"]
                seg_end = pl["end_s"]

                if seg_end > max_detected_time:
                    max_detected_time = seg_end

                words_formatted = []
                line_words = []

                for w in pl["words"]:
                    words_formatted.append(
                        f"{w['word']}:{format_word_time(w['start'])}:{format_word_time(w['end'])}"
                    )
                    line_words.append({
                        "word": w["word"],
                        "start_ms": int(round(w["start"] * 1000)),
                        "end_ms": int(round(w["end"] * 1000)),
                    })

                start_ts = format_lrc_time(seg_start)
                tag_prefix = "{bg}" if is_bg else f"{{agent:{agent}}}"

                output_lines.append(f"{start_ts}{tag_prefix}{line_text}")
                if words_formatted:
                    output_lines.append("<" + "|".join(words_formatted) + ">")

                structured_lines.append({
                    "text": line_text,
                    "agent": agent,
                    "is_bg": is_bg,
                    "start_ms": int(round(seg_start * 1000)),
                    "end_ms": int(round(seg_end * 1000)),
                    "words": line_words,
                })

                if formato == "normale" and not is_bg and (i + 1 < len(processed_lines)):
                    next_pl = processed_lines[i + 1]
                    if not next_pl["is_bg"] and (next_pl["start_s"] - seg_end) >= PAUSA_SOGLIA:
                        break_ts = seg_end + CODA_VOCE
                        if break_ts < total_audio_sec:
                            output_lines.append(f"{format_lrc_time(break_ts)}")

            self.calculated_duration = int(round(total_audio_sec))
            self.calculated_duration_exact = total_audio_sec
            norm_audio = os.path.normcase(os.path.abspath(audio_path))
            prev_audio = os.path.normcase(os.path.abspath(self.last_audio_path)) if self.last_audio_path else ""
            same_audio = bool(prev_audio and norm_audio == prev_audio)
            self.last_audio_path = audio_path

            audio_meta = extract_audio_metadata(audio_path)
            lyr_meta = extract_lyrics_metadata(lyrics_text)

            title = (params.get("title") or (self.current_title if same_audio else "") or audio_meta.get("title") or lyr_meta.get("title") or nome_base).strip()
            artist = (params.get("artist") or (self.current_artist if same_audio else "") or audio_meta.get("artist") or lyr_meta.get("artist") or "").strip()

            def _pick_optional(key: str, current_val: str, *fallbacks: str) -> str:
                if key in params and params.get(key) is not None:
                    return str(params[key]).strip()
                if same_audio and current_val:
                    return current_val.strip()
                for fb in fallbacks:
                    if fb and str(fb).strip():
                        return str(fb).strip()
                return ""

            album = _pick_optional("album", self.current_album, audio_meta.get("album"), lyr_meta.get("album"))
            youtube = _pick_optional("youtube", self.current_youtube, audio_meta.get("youtube"))
            songwriters = _pick_optional("songwriters", self.current_songwriters, lyr_meta.get("songwriters"), audio_meta.get("songwriters"))
            notes = _pick_optional("notes", self.current_notes, lyr_meta.get("notes"))
            outro_text = _pick_optional("outro_text", self.current_outro_text if same_audio else "")
            outro_credits = params.get("outro_credits")
            if outro_credits is None:
                outro_credits = self.current_outro_credits
            else:
                self.current_outro_credits = bool(outro_credits)

            self.current_title = title
            self.current_artist = artist
            self.current_album = album
            self.current_youtube = youtube
            self.current_songwriters = songwriters
            self.current_notes = notes
            self.current_outro_text = outro_text

            self._rebuild_outputs_from_structured(
                structured=structured_lines,
                title=title,
                artist=artist,
                album=album,
                duration_sec=self.calculated_duration,
                language=lang_code,
                songwriters=songwriters,
                notes=notes,
                plain_text=self.plain_lyrics,
                preserve_body_lrc="\n".join(output_lines),
                include_headers=True,
                outro_credits=bool(outro_credits),
                outro_text=outro_text,
                exact_duration_sec=total_audio_sec,
            )

            out_base = _build_song_folder_name(title=title, artist=artist, fallback=nome_base)
            cartella = params.get("output_dir") or _get_song_output_dir(title=title, artist=artist, fallback=nome_base)
            os.makedirs(cartella, exist_ok=True)
            output_lrc = os.path.join(cartella, f"{out_base}.lrc")
            output_lyricsfile = os.path.join(cartella, f"{out_base}.lyricsfile.yaml")
            output_ttml = os.path.join(cartella, f"{out_base}.ttml")

            self.last_output_paths = {
                "lrc": output_lrc,
                "yaml": output_lyricsfile,
                "ttml": output_ttml,
            }
            self._write_synced_output_files(output_lrc, output_lyricsfile, output_ttml)

            self.progress(100)
            self.log("\n✅ SINCRONIZZAZIONE COMPLETATA CON SUCCESSO!")
            self.log(f"⏱️ Durata traccia: {self.calculated_duration}s ({len(processed_lines)} righe)")
            self.log(f"📁 LRC: {output_lrc}")
            self.log(f"📁 Lyricsfile YAML: {output_lyricsfile}")
            self.log(f"📁 TTML: {output_ttml}")

            meta_out = {
                "title": title,
                "artist": artist,
                "album": album,
                "youtube": youtube,
                "songwriters": songwriters,
                "notes": notes,
                "outro_text": outro_text,
                "language": lang_code,
                "duration": self.calculated_duration,
            }

            win = self._get_window()
            if win:
                payload = json.dumps({
                    "audio_path": audio_path,
                    "lrc": self.generated_lrc,
                    "lyricsfile": self.generated_lyricsfile,
                    "ttml": self.generated_ttml,
                    "duration": self.calculated_duration,
                    "auto_offset_ms": auto_offset_ms,
                    "meta": meta_out,
                })
                win.evaluate_js(f"window.syncFinished(true, {payload});")

            return {
                "success": True,
                "lrc_path": output_lrc,
                "yaml_path": output_lyricsfile,
                "ttml_path": output_ttml,
                "structured_lines": structured_lines,
                "duration": self.calculated_duration,
                "auto_offset_ms": auto_offset_ms,
                "language": lang_code,
                "meta": meta_out,
            }

        except Exception as e:
            traceback.print_exc()
            self.log(f"\n❌ ERRORE: {str(e)}")
            win = self._get_window()
            if win:
                win.evaluate_js("window.syncFinished(false);")
            return {"success": False, "error": str(e)}
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    def start_synchronization(self, params):
        threading.Thread(target=lambda: self._execute_synchronization(params), daemon=True).start()

    def publish_lrclib(self, metadata):
        def worker():
            track_name = metadata.get("title", "").strip()
            artist_name = metadata.get("artist", "").strip()
            album_name = metadata.get("album", "").strip()
            songwriters = metadata.get("songwriters", self.current_songwriters).strip()
            notes = metadata.get("notes", self.current_notes).strip()
            outro_text = metadata.get("outro_text", self.current_outro_text).strip()
            duration = int(metadata.get("duration", self.calculated_duration))
            lang_code = _normalize_lang_code(metadata.get("language", "")) or self.detected_language
            if "outro_credits" in metadata and metadata.get("outro_credits") is not None:
                self.current_outro_credits = bool(metadata.get("outro_credits"))

            self.current_title = track_name
            self.current_artist = artist_name
            self.current_album = album_name
            self.current_songwriters = songwriters
            self.current_notes = notes
            self.current_outro_text = outro_text

            if not track_name or not artist_name:
                win = self._get_window()
                if win:
                    err_res = json.dumps({
                        "success": False,
                        "detail": "Titolo e Artista sono obbligatori per LRCLIB!"
                    })
                    win.evaluate_js(f"window.publishFinished({err_res});")
                return

            if duration < 1:
                win = self._get_window()
                if win:
                    err_res = json.dumps({
                        "success": False,
                        "detail": "Durata traccia non valida."
                    })
                    win.evaluate_js(f"window.publishFinished({err_res});")
                return

            structured = parse_lrc_to_structured_lines(self.generated_lrc) if self.generated_lrc else []
            if structured:
                self._rebuild_outputs_from_structured(
                    structured,
                    title=track_name,
                    artist=artist_name,
                    album=album_name,
                    duration_sec=duration,
                    language=lang_code,
                    songwriters=songwriters,
                    notes=notes,
                    plain_text=self.plain_lyrics,
                    preserve_body_lrc=self.generated_lrc,
                    include_headers=True,
                    outro_credits=self.current_outro_credits,
                    outro_text=outro_text,
                    exact_duration_sec=self.calculated_duration_exact,
                )

            cartella = _get_song_output_dir(title=track_name, artist=artist_name, fallback=f"{artist_name} - {track_name}")
            nome_base = _build_song_folder_name(title=track_name, artist=artist_name)

            def log_ui(msg):
                self.log(msg)

            success, track_id, path, detail = publish_and_verify_lrclib(
                track_name=track_name,
                artist_name=artist_name,
                album_name=album_name,
                duration=duration,
                plain_lyrics=self.plain_lyrics,
                synced_lyrics=self.generated_lrc,
                lyricsfile_yaml=self.generated_lyricsfile,
                cartella_output=cartella,
                nome_base=nome_base,
                log_func=log_ui,
            )

            win = self._get_window()
            if win:
                res = json.dumps({
                    "success": success,
                    "track_id": track_id,
                    "download_path": path,
                    "detail": detail,
                })
                win.evaluate_js(f"window.publishFinished({res});")

        threading.Thread(target=worker, daemon=True).start()

    def publish_betterlyrics(self, metadata):
        def worker():
            video_input = metadata.get("youtube", "").strip()
            track_name = metadata.get("title", "").strip()
            artist_name = metadata.get("artist", "").strip()
            album_name = metadata.get("album", "").strip()
            songwriters = metadata.get("songwriters", self.current_songwriters).strip()
            notes = metadata.get("notes", self.current_notes).strip()
            outro_text = metadata.get("outro_text", self.current_outro_text).strip()
            duration = int(metadata.get("duration", self.calculated_duration))
            lang_code = _normalize_lang_code(metadata.get("language", "")) or self.detected_language
            if "outro_credits" in metadata and metadata.get("outro_credits") is not None:
                self.current_outro_credits = bool(metadata.get("outro_credits"))

            self.current_title = track_name
            self.current_artist = artist_name
            self.current_album = album_name
            self.current_youtube = video_input
            self.current_songwriters = songwriters
            self.current_notes = notes
            self.current_outro_text = outro_text

            if not video_input:
                win = self._get_window()
                if win:
                    err_res = json.dumps({
                        "success": False,
                        "detail": "Inserisci il link o l'ID del video di YouTube Music!"
                    })
                    win.evaluate_js(f"window.publishBetterLyricsFinished({err_res});")
                return

            if not track_name or not artist_name:
                win = self._get_window()
                if win:
                    err_res = json.dumps({
                        "success": False,
                        "detail": "Titolo e Artista sono obbligatori!"
                    })
                    win.evaluate_js(f"window.publishBetterLyricsFinished({err_res});")
                return

            structured = parse_lrc_to_structured_lines(self.generated_lrc) if self.generated_lrc else []
            if not structured and self.generated_ttml:
                structured = parse_ttml_to_structured_lines(self.generated_ttml)
            if structured:
                self._rebuild_outputs_from_structured(
                    structured,
                    title=track_name,
                    artist=artist_name,
                    album=album_name,
                    duration_sec=duration,
                    language=lang_code,
                    songwriters=songwriters,
                    notes=notes,
                    plain_text=self.plain_lyrics,
                    preserve_body_lrc=self.generated_lrc or None,
                    include_headers=True,
                    outro_credits=self.current_outro_credits,
                    outro_text=outro_text,
                    exact_duration_sec=self.calculated_duration_exact,
                )

            if not self.generated_ttml or not self.generated_ttml.strip():
                win = self._get_window()
                if win:
                    err_res = json.dumps({
                        "success": False,
                        "detail": "Nessun testo sincronizzato disponibile. Clicca su 'AVVIA SYNC' prima di pubblicare!"
                    })
                    win.evaluate_js(f"window.publishBetterLyricsFinished({err_res});")
                return

            cartella = _get_song_output_dir(title=track_name, artist=artist_name, fallback=f"{artist_name} - {track_name}")
            nome_base = _build_song_folder_name(title=track_name, artist=artist_name)

            def log_ui(msg):
                self.log(msg)

            success, vid, path, detail = publish_and_verify_betterlyrics(
                video_id=video_input,
                track_name=track_name,
                artist_name=artist_name,
                album_name=album_name,
                duration=duration,
                ttml_lyrics=self.generated_ttml,
                cartella_output=cartella,
                nome_base=nome_base,
                language=self.detected_language,
                log_func=log_ui,
            )

            win = self._get_window()
            if win:
                res = json.dumps({
                    "success": success,
                    "video_id": vid,
                    "download_path": path,
                    "detail": detail,
                })
                win.evaluate_js(f"window.publishBetterLyricsFinished({res});")

        threading.Thread(target=worker, daemon=True).start()

    # ==========================================================================
    # AUTENTICAZIONE SERVIZI DI PUBBLICAZIONE & IDENTITÀ UNISON
    # ==========================================================================
    def get_publishing_settings(self):
        return {
            "success": True,
            "publishing": get_publishing_config(),
            "identity": get_unison_identity_status(),
        }

    def save_publishing_settings(self, pub_cfg):
        cfg = load_config()
        merged_pub = get_publishing_config(pub_cfg if isinstance(pub_cfg, dict) else None)
        cfg["publishing"] = merged_pub
        ok = save_config(cfg)
        return {
            "success": bool(ok),
            "publishing": merged_pub,
            "identity": get_unison_identity_status(),
        }

    def reset_publishing_settings(self):
        cfg = load_config()
        cfg["publishing"] = dict(DEFAULT_PUBLISHING_CONFIG)
        save_config(cfg)
        return {
            "success": True,
            "publishing": get_publishing_config(),
            "identity": get_unison_identity_status(),
        }

    def generate_unison_identity(self, label: str = ""):
        try:
            generate_new_unison_identity()
            if label is not None:
                cfg = load_config()
                pub = get_publishing_config({"unison_label": str(label).strip()})
                cfg["publishing"] = pub
                save_config(cfg)
            return {
                "success": True,
                "identity": get_unison_identity_status(),
                "publishing": get_publishing_config(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def import_unison_identity_file(self):
        win = self._get_window()
        if not win:
            return {"success": False, "error": "Finestra non disponibile."}
        file_types = ("Identità Unison / Chiave ECDSA (*.json;*.pem)",)
        result = win.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=False,
            file_types=file_types,
        )
        if not result:
            return None
        file_path = result[0]
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read()
            bundle = parse_unison_identity_input(raw)
            save_unison_identity_bundle(bundle)
            return {
                "success": True,
                "path": file_path,
                "identity": get_unison_identity_status(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def import_unison_identity_text(self, raw_text: str, label: str = ""):
        try:
            bundle = parse_unison_identity_input(raw_text)
            save_unison_identity_bundle(bundle)
            if label is not None:
                cfg = load_config()
                cfg["publishing"] = get_publishing_config({"unison_label": str(label).strip()})
                save_config(cfg)
            return {
                "success": True,
                "identity": get_unison_identity_status(),
                "publishing": get_publishing_config(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def export_unison_identity_file(self, fmt: str = "metrosync"):
        win = self._get_window()
        if not win:
            return {"success": False, "error": "Finestra non disponibile."}
        if not os.path.exists(UNISON_KEY_FILE):
            return {"success": False, "error": "Nessuna identità Unison salvata da esportare."}
        try:
            with open(UNISON_KEY_FILE, "r", encoding="utf-8") as f:
                bundle = parse_unison_identity_input(f.read())
        except Exception as e:
            return {"success": False, "error": f"Identità locale non valida: {e}"}

        def_name = f"unison-identity-{bundle['key_id'][:8]}.json"
        result = win.create_file_dialog(
            webview.FileDialog.SAVE,
            save_filename=def_name,
            file_types=("File JSON (*.json)",),
        )
        if not result:
            return None
        save_path = result[0] if isinstance(result, (list, tuple)) else result
        try:
            if fmt == "betterlyrics_jwk":
                export_obj = {
                    "keyId": bundle["key_id"],
                    "publicKey": bundle["public_jwk"],
                    "privateKey": bundle["private_jwk"],
                }
            else:
                export_obj = {
                    "key_id": bundle["key_id"],
                    "private_key_pem": bundle["private_key_pem"],
                    "public_jwk": bundle["public_jwk"],
                    "private_jwk": bundle["private_jwk"],
                }
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(export_obj, f, indent=2)
            return {"success": True, "path": save_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_unison_identity(self):
        try:
            if os.path.exists(UNISON_KEY_FILE):
                os.remove(UNISON_KEY_FILE)
            return {"success": True, "identity": get_unison_identity_status()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_publishing_services(self, pub_override=None):
        pub_cfg = get_publishing_config(pub_override if isinstance(pub_override, dict) else None)
        report = {"success": True, "lrclib": {}, "unison": {}}

        try:
            l_headers = {
                "User-Agent": pub_cfg["lrclib_client_id"],
                "X-User-Agent": pub_cfg["lrclib_client_id"],
                "Lrclib-Client": pub_cfg["lrclib_client_id"],
            }
            if pub_cfg.get("lrclib_auth_mode") == "bearer_auth" and pub_cfg.get("lrclib_token"):
                l_headers["Authorization"] = f"Bearer {pub_cfg['lrclib_token']}"
            _http_json_get(f"{pub_cfg['lrclib_url']}/api/search?q=test", headers=l_headers, timeout=8)
            report["lrclib"] = {
                "ok": True,
                "message": f"Connesso ({pub_cfg['lrclib_auth_mode']})",
            }
        except Exception as ex:
            report["lrclib"] = {"ok": False, "message": str(ex)}

        id_status = get_unison_identity_status()
        try:
            u_headers = {"User-Agent": pub_cfg["unison_client_id"]}
            if pub_cfg.get("unison_auth_mode") == "ecdsa_plus_bearer" and pub_cfg.get("unison_bearer_token"):
                u_headers["Authorization"] = f"Bearer {pub_cfg['unison_bearer_token']}"
            _http_json_get(f"{pub_cfg['unison_url']}/lyrics/search?q=test", headers=u_headers, timeout=8)
            if id_status.get("valid"):
                key_msg = f"Chiave attiva ({id_status['key_id'][:10]}...)"
            else:
                key_msg = "Server OK (chiave ECDSA generata al primo invio)"
            report["unison"] = {"ok": True, "message": key_msg}
        except Exception as ex:
            report["unison"] = {"ok": False, "message": str(ex)}

        return report

    # ==========================================================================
    # RICERCA ONLINE & DOWNLOAD LOCALE FORMATO METROSYNC / METROLIST
    # ==========================================================================
    def _build_converted_lyrics_package(self, fetched: dict, form_meta: dict = None, formato: str = None) -> dict:
        form_meta = form_meta or {}
        structured, plain_text, sync_level, l_meta = parse_downloaded_lyrics_bundle(fetched)

        title = (form_meta.get("title") or fetched.get("title") or l_meta.get("title") or self.current_title or "Senza Titolo").strip()
        artist = (form_meta.get("artist") or fetched.get("artist") or l_meta.get("artist") or self.current_artist or "").strip()
        album = (form_meta.get("album") or fetched.get("album") or l_meta.get("album") or self.current_album or "").strip()
        youtube = (form_meta.get("youtube") or fetched.get("video_id") or self.current_youtube or "").strip()
        songwriters = (form_meta.get("songwriters") or l_meta.get("songwriters") or self.current_songwriters or "").strip()
        notes = (form_meta.get("notes") or l_meta.get("notes") or self.current_notes or "").strip()
        language = _normalize_lang_code(form_meta.get("language") or "") or fetched.get("language") or l_meta.get("language") or self.detected_language or "it"

        max_ms = max((int(ln.get("end_ms") or ln.get("start_ms") or 0) for ln in structured), default=0)
        dur_candidates = [
            float(form_meta.get("duration") or 0),
            float(self.calculated_duration or 0),
            float(fetched.get("duration") or 0),
            (max_ms / 1000.0 + 2.0) if max_ms > 0 else 0.0,
        ]
        duration_sec = int(round(next((d for d in dur_candidates if d > 0), 0)))
        exact_dur = self.calculated_duration_exact if self.calculated_duration_exact > 0 else float(duration_sec)
        target_format = formato or load_config().get("format", "metrolist")

        outro_enabled = form_meta.get("outro_credits")
        if outro_enabled is None:
            outro_enabled = self.current_outro_credits
        outro_text = form_meta.get("outro_text") if "outro_text" in form_meta else self.current_outro_text

        lrc_out = ""
        ttml_out = ""
        yaml_out = ""
        line_count = len(structured)
        word_count = sum(len(ln.get("words") or []) for ln in structured)

        if structured:
            structured_out, credit_entry = apply_outro_credits_to_structured(
                structured,
                songwriters=songwriters,
                duration_sec=exact_dur,
                enabled=bool(outro_enabled),
                outro_text=str(outro_text or ""),
                notes=notes,
                known_outro_texts=self._last_outro_candidates,
            )
            lrc_out = convert_structured_to_lrc(
                structured_out,
                title=title,
                artist=artist,
                album=album,
                duration_seconds=duration_sec,
                songwriters=songwriters,
                notes=notes,
                include_headers=True,
                formato=target_format,
            )
            clean_plain = strip_outro_credits_from_plain(plain_text, extra_texts=self._last_outro_candidates)
            if credit_entry is not None:
                plain_with_outro = f"{clean_plain}\n{credit_entry['text']}".strip() if clean_plain else credit_entry["text"]
            else:
                plain_with_outro = clean_plain

            yaml_out = build_lyricsfile_yaml(
                title=title,
                artist=artist,
                album=album,
                duration_ms=max(0, duration_sec) * 1000,
                language=language,
                structured_lines=structured_out,
                plain_text=plain_with_outro,
                songwriters=songwriters,
                notes=notes,
            )
            ttml_out = build_ttml(
                title=title,
                artist=artist,
                album=album,
                duration_seconds=duration_sec,
                language=language,
                structured_lines=structured_out,
                songwriters=songwriters,
                notes=notes,
            )
        else:
            clean_plain = plain_text.strip()
            plain_with_outro = clean_plain

        return {
            "sync_level": sync_level,
            "line_count": line_count,
            "word_count": word_count,
            "lrc": lrc_out,
            "ttml": ttml_out,
            "lyricsfile": yaml_out,
            "plain": clean_plain,
            "plain_synced": plain_with_outro,
            "duration": duration_sec,
            "meta": {
                "title": title,
                "artist": artist,
                "album": album,
                "youtube": youtube,
                "songwriters": songwriters,
                "notes": notes,
                "language": language,
                "duration": duration_sec,
            },
        }

    def search_online_lyrics(self, params):
        try:
            return search_all_online_lyrics(params or {})
        except Exception as e:
            return {"success": False, "error": str(e), "results": [], "errors": [str(e)]}

    def fetch_online_lyrics_detail(self, params):
        try:
            p = params or {}
            fetched = fetch_online_lyrics_raw(
                provider=p.get("provider", ""),
                item_id=p.get("id"),
                video_id=p.get("video_id", ""),
                raw_item=p.get("raw_item"),
            )
            pkg = self._build_converted_lyrics_package(
                fetched,
                form_meta=p.get("form_meta") or {},
                formato=p.get("format"),
            )
            return {
                "success": True,
                "provider": fetched["provider"],
                "raw_format": fetched["format"],
                "fetched": fetched,
                **pkg,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _safe_base_filename(self, meta: dict) -> str:
        artist = (meta.get("artist") or self.current_artist or "").strip()
        title = (meta.get("title") or self.current_title or "").strip()
        if title:
            return _build_song_folder_name(title=title, artist=artist)
        if self.last_audio_path and not self._is_yt_temp_audio:
            return os.path.splitext(os.path.basename(self.last_audio_path))[0]
        if self._is_yt_temp_audio and self._yt_output_basename:
            return self._yt_output_basename
        return "Testo"

    def _default_output_dir(self, meta: dict = None) -> str:
        m = meta or {}
        fallback = self._safe_base_filename(m)
        return _get_song_output_dir(
            title=m.get("title") or self.current_title,
            artist=m.get("artist") or self.current_artist,
            fallback=fallback,
        )

    def _write_format_files(self, base_dir: str, base_name: str, target_format: str, pkg: dict, explicit_path: str = "") -> list:
        saved = []
        os.makedirs(base_dir, exist_ok=True)
        if explicit_path:
            target_dir = os.path.dirname(explicit_path)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)

        if target_format == "metrolist_bundle":
            pairs = [
                (os.path.join(base_dir, f"{base_name}.lrc"), (pkg.get("lrc") or "") + "\n", bool(pkg.get("lrc"))),
                (os.path.join(base_dir, f"{base_name}.ttml"), pkg.get("ttml") or "", bool(pkg.get("ttml"))),
                (os.path.join(base_dir, f"{base_name}.lyricsfile.yaml"), pkg.get("lyricsfile") or "", bool(pkg.get("lyricsfile"))),
            ]
            for fpath, content, has_c in pairs:
                if has_c:
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(content)
                    saved.append(fpath)
            return saved

        ext_map = {
            "lrc": (".lrc", (pkg.get("lrc") or "") + ("\n" if pkg.get("lrc") else "")),
            "ttml": (".ttml", pkg.get("ttml") or ""),
            "yaml": (".lyricsfile.yaml", pkg.get("lyricsfile") or ""),
            "txt": (".txt", (pkg.get("plain") or "") + ("\n" if pkg.get("plain") else "")),
        }
        ext, content = ext_map.get(target_format, (".txt", pkg.get("plain") or ""))
        if not content.strip():
            raise Exception(f"Il formato '{target_format}' non è disponibile per questo risultato (testo non sincronizzato).")
        out_path = explicit_path or os.path.join(base_dir, f"{base_name}{ext}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        saved.append(out_path)
        return saved

    def download_and_save_online_lyrics(self, params):
        try:
            p = params or {}
            fetched = p.get("fetched")
            if not isinstance(fetched, dict) or not (fetched.get("lyrics_raw") or fetched.get("plain_raw") or fetched.get("lyricsfile_raw")):
                fetched = fetch_online_lyrics_raw(
                    provider=p.get("provider", ""),
                    item_id=p.get("id"),
                    video_id=p.get("video_id", ""),
                    raw_item=p.get("raw_item"),
                )

            form_meta = p.get("form_meta") or {}
            pkg = self._build_converted_lyrics_package(fetched, form_meta=form_meta, formato=p.get("format"))
            target_format = str(p.get("target_format") or ("metrolist_bundle" if pkg["sync_level"] != "plain" else "txt")).strip()
            save_mode = str(p.get("save_mode") or "auto_song_folder").strip()
            load_into_studio = bool(p.get("load_into_studio", True))

            if target_format != "txt" and pkg["sync_level"] == "plain":
                target_format = "txt"

            saved_paths = []
            if save_mode != "load_only":
                base_name = self._safe_base_filename(pkg["meta"])
                default_dir = self._default_output_dir(pkg["meta"])

                if save_mode == "save_dialog":
                    win = self._get_window()
                    if not win:
                        return {"success": False, "error": "Finestra non disponibile."}
                    if target_format == "metrolist_bundle":
                        folder_res = win.create_file_dialog(
                            webview.FileDialog.FOLDER,
                            directory=default_dir,
                        )
                        if not folder_res:
                            return {"success": False, "cancelled": True}
                        chosen_dir = folder_res[0] if isinstance(folder_res, (list, tuple)) else folder_res
                        saved_paths = self._write_format_files(chosen_dir, base_name, target_format, pkg)
                    else:
                        ext_labels = {
                            "lrc": (".lrc", "File LRC Metrolist (*.lrc)"),
                            "ttml": (".ttml", "File TTML BetterLyrics/Metrolist (*.ttml)"),
                            "yaml": (".lyricsfile.yaml", "File Lyricsfile YAML (*.yaml)"),
                            "txt": (".txt", "Testo Grezzo (*.txt)"),
                        }
                        ext, ft_label = ext_labels.get(target_format, (".txt", "File di Testo (*.txt)"))
                        file_res = win.create_file_dialog(
                            webview.FileDialog.SAVE,
                            directory=default_dir,
                            save_filename=f"{base_name}{ext}",
                            file_types=(ft_label,),
                        )
                        if not file_res:
                            return {"success": False, "cancelled": True}
                        chosen_path = file_res[0] if isinstance(file_res, (list, tuple)) else file_res
                        saved_paths = self._write_format_files(os.path.dirname(chosen_path), base_name, target_format, pkg, explicit_path=chosen_path)
                else:
                    saved_paths = self._write_format_files(default_dir, base_name, target_format, pkg)

            loaded_mode = "none"
            if load_into_studio:
                meta_out = pkg["meta"]
                self._sync_state_metadata(meta_out)
                if pkg["duration"] > 0 and self.calculated_duration < 1:
                    self.calculated_duration = pkg["duration"]
                    self.calculated_duration_exact = float(pkg["duration"])

                if target_format != "txt" and pkg["sync_level"] != "plain" and pkg["lrc"]:
                    self.generated_lrc = pkg["lrc"]
                    self.generated_ttml = pkg["ttml"]
                    self.generated_lyricsfile = pkg["lyricsfile"]
                    self.plain_lyrics = pkg.get("plain_synced") or pkg["plain"]
                    self.last_output_paths = {}
                    for sp in saved_paths:
                        if sp.endswith(".lrc"):
                            self.last_output_paths["lrc"] = sp
                        elif sp.endswith(".ttml"):
                            self.last_output_paths["ttml"] = sp
                        elif sp.endswith(".yaml"):
                            self.last_output_paths["yaml"] = sp
                    loaded_mode = "synced"
                else:
                    self.plain_lyrics = pkg["plain"]
                    loaded_mode = "plain"

            primary_path = saved_paths[0] if saved_paths else f"Online ({fetched['provider'].upper()})"
            return {
                "success": True,
                "saved_paths": saved_paths,
                "path": primary_path,
                "loaded_mode": loaded_mode,
                "target_format": target_format,
                **pkg,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def export_current_lyrics(self, params):
        try:
            p = params or {}
            target_format = str(p.get("target_format") or "metrolist_bundle").strip()
            form_meta = p.get("form_meta") or {}
            if form_meta and self.generated_lrc:
                self.update_outputs_metadata(form_meta)

            if target_format != "txt" and not self.generated_lrc:
                return {"success": False, "error": "Nessun testo sincronizzato presente in Studio."}

            plain_val = self.plain_lyrics or (format_plain_for_lrclib("", self.generated_lrc) if self.generated_lrc else "")
            if target_format == "txt" and not plain_val.strip():
                return {"success": False, "error": "Nessun testo disponibile da esportare."}

            pkg = {
                "lrc": self.generated_lrc,
                "ttml": self.generated_ttml,
                "lyricsfile": self.generated_lyricsfile,
                "plain": plain_val,
                "meta": {
                    "title": form_meta.get("title") or self.current_title or "Testo",
                    "artist": form_meta.get("artist") or self.current_artist or "",
                },
            }
            base_name = self._safe_base_filename(pkg["meta"])
            default_dir = self._default_output_dir(pkg["meta"])

            win = self._get_window()
            if not win:
                return {"success": False, "error": "Finestra non disponibile."}

            if target_format == "metrolist_bundle":
                folder_res = win.create_file_dialog(webview.FileDialog.FOLDER, directory=default_dir)
                if not folder_res:
                    return {"success": False, "cancelled": True}
                chosen_dir = folder_res[0] if isinstance(folder_res, (list, tuple)) else folder_res
                saved = self._write_format_files(chosen_dir, base_name, target_format, pkg)
            else:
                ext_labels = {
                    "lrc": (".lrc", "File LRC Metrolist (*.lrc)"),
                    "ttml": (".ttml", "File TTML BetterLyrics/Metrolist (*.ttml)"),
                    "yaml": (".lyricsfile.yaml", "File Lyricsfile YAML (*.yaml)"),
                    "txt": (".txt", "Testo Grezzo (*.txt)"),
                }
                ext, ft_label = ext_labels.get(target_format, (".lrc", "File LRC (*.lrc)"))
                file_res = win.create_file_dialog(
                    webview.FileDialog.SAVE,
                    directory=default_dir,
                    save_filename=f"{base_name}{ext}",
                    file_types=(ft_label,),
                )
                if not file_res:
                    return {"success": False, "cancelled": True}
                chosen_path = file_res[0] if isinstance(file_res, (list, tuple)) else file_res
                saved = self._write_format_files(os.path.dirname(chosen_path), base_name, target_format, pkg, explicit_path=chosen_path)

            return {"success": True, "saved_paths": saved}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def open_external_url(self, url: str):
        try:
            raw = str(url or "").strip()
            parsed = urllib.parse.urlparse(raw)
            if parsed.scheme != "https" or not parsed.netloc:
                return {"success": False, "error": "URL non consentito."}
            webbrowser.open(raw)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==========================================================================
    # DOWNLOAD MODELLI WHISPER LOCALI
    # ==========================================================================
    def get_whisper_models_status(self):
        return {
            "success": True,
            "catalog": get_whisper_models_status(self._model_dl_active_id, self._model_dl_active_pct),
            "available_models": get_available_models_list(),
        }

    def _emit_model_dl_event(self, payload: dict):
        win = self._get_window()
        if win and self.is_alive:
            try:
                win.evaluate_js(f"window.modelDownloadProgress({json.dumps(payload)});")
            except Exception:
                pass

    def cancel_model_download(self):
        self._model_dl_cancel.set()
        return {"success": True}

    def download_whisper_models(self, model_ids=None):
        with self._model_dl_lock:
            if self._model_dl_active_id:
                return {"success": False, "error": f"Download già in corso ({self._model_dl_active_id})."}

            catalog_map = {m["id"]: m for m in WHISPER_MODEL_CATALOG}
            if isinstance(model_ids, str):
                target_ids = [model_ids]
            elif isinstance(model_ids, (list, tuple)) and model_ids:
                target_ids = [str(x) for x in model_ids]
            else:
                target_ids = [
                    m["id"] for m in WHISPER_MODEL_CATALOG
                    if not (os.path.exists(m["path"]) and os.path.getsize(m["path"]) > 100_000_000)
                ]

            queue = [catalog_map[mid] for mid in target_ids if mid in catalog_map]
            if not queue:
                return {
                    "success": True,
                    "nothing_to_do": True,
                    "catalog": get_whisper_models_status(),
                    "available_models": get_available_models_list(),
                }

            self._model_dl_cancel.clear()
            self._model_dl_active_id = queue[0]["id"]
            self._model_dl_active_pct = 0.0

        def worker():
            try:
                for idx, item in enumerate(queue):
                    if self._model_dl_cancel.is_set():
                        break
                    mid = item["id"]
                    target_path = item["path"]
                    part_path = f"{target_path}.part"
                    url = item["url"]
                    expected_sha = item["sha256"]

                    parsed_u = urllib.parse.urlparse(url)
                    if parsed_u.scheme != "https" or not parsed_u.netloc:
                        raise Exception(f"URL modello non valido: {url}")

                    self._model_dl_active_id = mid
                    self._model_dl_active_pct = 0.0
                    self.log(f"⬇️ Download modello {item['filename']} ({idx + 1}/{len(queue)})...")
                    self._emit_model_dl_event({
                        "model_id": mid,
                        "filename": item["filename"],
                        "status": "downloading",
                        "pct": 0.0,
                        "downloaded_mb": 0.0,
                        "total_mb": round(item["approx_gb"] * 1024.0, 1),
                        "speed_mb_s": 0.0,
                        "catalog": get_whisper_models_status(mid, 0.0),
                    })

                    try:
                        req = urllib.request.Request(url, headers={"User-Agent": "MetroSync-Studio/1.0.0"})
                        hasher = hashlib.sha256()
                        downloaded = 0
                        start_t = time.time()
                        last_emit = 0.0

                        with urllib.request.urlopen(req, timeout=30) as resp:
                            total_bytes = int(resp.headers.get("Content-Length") or 0)
                            with open(part_path, "wb") as out_f:
                                while True:
                                    if self._model_dl_cancel.is_set():
                                        raise InterruptedError("Download annullato dall'utente.")
                                    chunk = resp.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    out_f.write(chunk)
                                    hasher.update(chunk)
                                    downloaded += len(chunk)

                                    now = time.time()
                                    if now - last_emit >= 0.25:
                                        last_emit = now
                                        elapsed = max(0.05, now - start_t)
                                        speed = (downloaded / (1024 * 1024)) / elapsed
                                        pct = round((downloaded * 100.0 / total_bytes), 1) if total_bytes > 0 else 0.0
                                        self._model_dl_active_pct = pct
                                        self._emit_model_dl_event({
                                            "model_id": mid,
                                            "filename": item["filename"],
                                            "status": "downloading",
                                            "pct": pct,
                                            "downloaded_mb": round(downloaded / (1024 * 1024), 1),
                                            "total_mb": round((total_bytes / (1024 * 1024)) if total_bytes > 0 else (item["approx_gb"] * 1024.0), 1),
                                            "speed_mb_s": round(speed, 1),
                                        })

                        self._emit_model_dl_event({
                            "model_id": mid,
                            "filename": item["filename"],
                            "status": "verifying",
                            "pct": 100.0,
                            "downloaded_mb": round(downloaded / (1024 * 1024), 1),
                            "total_mb": round(downloaded / (1024 * 1024), 1),
                            "speed_mb_s": 0.0,
                        })
                        got_sha = hasher.hexdigest().lower()
                        if expected_sha and got_sha != expected_sha.lower():
                            raise Exception(f"Checksum SHA-256 non valido per {item['filename']}.")

                        os.replace(part_path, target_path)
                        self.log(f"✅ Modello {item['filename']} scaricato e verificato in locale!")
                        self._emit_model_dl_event({
                            "model_id": mid,
                            "filename": item["filename"],
                            "status": "completed",
                            "pct": 100.0,
                            "path": target_path,
                            "catalog": get_whisper_models_status(mid if idx + 1 < len(queue) else "", 0.0),
                            "available_models": get_available_models_list(),
                        })
                    except InterruptedError:
                        if os.path.exists(part_path):
                            try:
                                os.remove(part_path)
                            except Exception:
                                pass
                        self.log(f"⏹️ Download modello {item['filename']} annullato.")
                        self._emit_model_dl_event({
                            "model_id": mid,
                            "filename": item["filename"],
                            "status": "cancelled",
                            "catalog": get_whisper_models_status(),
                            "available_models": get_available_models_list(),
                        })
                        break
                    except Exception as ex_item:
                        if os.path.exists(part_path):
                            try:
                                os.remove(part_path)
                            except Exception:
                                pass
                        self.log(f"❌ Errore download {item['filename']}: {ex_item}")
                        self._emit_model_dl_event({
                            "model_id": mid,
                            "filename": item["filename"],
                            "status": "error",
                            "error": str(ex_item),
                            "catalog": get_whisper_models_status(),
                            "available_models": get_available_models_list(),
                        })
                        break
            finally:
                with self._model_dl_lock:
                    self._model_dl_active_id = ""
                    self._model_dl_active_pct = 0.0
                self._emit_model_dl_event({
                    "status": "all_done",
                    "catalog": get_whisper_models_status(),
                    "available_models": get_available_models_list(),
                })

        threading.Thread(target=worker, daemon=True).start()
        return {"success": True, "queued": [m["id"] for m in queue]}

    def download_demucs_model(self, model_name: str = "htdemucs_ft"):
        chosen = model_name if model_name in ("htdemucs", "htdemucs_ft") else "htdemucs_ft"

        def worker():
            win = self._get_window()
            try:
                self.log(f"⬇️ Download modello Demucs '{chosen}' nella cache PyTorch Hub...")
                if win and self.is_alive:
                    win.evaluate_js(f"window.demucsDownloadProgress({json.dumps({'status': 'downloading', 'model': chosen})});")
                import demucs.pretrained
                m = demucs.pretrained.get_model(chosen)
                del m
                self.log(f"✅ Modello Demucs '{chosen}' pronto all'uso!")
                if win and self.is_alive:
                    win.evaluate_js(f"window.demucsDownloadProgress({json.dumps({'status': 'completed', 'model': chosen, 'demucs_catalog': get_demucs_models_status()})});")
            except Exception as ex:
                self.log(f"❌ Errore download Demucs '{chosen}': {ex}")
                if win and self.is_alive:
                    win.evaluate_js(f"window.demucsDownloadProgress({json.dumps({'status': 'error', 'model': chosen, 'error': str(ex), 'demucs_catalog': get_demucs_models_status()})});")

        threading.Thread(target=worker, daemon=True).start()
        return {"success": True, "model": chosen}

    def load_audio_from_youtube(self, url_or_id: str):
        def worker():
            win = self._get_window()
            try:
                res = download_youtube_audio_for_sync(url_or_id, log_func=self.log)
                audio_path = res["audio_path"]
                meta = res.get("meta") or {}
                if (meta.get("duration") or 0) < 1:
                    try:
                        arr = whisper.load_audio(audio_path)
                        self.calculated_duration_exact = float(len(arr) / 16000.0)
                        meta["duration"] = int(round(self.calculated_duration_exact))
                    except Exception:
                        pass
                else:
                    self.calculated_duration_exact = float(meta["duration"])

                self.last_audio_path = audio_path
                self._is_yt_temp_audio = True
                self._yt_output_basename = res["base_name"]
                self.last_output_paths = {}
                self.calculated_duration = int(meta.get("duration") or 0)
                self.current_title = meta.get("title", "")
                self.current_artist = meta.get("artist", "")
                self.current_album = meta.get("album", "")
                self.current_youtube = meta.get("youtube", "")
                self.current_songwriters = meta.get("songwriters", "")

                payload = {
                    "success": True,
                    "path": audio_path,
                    "display_path": f"YouTube ({res['video_id']}): {res['base_name']}",
                    **meta,
                }
                if win and self.is_alive:
                    win.evaluate_js(f"window.youtubeAudioLoaded({json.dumps(payload)});")
            except Exception as ex:
                self.log(f"❌ Errore caricamento YouTube: {ex}")
                if win and self.is_alive:
                    win.evaluate_js(f"window.youtubeAudioLoaded({json.dumps({'success': False, 'error': str(ex)})});")

        threading.Thread(target=worker, daemon=True).start()
        return {"success": True}

    def fit_window_to_content(self, content_width: int, content_height: int) -> dict:
        """Adatta automaticamente larghezza e altezza della finestra ai contenuti entro l'area utile dello schermo."""
        win = self._get_window()
        if not win:
            return {"success": False}
        try:
            native = getattr(win, "native", None)
            is_maximized = False
            scale = 1.0
            work_left, work_top, work_right, work_bottom = 0, 0, 1920, 1040

            if os.name == "nt":
                import ctypes
                from ctypes import wintypes

                class RECT(ctypes.Structure):
                    _fields_ = [
                        ("left", ctypes.c_long),
                        ("top", ctypes.c_long),
                        ("right", ctypes.c_long),
                        ("bottom", ctypes.c_long),
                    ]

                class MONITORINFO(ctypes.Structure):
                    _fields_ = [
                        ("cbSize", wintypes.DWORD),
                        ("rcMonitor", RECT),
                        ("rcWork", RECT),
                        ("dwFlags", wintypes.DWORD),
                    ]

                class WINDOWPLACEMENT(ctypes.Structure):
                    _fields_ = [
                        ("length", wintypes.UINT),
                        ("flags", wintypes.UINT),
                        ("showCmd", wintypes.UINT),
                        ("ptMinPosition", wintypes.POINT),
                        ("ptMaxPosition", wintypes.POINT),
                        ("rcNormalPosition", RECT),
                    ]

                user32 = ctypes.windll.user32
                hwnd = None
                if native is not None:
                    try:
                        scale = float(getattr(native, "_scale", 1.0) or 1.0)
                    except Exception:
                        scale = 1.0
                    try:
                        hwnd = int(native.Handle.ToInt32())
                    except Exception:
                        hwnd = None
                if not hwnd:
                    hwnd = user32.GetForegroundWindow()

                if hwnd:
                    try:
                        dpi = user32.GetDpiForWindow(hwnd)
                        if dpi and dpi > 0:
                            scale = float(dpi) / 96.0
                    except Exception:
                        pass
                    wp = WINDOWPLACEMENT()
                    wp.length = ctypes.sizeof(WINDOWPLACEMENT)
                    if user32.GetWindowPlacement(hwnd, ctypes.byref(wp)):
                        is_maximized = (wp.showCmd == 3)

                    hmon = user32.MonitorFromWindow(hwnd, 2)
                    if hmon:
                        mi = MONITORINFO()
                        mi.cbSize = ctypes.sizeof(MONITORINFO)
                        if user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                            work_left = int(round(mi.rcWork.left / scale))
                            work_top = int(round(mi.rcWork.top / scale))
                            work_right = int(round(mi.rcWork.right / scale))
                            work_bottom = int(round(mi.rcWork.bottom / scale))

            if is_maximized:
                return {"success": True, "maximized": True}

            avail_w = max(760, work_right - work_left - 12)
            avail_h = max(600, work_bottom - work_top - 12)

            min_w, min_h = (760, 600)
            target_w = max(min_w, min(int(round(content_width)), avail_w))
            target_h = max(min_h, min(int(round(content_height)), avail_h))

            cur_x = int(getattr(win, "x", work_left) or work_left)
            cur_y = int(getattr(win, "y", work_top) or work_top)
            cur_w = int(getattr(win, "width", target_w) or target_w)
            cur_h = int(getattr(win, "height", target_h) or target_h)

            new_x = cur_x
            new_y = cur_y
            if new_x + target_w > work_right - 6:
                new_x = max(work_left + 6, work_right - target_w - 6)
            if new_y + target_h > work_bottom - 6:
                new_y = max(work_top + 6, work_bottom - target_h - 6)
            if new_x < work_left + 6:
                new_x = work_left + 6
            if new_y < work_top + 6:
                new_y = work_top + 6

            if abs(new_x - cur_x) > 4 or abs(new_y - cur_y) > 4:
                win.move(new_x, new_y)
            if abs(target_w - cur_w) > 4 or abs(target_h - cur_h) > 4:
                win.resize(target_w, target_h)

            return {
                "success": True,
                "width": target_w,
                "height": target_h,
                "avail_w": avail_w,
                "avail_h": avail_h,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


def main():
    api = BackendAPI()
    html_file = os.path.join(APP_DIR, "index.html")

    window = webview.create_window(
        title="MetroSync Studio Pro",
        url=html_file,
        js_api=api,
        width=1140,
        height=880,
        min_size=(760, 600),
        background_color="#0e0e14",
    )

    def on_closed():
        api.is_alive = False
        os._exit(0)

    window.events.closed += on_closed
    webview.start()


if __name__ == "__main__":
    main()