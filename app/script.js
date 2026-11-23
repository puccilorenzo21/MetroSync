let appConfig = null;
let currentLyrics = [];
let audioLoaded = false;
let audioDuration = 0;
let lastActiveIdx = -1;

window.addEventListener('pywebviewready', async () => {
  try {
    appConfig = await window.pywebview.api.init_info();
    document.getElementById('hwBadge').innerText = `${appConfig.gpu_name} (${appConfig.device})`;
    document.getElementById('settingModel').value = appConfig.default_model;

    // Popola preset modelli disponibili
    populateModelDropdown(appConfig.available_models || [], appConfig.default_model);
    if (appConfig.model_catalog) {
      renderModelCatalog(appConfig.model_catalog);
    }

    const langSelect = document.getElementById('settingLanguage');
    langSelect.innerHTML = '';
    appConfig.languages.forEach(l => {
      const opt = document.createElement('option');
      opt.value = l;
      opt.innerText = l;
      if (l === appConfig.language) opt.selected = true;
      langSelect.appendChild(opt);
    });

    document.getElementById('settingDemucs').checked = appConfig.demucs;
    const dmSel = document.getElementById('settingDemucsModel');
    if (dmSel && appConfig.demucs_model) {
      dmSel.value = appConfig.demucs_model;
    }
    if (appConfig.demucs_catalog) {
      updateDemucsCatalogUI(appConfig.demucs_catalog);
    }
    document.getElementById('settingVAD').checked = appConfig.vad;
    document.querySelector(`input[name="settingFormat"][value="${appConfig.format}"]`).checked = true;
    document.getElementById('settingOffset').value = appConfig.offset;
    updateOffsetLabel(appConfig.offset, true);

    // Inizializzazione Auto-Offset da configurazione
    const autoOffsetToggle = document.getElementById('settingAutoOffset');
    if (autoOffsetToggle) {
      autoOffsetToggle.checked = (appConfig.auto_offset !== false);
      updateAutoOffsetUI(autoOffsetToggle.checked);
    }

    const outroCreditsToggle = document.getElementById('settingOutroCredits');
    if (outroCreditsToggle) {
      outroCreditsToggle.checked = (appConfig.outro_credits !== false);
      updateOutroTextUI(outroCreditsToggle.checked);
    }

    const outroTextEl = document.getElementById('metaOutroText');
    if (outroTextEl && appConfig.outro_text !== undefined) {
      outroTextEl.value = appConfig.outro_text || 'MetroSync Studio';
    }

    if (appConfig.publishing || appConfig.unison_identity) {
      applyPublishingState(appConfig.publishing, appConfig.unison_identity);
    }

    scheduleWindowFit(true);

    ['searchTitle', 'searchArtist', 'searchYoutube'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            runOnlineLyricsSearch();
          }
        });
      }
    });
  } catch (e) {
    console.error(e);
  }
});

let _refreshTimer = null;
function scheduleOutputsRefresh() {
  if (_refreshTimer) clearTimeout(_refreshTimer);
  _refreshTimer = setTimeout(() => { refreshOutputsFromForm(); }, 300);
}

function updateOutroTextUI(isActive) {
  const outroInput = document.getElementById('metaOutroText');
  if (outroInput) {
    outroInput.classList.toggle('outro-disabled', !isActive);
  }
}

async function autoSaveSettings() {
  const autoOffsetEl = document.getElementById('settingAutoOffset');
  const outroCreditsEl = document.getElementById('settingOutroCredits');
  const currentSettings = {
    model_path: document.getElementById('settingModel').value,
    language: document.getElementById('settingLanguage').value,
    demucs: document.getElementById('settingDemucs').checked,
    demucs_model: document.getElementById('settingDemucsModel')?.value || 'htdemucs',
    vad: document.getElementById('settingVAD').checked,
    format: document.querySelector('input[name="settingFormat"]:checked').value,
    offset: parseInt(document.getElementById('settingOffset').value),
    auto_offset: autoOffsetEl ? autoOffsetEl.checked : true,
    outro_credits: outroCreditsEl ? outroCreditsEl.checked : true
  };
  await window.pywebview.api.save_user_settings(currentSettings);
}

async function refreshOutputsFromForm() {
  if (!window.generatedLrc && !window.generatedTtml) return;
  try {
    const upd = await window.pywebview.api.update_outputs_metadata(getFormMetadata());
    if (upd && upd.success && upd.lrc) {
      window.generatedLrc = upd.lrc;
      window.generatedTtml = upd.ttml;
      window.generatedLyricsfile = upd.lyricsfile;
    }
  } catch (e) {}
}

async function onOutroCreditsToggle() {
  const outroCreditsEl = document.getElementById('settingOutroCredits');
  updateOutroTextUI(outroCreditsEl ? outroCreditsEl.checked : true);
  await autoSaveSettings();
  await refreshOutputsFromForm();
}

function toggleAutoOffset() {
  const isChecked = document.getElementById('settingAutoOffset').checked;
  updateAutoOffsetUI(isChecked);
  autoSaveSettings();
}

let lastCalibratedAutoMs = null;

function _formatStudioOffsetText(isAuto) {
  const manMs = parseInt(document.getElementById('settingOffset')?.value || '0', 10) || 0;
  const manSuffix = manMs !== 0 ? ` (${manMs > 0 ? '+' : ''}${manMs}ms man.)` : '';
  if (isAuto) {
    if (lastCalibratedAutoMs !== null) {
      const sign = lastCalibratedAutoMs > 0 ? '+' : '';
      const valStr = lastCalibratedAutoMs === 0 ? '0ms' : `${sign}${lastCalibratedAutoMs}ms`;
      return `⚡ Auto: ${valStr}${manSuffix}`;
    }
    return `⚡ Auto${manSuffix || ' (Attiva in sync)'}`;
  }
  return manMs !== 0 ? `Manuale: ${manMs > 0 ? '+' : ''}${manMs}ms` : 'Off (0ms)';
}

function updateAutoOffsetUI(isActive) {
  const studioBadge = document.getElementById('studioOffsetBadge');
  const btnQuick = document.getElementById('btnQuickCalibrate');
  const autoBox = document.getElementById('autoOffsetInfoBox');
  const sliderTitle = document.getElementById('offsetSliderTitle');
  const manMs = parseInt(document.getElementById('settingOffset')?.value || '0', 10) || 0;

  if (isActive) {
    if (studioBadge) {
      studioBadge.innerText = _formatStudioOffsetText(true);
      studioBadge.classList.remove('disabled');
      if (manMs !== 0) {
        studioBadge.title = `Attenzione: è attivo anche un offset manuale di ${manMs > 0 ? '+' : ''}${manMs}ms (sommato all'Auto-Compensazione).`;
      } else {
        studioBadge.title = 'Calibrazione automatica dell\'anticipo vocale';
      }
    }
    if (btnQuick) btnQuick.style.display = 'inline-flex';
    if (autoBox) autoBox.classList.remove('disabled');
    if (sliderTitle) sliderTitle.innerText = 'Regolazione Fine Manuale (Offset Addizionale):';
  } else {
    if (studioBadge) {
      studioBadge.innerText = _formatStudioOffsetText(false);
      studioBadge.classList.toggle('disabled', manMs === 0);
      studioBadge.classList.remove('calibrated');
    }
    if (btnQuick) btnQuick.style.display = 'none';
    if (autoBox) autoBox.classList.add('disabled');
    if (sliderTitle) sliderTitle.innerText = 'Compensazione Manuale Fissa:';
  }
}

async function runAutoCalibration() {
  const audioPath = document.getElementById('audioPath').value;
  const ytAudioUrl = (document.getElementById('youtubeAudioUrl')?.value || document.getElementById('metaYoutube')?.value || '').trim();
  const lyricsText = window.tempLyricsContent || "";

  if (!audioPath && !ytAudioUrl) {
    alert("Seleziona prima un file audio o incolla un link YouTube!");
    return;
  }

  const btnQuick = document.getElementById('btnQuickCalibrate');
  const btnFull = document.getElementById('btnCalibrate');
  const badgeStudio = document.getElementById('studioOffsetBadge');
  const badgeSettings = document.getElementById('autoOffsetBadge');

  if (btnQuick) {
    btnQuick.disabled = true;
    btnQuick.innerHTML = '<span class="spinner-dot"></span> Analisi...';
  }
  if (btnFull) {
    btnFull.disabled = true;
    btnFull.innerHTML = '<span class="spinner-dot"></span> Analisi acustica in corso...';
  }

  if (badgeStudio) badgeStudio.innerText = "⏳ Calibrazione...";
  if (badgeSettings) badgeSettings.innerText = "⏳ Calcolo latenza vocale...";

  window.appendLog("🔍 Avvio pre-analisi acustica del brano per compensazione automatica...");

  const params = {
    audio_path: audioPath,
    youtube_audio_url: ytAudioUrl,
    lyrics_text: lyricsText,
    model_path: document.getElementById('settingModel').value,
    language: document.getElementById('settingLanguage').value,
    demucs: document.getElementById('settingDemucs') ? document.getElementById('settingDemucs').checked : true,
    demucs_model: document.getElementById('settingDemucsModel')?.value || 'htdemucs'
  };

  try {
    await window.pywebview.api.analyze_auto_compensation(params);
  } catch (err) {
    console.error(err);
    resetCalibrateButtons();
  }
}

window.autoOffsetFinished = function(res) {
  resetCalibrateButtons();
  const isAuto = document.getElementById('settingAutoOffset')?.checked !== false;
  if (res && res.success) {
    const ms = Number(res.offset_ms || 0);
    lastCalibratedAutoMs = ms;
    const sign = ms > 0 ? '+' : '';
    const valStr = ms === 0 ? '0ms (Fase 0)' : `${sign}${ms}ms`;

    const badgeStudio = document.getElementById('studioOffsetBadge');
    if (badgeStudio) {
      badgeStudio.innerText = _formatStudioOffsetText(isAuto);
      badgeStudio.classList.add('calibrated');
    }

    const badgeSettings = document.getElementById('autoOffsetBadge');
    if (badgeSettings) {
      const modeTxt = res.mode ? ` • ${res.mode}` : '';
      badgeSettings.innerText = `Calibrato: ${valStr}${modeTxt}`;
      badgeSettings.classList.add('calibrated');
    }
  } else {
    lastCalibratedAutoMs = null;
    const badgeStudio = document.getElementById('studioOffsetBadge');
    if (badgeStudio) badgeStudio.innerText = _formatStudioOffsetText(isAuto);
    const badgeSettings = document.getElementById('autoOffsetBadge');
    if (badgeSettings) badgeSettings.innerText = "Non rilevato (auto in sync)";
  }
};

function resetCalibrateButtons() {
  const btnQuick = document.getElementById('btnQuickCalibrate');
  if (btnQuick) {
    btnQuick.disabled = false;
    btnQuick.innerHTML = '🔍 Calibra';
  }
  const btnFull = document.getElementById('btnCalibrate');
  if (btnFull) {
    btnFull.disabled = false;
    btnFull.innerHTML = '<span class="icon">🔍</span> Analizza Brano e Calibra Ora';
  }
}

function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

  if (tabId === 'studio') {
    document.querySelectorAll('.tab-btn')[0].classList.add('active');
    document.getElementById('tabStudio').classList.add('active');
  } else {
    document.querySelectorAll('.tab-btn')[1].classList.add('active');
    document.getElementById('tabSettings').classList.add('active');
  }
  scheduleWindowFit(true);
}

function switchSettingsSubTab(subId) {
  const map = {
    models: 'settingsSubModels',
    audio: 'settingsSubAudio',
    auth: 'settingsSubAuth'
  };
  document.querySelectorAll('.settings-subtab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-subtab') === subId);
  });
  Object.entries(map).forEach(([k, paneId]) => {
    const pane = document.getElementById(paneId);
    if (pane) pane.classList.toggle('active', k === subId);
  });
  scheduleWindowFit(true);
}

function getFormMetadata() {
  const outroCreditsEl = document.getElementById('settingOutroCredits');
  return {
    title: (document.getElementById('metaTitle')?.value || "").trim(),
    artist: (document.getElementById('metaArtist')?.value || "").trim(),
    album: (document.getElementById('metaAlbum')?.value || "").trim(),
    youtube: (document.getElementById('metaYoutube')?.value || "").trim(),
    songwriters: (document.getElementById('metaSongwriters')?.value || "").trim(),
    notes: (document.getElementById('metaNotes')?.value || "").trim(),
    outro_text: (document.getElementById('metaOutroText')?.value || "").trim(),
    outro_credits: outroCreditsEl ? outroCreditsEl.checked : true,
    duration: audioDuration || 0,
    language: document.getElementById('settingLanguage')?.value || ""
  };
}

function syncLanguageDropdown(isoCode, force = false) {
  if (!isoCode) return;
  const langSelect = document.getElementById('settingLanguage');
  if (!langSelect) return;
  const curVal = (langSelect.value || "").toLowerCase();
  if (!force && !curVal.includes('auto')) return;
  const targetSuffix = `(${isoCode.toLowerCase()})`;
  for (const opt of langSelect.options) {
    if ((opt.value || "").toLowerCase().includes(targetSuffix)) {
      langSelect.value = opt.value;
      break;
    }
  }
}

function applyPrefilledMetadata(meta, overwrite = false) {
  if (!meta) return;
  const setField = (id, val, clearIfEmpty) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (overwrite) {
      if (val) el.value = val;
      else if (clearIfEmpty) el.value = "";
    } else if (!el.value.trim() && val) {
      el.value = val;
    }
  };

  setField('metaTitle', meta.title, true);
  setField('metaArtist', meta.artist, true);
  setField('metaAlbum', meta.album, true);
  setField('metaYoutube', meta.youtube, true);
  setField('metaSongwriters', meta.songwriters, true);
  setField('metaNotes', meta.notes, false);
  setField('metaOutroText', meta.outro_text, false);

  if (meta.duration && meta.duration > 0 && (overwrite || !audioDuration)) {
    audioDuration = meta.duration;
    document.getElementById('metaDuration').innerText = `${audioDuration}s`;
  }
  if (meta.language) {
    syncLanguageDropdown(meta.language, overwrite);
  }
}

async function autoFillMetadata() {
  const audioPath = document.getElementById('audioPath')?.value || "";
  const lyricsPath = document.getElementById('lyricsPath')?.value || "";
  if (!audioPath && !lyricsPath) {
    alert("Seleziona prima un file audio o un file testo!");
    return;
  }
  const meta = await window.pywebview.api.auto_fill_metadata_from_files(audioPath, lyricsPath);
  if (meta) {
    applyPrefilledMetadata(meta, true);
    const found = [
      meta.title && `Titolo: ${meta.title}`,
      meta.artist && `Artista: ${meta.artist}`,
      meta.album && `Album: ${meta.album}`,
      meta.youtube && `YouTube: ${meta.youtube}`,
      meta.songwriters && `Autori: ${meta.songwriters}`,
      meta.notes && `Note rilevate`
    ].filter(Boolean);
    window.appendLog(found.length
      ? `🪄 Precompilazione metadati: ${found.join(' • ')}`
      : "ℹ️ Nessun tag aggiuntivo trovato nei file selezionati.");
  }
}

async function browseAudio() {
  const res = await window.pywebview.api.select_audio_file();
  if (res) {
    const ytInp = document.getElementById('youtubeAudioUrl');
    if (ytInp) ytInp.value = '';
    document.getElementById('audioPath').value = res.path;
    applyPrefilledMetadata(res, true);
    audioLoaded = false;

    const extras = [res.album && `Album: ${res.album}`, res.youtube && `YT: ${res.youtube}`, res.songwriters && `Autori: ${res.songwriters}`].filter(Boolean);
    if (extras.length > 0) {
      window.appendLog(`🏷️ Tag rilevati dall'audio → ${extras.join(' | ')}`);
    }

    const badgeStudio = document.getElementById('studioOffsetBadge');
    if (badgeStudio) {
      const isAuto = document.getElementById('settingAutoOffset') ? document.getElementById('settingAutoOffset').checked : true;
      badgeStudio.innerText = isAuto ? "⚡ Auto (Pronto per analisi)" : "Off (Manuale)";
      badgeStudio.classList.remove('calibrated');
    }
    const badgeSettings = document.getElementById('autoOffsetBadge');
    if (badgeSettings) {
      badgeSettings.innerText = "Pronto per analisi";
      badgeSettings.classList.remove('calibrated');
    }
  }
}

function applySyncedLyricsData(res) {
  document.getElementById('lyricsPath').value = res.path;
  window.tempLyricsContent = res.lrc;
  window.generatedLrc = res.lrc;
  window.generatedLyricsfile = res.lyricsfile;
  window.generatedTtml = res.ttml;

  if (res.meta) {
    applyPrefilledMetadata(res.meta, false);
  }
  if (res.duration && res.duration > 0) {
    audioDuration = res.duration;
    document.getElementById('metaDuration').innerText = `${audioDuration}s`;
  }

  document.getElementById('btnPreview').disabled = false;
  document.getElementById('btnPublish').disabled = false;
  document.getElementById('btnPublishBL').disabled = false;
  document.getElementById('btnCopy').disabled = false;
  document.getElementById('btnSaveProj').disabled = false;
  const btnExp = document.getElementById('btnExportFiles');
  if (btnExp) btnExp.disabled = false;

  window.appendLog(`✨ File sincronizzato caricato: ${res.path}`);
  window.appendLog(`📊 Righe caricate: ${res.line_count || 'OK'} (Durata: ${res.duration}s)`);
  window.appendLog(`✅ Sbloccati subito: Anteprima, LRCLIB, BetterLyrics, Esporta e Copia.`);
}

async function browseLyrics() {
  const res = await window.pywebview.api.select_lyrics_file();
  if (res && res.path) {
    if (res.is_synced) {
      applySyncedLyricsData(res);
      alert("✨ File sincronizzato rilevato!\n\nI tempi e i millisecondi sono pronti all'uso.");
    } else {
      document.getElementById('lyricsPath').value = res.path;
      window.tempLyricsContent = res.content;
      if (res.meta) applyPrefilledMetadata(res.meta, false);
      const btnExp = document.getElementById('btnExportFiles');
      if (btnExp && res.content) btnExp.disabled = false;
      window.appendLog(`📄 Testo grezzo caricato: ${res.path}`);
    }
  }
}

async function importSyncedLyrics() {
  const res = await window.pywebview.api.import_synced_lyrics_file();
  if (res && res.success) {
    applySyncedLyricsData(res);
    alert(`✨ Testo già sincronizzato importato con successo!`);
  } else if (res && res.error) {
    alert("Errore durante l'importazione: " + res.error);
  }
}

function onModelPresetChange(val) {
  if (val !== 'custom') {
    document.getElementById('settingModel').value = val;
    autoSaveSettings();
    renderModelCatalog(modelCatalogCache);
  } else {
    browseModel();
  }
}

async function browseModel() {
  const res = await window.pywebview.api.select_model_file();
  if (res && res.path) {
    document.getElementById('settingModel').value = res.path;
    const modelSelect = document.getElementById('settingModelSelect');
    if (modelSelect) {
      let matched = false;
      for (let i = 0; i < modelSelect.options.length; i++) {
        if (modelSelect.options[i].value.toLowerCase() === res.path.toLowerCase()) {
          modelSelect.selectedIndex = i;
          matched = true;
          break;
        }
      }
      if (!matched) {
        modelSelect.value = 'custom';
      }
    }
    autoSaveSettings();
    renderModelCatalog(modelCatalogCache);
  }
}

function updateOffsetLabel(val, skipSave = false) {
  const n = parseInt(val, 10) || 0;
  const sec = (n / 1000).toFixed(2);
  document.getElementById('offsetValue').innerText = `${n > 0 ? '+' : ''}${n} ms (${n > 0 ? '+' : ''}${sec} s)`;
  const btnReset = document.getElementById('btnResetManualOffset');
  if (btnReset) btnReset.style.opacity = n === 0 ? '0.45' : '1';
  const isAuto = document.getElementById('settingAutoOffset')?.checked !== false;
  updateAutoOffsetUI(isAuto);
  if (!skipSave) autoSaveSettings();
}

function resetManualOffset() {
  const slider = document.getElementById('settingOffset');
  if (slider) {
    slider.value = 0;
    updateOffsetLabel(0, false);
  }
}

window.appendLog = function(msg) {
  const con = document.getElementById('logConsole');
  const text = msg.toString();
  if (text.includes('Align:') || text.includes('Adjustment:') || text.includes('Refine:')) {
    const lines = con.innerText.split('\n');
    if (lines.length > 1 && (lines[lines.length - 2].includes('Align:') || lines[lines.length - 2].includes('Adjustment:') || lines[lines.length - 2].includes('Refine:'))) {
      lines[lines.length - 2] = text;
      con.innerText = lines.join('\n');
    } else {
      con.innerText += text + '\n';
    }
  } else {
    con.innerText += text + '\n';
  }
  con.scrollTop = con.scrollHeight;
};

window.updateProgress = function(pct) {
  document.getElementById('progressBar').style.width = `${pct}%`;
  document.getElementById('progressText').innerText = `${pct}%`;
};

function clearLog() {
  document.getElementById('logConsole').innerText = '';
}

// ==============================================================================
// GESTIONE PROGETTO
// ==============================================================================
async function saveProject() {
  await refreshOutputsFromForm();
  const autoOffsetEl = document.getElementById('settingAutoOffset');
  const formMeta = getFormMetadata();
  const projectData = {
    ...formMeta,
    audio_path: document.getElementById('audioPath').value,
    lyrics_text: window.tempLyricsContent || "",
    settings: {
      format: document.querySelector('input[name="settingFormat"]:checked').value,
      offset: document.getElementById('settingOffset').value,
      auto_offset: autoOffsetEl ? autoOffsetEl.checked : true,
      outro_credits: formMeta.outro_credits
    }
  };

  const res = await window.pywebview.api.save_project_file(projectData);
  if (res && res.success) {
    alert(`💾 Progetto salvato con successo in:\n${res.path}`);
  }
}

async function loadProject() {
  const res = await window.pywebview.api.load_project_file();
  if (res && res.success && res.data) {
    const d = res.data;

    document.getElementById('audioPath').value = d.audio_path || "";
    document.getElementById('lyricsPath').value = d.project_file_path || "Caricato da progetto";
    window.tempLyricsContent = d.lyrics_text || "";

    applyPrefilledMetadata({
      title: d.title || "",
      artist: d.artist || "",
      album: d.album || "",
      youtube: d.youtube || d.youtube_id || "",
      songwriters: d.songwriters || "",
      notes: d.notes || "",
      outro_text: d.outro_text || "",
      duration: d.duration || 0,
      language: d.language || ""
    }, true);

    if (d.settings) {
      if (d.settings.offset !== undefined) {
        document.getElementById('settingOffset').value = d.settings.offset;
        updateOffsetLabel(d.settings.offset, true);
      }
      if (d.settings.auto_offset !== undefined && document.getElementById('settingAutoOffset')) {
        document.getElementById('settingAutoOffset').checked = d.settings.auto_offset;
        updateAutoOffsetUI(d.settings.auto_offset);
      }
    }
    const outroVal = d.outro_credits !== undefined ? d.outro_credits : d.settings?.outro_credits;
    if (outroVal !== undefined && document.getElementById('settingOutroCredits')) {
      document.getElementById('settingOutroCredits').checked = Boolean(outroVal);
      updateOutroTextUI(Boolean(outroVal));
    }

    if (d.generated_lrc) {
      window.generatedLrc = d.generated_lrc;
      window.generatedLyricsfile = d.generated_lyricsfile || "";
      window.generatedTtml = d.generated_ttml || "";
      audioLoaded = false;

      document.getElementById('btnPreview').disabled = false;
      document.getElementById('btnPublish').disabled = false;
      document.getElementById('btnPublishBL').disabled = false;
      document.getElementById('btnCopy').disabled = false;
      document.getElementById('btnSaveProj').disabled = false;
      const btnExp = document.getElementById('btnExportFiles');
      if (btnExp) btnExp.disabled = false;

      window.appendLog(`📂 Progetto caricato: ${d.artist} - ${d.title}`);
      window.appendLog(`✅ Testo sincronizzato pronto all'uso!`);
      alert("Progetto caricato con successo!");
    }
  }
}

async function loadYoutubeAudio() {
  const ytUrl = (document.getElementById('youtubeAudioUrl')?.value || document.getElementById('metaYoutube')?.value || '').trim();
  if (!ytUrl) {
    alert("Incolla prima un link o Video ID di YouTube / YouTube Music!");
    return;
  }
  const btn = document.getElementById('btnLoadYoutubeAudio');
  if (btn) {
    btn.disabled = true;
    btn.innerText = '⏳ Download in background...';
  }
  window.appendLog(`🎬 Caricamento audio da YouTube in background: ${ytUrl}`);
  await window.pywebview.api.load_audio_from_youtube(ytUrl);
}

window.youtubeAudioLoaded = function(res) {
  const btn = document.getElementById('btnLoadYoutubeAudio');
  if (btn) {
    btn.disabled = false;
    btn.innerText = '🎬 Carica da YouTube';
  }
  if (!res || !res.success) {
    alert(`Errore caricamento audio YouTube:\n${res?.error || 'Errore sconosciuto'}`);
    return;
  }
  document.getElementById('audioPath').value = res.path || '';
  applyPrefilledMetadata(res, true);
  audioLoaded = false;

  const badgeStudio = document.getElementById('studioOffsetBadge');
  if (badgeStudio) {
    const isAuto = document.getElementById('settingAutoOffset') ? document.getElementById('settingAutoOffset').checked : true;
    badgeStudio.innerText = isAuto ? "⚡ Auto (Pronto per analisi)" : "Off (Manuale)";
    badgeStudio.classList.remove('calibrated');
  }
  const badgeSettings = document.getElementById('autoOffsetBadge');
  if (badgeSettings) {
    badgeSettings.innerText = "Pronto per analisi";
    badgeSettings.classList.remove('calibrated');
  }
  window.appendLog(`✅ Audio YouTube pronto: ${res.display_path || res.path}`);
};

async function startSync() {
  const audioPath = document.getElementById('audioPath').value;
  const ytAudioUrl = (document.getElementById('youtubeAudioUrl')?.value || document.getElementById('metaYoutube')?.value || '').trim();
  const lyricsText = window.tempLyricsContent;

  if (!audioPath && !ytAudioUrl) {
    alert("Seleziona prima un file audio oppure incolla un link YouTube!");
    return;
  }
  if (!lyricsText) {
    alert("Seleziona prima il file del testo o cercalo online!");
    return;
  }

  document.getElementById('btnSync').disabled = true;
  document.getElementById('btnPreview').disabled = true;
  document.getElementById('btnPublish').disabled = true;
  document.getElementById('btnPublishBL').disabled = true;
  document.getElementById('btnCopy').disabled = true;
  document.getElementById('btnSaveProj').disabled = true;
  const btnExp = document.getElementById('btnExportFiles');
  if (btnExp) btnExp.disabled = true;

  const autoOffsetEl = document.getElementById('settingAutoOffset');
  const isAuto = autoOffsetEl ? autoOffsetEl.checked : true;
  const formMeta = getFormMetadata();

  const params = {
    audio_path: audioPath,
    youtube_audio_url: ytAudioUrl,
    lyrics_text: lyricsText,
    title: formMeta.title,
    artist: formMeta.artist,
    album: formMeta.album,
    youtube: formMeta.youtube,
    songwriters: formMeta.songwriters,
    notes: formMeta.notes,
    outro_text: formMeta.outro_text,
    outro_credits: formMeta.outro_credits,
    model_path: document.getElementById('settingModel').value,
    language: document.getElementById('settingLanguage').value,
    demucs: document.getElementById('settingDemucs').checked,
    demucs_model: document.getElementById('settingDemucsModel')?.value || 'htdemucs',
    vad: document.getElementById('settingVAD').checked,
    format: document.querySelector('input[name="settingFormat"]:checked').value,
    offset: document.getElementById('settingOffset').value,
    auto_offset: isAuto
  };

  await window.pywebview.api.start_synchronization(params);
}

window.syncFinished = function(success, data) {
  document.getElementById('btnSync').disabled = false;
  if (success) {
    window.generatedLrc = data.lrc;
    window.generatedLyricsfile = data.lyricsfile;
    window.generatedTtml = data.ttml;
    document.getElementById('metaDuration').innerText = `${data.duration}s`;
    audioDuration = data.duration;
    audioLoaded = false;

    if (data.audio_path && !document.getElementById('audioPath').value) {
      document.getElementById('audioPath').value = data.audio_path;
    }

    if (data.meta) {
      applyPrefilledMetadata(data.meta, false);
    }

    const badgeStudio = document.getElementById('studioOffsetBadge');
    const badgeSettings = document.getElementById('autoOffsetBadge');
    if (document.getElementById('settingAutoOffset').checked) {
      const hasMs = (data.auto_offset_ms !== undefined && data.auto_offset_ms !== null);
      const ms = Number(data.auto_offset_ms || 0);
      const sign = ms > 0 ? '+' : '';
      const valStr = hasMs ? (ms === 0 ? '0ms (Fase 0)' : `${sign}${ms}ms`) : 'Calibrato';
      if (badgeStudio) {
        badgeStudio.innerText = `⚡ Auto: ${valStr}`;
        badgeStudio.classList.add('calibrated');
      }
      if (badgeSettings) {
        badgeSettings.innerText = `Calibrato: ${valStr}`;
        badgeSettings.classList.add('calibrated');
      }
    }

    document.getElementById('btnPreview').disabled = false;
    document.getElementById('btnPublish').disabled = false;
    document.getElementById('btnPublishBL').disabled = false;
    document.getElementById('btnCopy').disabled = false;
    document.getElementById('btnSaveProj').disabled = false;
    const btnExp = document.getElementById('btnExportFiles');
    if (btnExp) btnExp.disabled = false;
  }
};

async function copyLyrics() {
  if (window.generatedLrc) {
    await refreshOutputsFromForm();
    await navigator.clipboard.writeText(window.generatedLrc);
    alert("Testo sincronizzato (con metadati aggiornati) copiato negli appunti!");
  }
}

async function publishLRCLIB() {
  const meta = getFormMetadata();

  if (!meta.title || !meta.artist) {
    alert("Titolo e Artista sono obbligatori per pubblicare su LRCLIB!");
    return;
  }

  document.getElementById('btnPublish').disabled = true;
  await window.pywebview.api.publish_lrclib(meta);
}

window.publishFinished = function(res) {
  document.getElementById('btnPublish').disabled = false;
  if (res && res.success) {
    alert("🎉 Pubblicazione completata su LRCLIB!");
  } else {
    alert(`Errore LRCLIB:\n${res.detail || JSON.stringify(res)}`);
  }
};

async function publishBetterLyrics() {
  const meta = getFormMetadata();
  if (!meta.youtube) {
    alert("Inserisci il link o l'ID del video di YouTube Music del brano!");
    return;
  }

  if (!meta.title || !meta.artist) {
    alert("Titolo e Artista sono obbligatori!");
    return;
  }

  document.getElementById('btnPublishBL').disabled = true;
  await window.pywebview.api.publish_betterlyrics(meta);
}

window.publishBetterLyricsFinished = function(res) {
  document.getElementById('btnPublishBL').disabled = false;
  if (res && res.success) {
    alert(`🎉 Pubblicato con successo su BetterLyrics!\nVideo ID: ${res.video_id}\n\nOra il testo parola per parola è disponibile a tutti su Metrolist!`);
  } else {
    alert(`Errore BetterLyrics:\n${res.detail || JSON.stringify(res)}`);
  }
};

// ==============================================================================
// ANTEPRIMA METROLIST (KARAOKE GLOW WIPE + PAUSE & PAROLE ALLUNGATE)
// ==============================================================================
const audioElement = document.getElementById('html5Audio');
const seekSlider = document.getElementById('seekSlider');
const previewSpeedLevels = [1.0, 0.75, 1.25, 1.5];
let previewSpeedIdx = 0;

function setPreviewPlayingState(playing) {
  const modal = document.getElementById('previewModal');
  const btn = document.getElementById('btnPlayPause');
  if (modal) modal.classList.toggle('is-playing', Boolean(playing));
  if (btn) btn.innerText = playing ? '⏸' : '▶';
}

audioElement.onplay = () => setPreviewPlayingState(true);
audioElement.onpause = () => setPreviewPlayingState(false);
audioElement.onended = () => setPreviewPlayingState(false);

async function openPreview() {
  if (!window.generatedLrc) return;

  await refreshOutputsFromForm();
  parseLRC(window.generatedLrc);
  renderLyricsDOM();

  const title = document.getElementById('metaTitle').value || "Anteprima Canzone";
  const artist = document.getElementById('metaArtist').value || "";
  document.getElementById('prevSongTitle').innerText = artist ? `${title} — ${artist}` : title;

  const hasWords = currentLyrics.some(l => l.words && l.words.length > 0);
  const subEl = document.getElementById('prevSongSub');
  if (subEl) {
    subEl.innerText = `${hasWords ? 'Word-by-Word Karaoke' : 'Line-Synced'} • ${currentLyrics.length} righe • Metrolist Stage`;
  }

  const modal = document.getElementById('previewModal');
  modal.setAttribute('data-agent', 'v1');
  modal.classList.add('active');
  requestAnimationFrame(fitPreviewLinesToViewport);

  if (!audioLoaded) {
    const dataUrl = await window.pywebview.api.get_audio_data_url();
    if (dataUrl) {
      audioElement.src = dataUrl;
      audioLoaded = true;
    }
  }

  audioElement.currentTime = 0;
  audioElement.playbackRate = previewSpeedLevels[previewSpeedIdx] || 1.0;
  lastActiveIdx = -1;
  const viewport = document.getElementById('lyricsViewport');
  viewport.scrollTop = 0;

  audioElement.play().catch(() => {});
  setPreviewPlayingState(true);
  requestAnimationFrame(updateLoop);
}

function closePreview() {
  audioElement.pause();
  setPreviewPlayingState(false);
  document.getElementById('previewModal').classList.remove('active');
}

function togglePlayPause() {
  if (audioElement.paused) {
    audioElement.play().catch(() => {});
  } else {
    audioElement.pause();
  }
}

function skipPreview(deltaSec) {
  const dur = audioElement.duration || audioDuration || 0;
  if (!dur) return;
  const next = Math.min(dur, Math.max(0, (audioElement.currentTime || 0) + deltaSec));
  audioElement.currentTime = next;
  lastActiveIdx = -1;
}

function cyclePreviewSpeed() {
  previewSpeedIdx = (previewSpeedIdx + 1) % previewSpeedLevels.length;
  const rate = previewSpeedLevels[previewSpeedIdx];
  audioElement.playbackRate = rate;
  const btn = document.getElementById('btnPreviewSpeed');
  if (btn) btn.innerText = `${rate.toFixed(rate % 1 === 0 ? 1 : 2).replace(/0$/, '')}x`;
}

function onSeekInput(val) {
  if (audioElement.duration) {
    audioElement.currentTime = (val / 1000) * audioElement.duration;
    lastActiveIdx = -1;
  }
}

window.addEventListener('keydown', (e) => {
  const modal = document.getElementById('previewModal');
  if (!modal || !modal.classList.contains('active')) return;
  if (e.code === 'Space') {
    e.preventDefault();
    togglePlayPause();
  } else if (e.code === 'ArrowLeft') {
    e.preventDefault();
    skipPreview(-5);
  } else if (e.code === 'ArrowRight') {
    e.preventDefault();
    skipPreview(5);
  } else if (e.code === 'Escape') {
    e.preventDefault();
    closePreview();
  }
});

function parseLRC(lrcText) {
  currentLyrics = [];
  const lines = lrcText.split('\n').map(l => l.trim()).filter(l => l.length > 0);

  for (let i = 0; i < lines.length; i++) {
    const match = lines[i].match(/^\[(\d+):(\d+(?:\.\d+)?)\](.*)/);
    if (match) {
      const min = parseInt(match[1]);
      const sec = parseFloat(match[2]);
      const baseStartTime = min * 60 + sec;

      const rest = match[3];
      const agentMatch = rest.match(/\{agent:(v\d+)\}/i);
      const agent = agentMatch ? agentMatch[1].toLowerCase() : 'v1';
      const isBg = rest.includes('{bg}');

      let text = rest.replace(/\{agent:v\d+\}/gi, '').replace(/\{bg\}/gi, '').trim();
      if (isBg && text.startsWith('(') && text.endsWith(')')) {
        text = text.slice(1, -1).trim();
      }

      const words = [];
      if (i + 1 < lines.length && lines[i + 1].startsWith('<') && lines[i + 1].endsWith('>')) {
        const wordTokens = lines[i + 1].slice(1, -1).split('|');
        wordTokens.forEach(tok => {
          const parts = tok.split(':');
          if (parts.length >= 3) {
            const wText = parts.slice(0, parts.length - 2).join(':').replace(/[()]/g, '');
            const wStart = parseFloat(parts[parts.length - 2]);
            const wEnd = parseFloat(parts[parts.length - 1]);
            words.push({ text: wText, start: wStart, end: wEnd, isElongated: false, pauseAfter: 0 });
          }
        });

        for (let k = 0; k < words.length; k++) {
          if (k > 0 && words[k].start < words[k - 1].end) {
            words[k - 1].end = words[k].start;
            if (words[k - 1].end <= words[k - 1].start) {
              words[k - 1].end = words[k - 1].start + 0.04;
              words[k].start = words[k - 1].end;
            }
          }
          if (words[k].end <= words[k].start) {
            words[k].end = words[k].start + 0.04;
          }
        }

        // Rileva parole allungate (tenute vocali >= 0.72s o grafia allungata) e pause interne alla riga
        for (let k = 0; k < words.length; k++) {
          const dur = words[k].end - words[k].start;
          words[k].isElongated = (dur >= 0.72) || /([a-zàèéìòù])\1{2,}/i.test(words[k].text);
          if (k + 1 < words.length) {
            const gap = words[k + 1].start - words[k].end;
            words[k].pauseAfter = gap >= 0.35 ? gap : 0;
          }
        }

        i++;
      }

      if (!text && words.length === 0) {
        continue;
      }

      const startTime = words.length > 0 ? words[0].start : baseStartTime;
      const endTime = words.length > 0 ? words[words.length - 1].end : (baseStartTime + 3.0);
      currentLyrics.push({ startTime, endTime, text, isBg, agent, words });
    }
  }

  for (let i = 0; i < currentLyrics.length - 1; i++) {
    if (!currentLyrics[i].words || currentLyrics[i].words.length === 0) {
      const nextStart = currentLyrics[i + 1].startTime;
      if (nextStart > currentLyrics[i].startTime + 0.12) {
        currentLyrics[i].endTime = Math.max(
          currentLyrics[i].startTime + 0.1,
          Math.min(nextStart - 0.02, currentLyrics[i].startTime + 6.0)
        );
      }
    }
  }
}

function renderLyricsDOM() {
  const container = document.getElementById('lyricsViewport');
  container.innerHTML = '';

  currentLyrics.forEach((line, lineIdx) => {
    const lineEl = document.createElement('div');
    lineEl.className = `lyric-line ${line.isBg ? 'bg-sub' : 'main'} agent-${line.agent}`;
    lineEl.id = `line-${lineIdx}`;

    lineEl.onclick = () => {
      audioElement.currentTime = line.startTime;
      lastActiveIdx = -1;
    };

    if (line.words && line.words.length > 0) {
      line.words.forEach((w, wIdx) => {
        const wordWrap = document.createElement('span');
        wordWrap.className = `word${w.isElongated ? ' elongated' : ''}`;
        wordWrap.id = `word-${lineIdx}-${wIdx}`;

        wordWrap.onclick = (e) => {
          e.stopPropagation();
          audioElement.currentTime = w.start;
          lastActiveIdx = -1;
        };

        const baseSpan = document.createElement('span');
        baseSpan.className = 'word-base';
        baseSpan.innerText = w.text;

        const fillSpan = document.createElement('span');
        fillSpan.className = 'word-fill';
        fillSpan.id = `word-fill-${lineIdx}-${wIdx}`;
        fillSpan.innerText = w.text;

        wordWrap.appendChild(baseSpan);
        wordWrap.appendChild(fillSpan);
        lineEl.appendChild(wordWrap);
        if (wIdx + 1 < line.words.length) {
          lineEl.appendChild(document.createTextNode(' '));
        }

        // Indicatore visivo di pausa/respiro interno alla riga
        if (w.pauseAfter >= 0.38) {
          const pauseDot = document.createElement('span');
          pauseDot.className = 'intra-pause-dot';
          pauseDot.id = `pause-${lineIdx}-${wIdx}`;
          pauseDot.title = `Pausa vocale (${w.pauseAfter.toFixed(2)}s)`;
          lineEl.appendChild(pauseDot);
        }
      });
    } else {
      const span = document.createElement('span');
      span.className = 'word-base';
      span.innerText = line.text;
      lineEl.appendChild(span);
    }

    container.appendChild(lineEl);
  });
}

function updateLoop() {
  const previewModal = document.getElementById('previewModal');
  if (!previewModal || !previewModal.classList.contains('active')) return;

  const curTime = audioElement.currentTime || 0;
  const totalTime = audioElement.duration || audioDuration || 1;
  const pct = Math.min(100, Math.max(0, (curTime / totalTime) * 100));

  seekSlider.value = (pct / 100) * 1000;
  seekSlider.style.setProperty('--seek-pct', `${pct.toFixed(2)}%`);
  document.getElementById('lblCurTime').innerText = formatTime(curTime);
  document.getElementById('lblTotTime').innerText = formatTime(totalTime);

  let activeIdx = -1;
  let inSilence = false;
  let silStart = 0, silEnd = 0;

  if (currentLyrics.length > 0 && curTime < currentLyrics[0].startTime && currentLyrics[0].startTime > 1.2) {
    inSilence = true;
    silStart = 0;
    silEnd = currentLyrics[0].startTime;
  }

  for (let i = 0; i < currentLyrics.length; i++) {
    const l = currentLyrics[i];
    const nextL = (i + 1 < currentLyrics.length) ? currentLyrics[i + 1] : null;
    const effectiveEnd = nextL ? Math.min(l.endTime + 0.25, nextL.startTime - 0.001) : (l.endTime + 0.25);
    if (curTime >= l.startTime && curTime <= effectiveEnd) {
      if (l.isBg && nextL && !nextL.isBg && curTime >= nextL.startTime) {
        activeIdx = i + 1;
      } else {
        activeIdx = i;
      }
      break;
    } else if (l.startTime > curTime) {
      if (i > 0) {
        const prevEnd = currentLyrics[i - 1].endTime;
        if (l.startTime - prevEnd >= 1.25 && curTime >= prevEnd && curTime < l.startTime) {
          inSilence = true;
          silStart = prevEnd;
          silEnd = l.startTime;
        }
        if (activeIdx === -1) activeIdx = i - 1;
      }
      break;
    }
  }

  if (activeIdx === -1 && currentLyrics.length > 0 && curTime >= currentLyrics[currentLyrics.length - 1].endTime) {
    activeIdx = currentLyrics.length - 1;
  }

  if (activeIdx >= 0 && currentLyrics[activeIdx]) {
    const activeAgent = currentLyrics[activeIdx].agent || 'v1';
    if (previewModal.getAttribute('data-agent') !== activeAgent) {
      previewModal.setAttribute('data-agent', activeAgent);
    }
  }

  const silenceBox = document.getElementById('silenceCountdown');
  const viewport = document.getElementById('lyricsViewport');

  if (inSilence && silEnd > silStart) {
    silenceBox.classList.add('active');
    const ratio = Math.min(1, Math.max(0, (curTime - silStart) / (silEnd - silStart)));
    const remain = Math.max(0, silEnd - curTime).toFixed(1);
    document.getElementById('silenceText').innerText = `⏳ PAUSA STRUMENTALE • Attacco tra ${remain}s`;
    document.getElementById('silenceFill').style.width = `${ratio * 100}%`;
  } else {
    silenceBox.classList.remove('active');
  }

  currentLyrics.forEach((line, lIdx) => {
    const lineEl = document.getElementById(`line-${lIdx}`);
    if (!lineEl) return;

    const isCurrent = (lIdx === activeIdx) || (line.isBg && curTime >= line.startTime && curTime <= line.endTime + 0.12);
    const isFinishedHold = isCurrent && (curTime > line.endTime + 0.15);
    const dist = Math.abs(lIdx - activeIdx);

    lineEl.classList.remove('active', 'passed-hold', 'near', 'far');

    if (isFinishedHold) {
      lineEl.classList.add('passed-hold');
    } else if (isCurrent) {
      lineEl.classList.add('active');
    } else if (dist <= 2) {
      lineEl.classList.add('near');
    } else {
      lineEl.classList.add('far');
    }

    if (line.words) {
      line.words.forEach((w, wIdx) => {
        const wWrap = document.getElementById(`word-${lIdx}-${wIdx}`);
        const wFill = document.getElementById(`word-fill-${lIdx}-${wIdx}`);
        const pauseEl = document.getElementById(`pause-${lIdx}-${wIdx}`);
        if (!wWrap || !wFill) return;

        const baseClass = w.isElongated ? 'word elongated' : 'word';

        if (curTime < w.start) {
          wWrap.className = baseClass;
          wFill.style.width = '0%';
        } else if (curTime >= w.start && curTime < w.end) {
          wWrap.className = `${baseClass} active`;
          const dur = Math.max(0.05, w.end - w.start);
          const progress = Math.min(1, Math.max(0, (curTime - w.start) / dur));
          wFill.style.width = `${progress * 100}%`;
        } else {
          wWrap.className = `${baseClass} passed`;
          wFill.style.width = '100%';
        }

        if (pauseEl && wIdx + 1 < line.words.length) {
          const nextStart = line.words[wIdx + 1].start;
          if (curTime >= w.end && curTime < nextStart) {
            pauseEl.classList.add('active');
          } else {
            pauseEl.classList.remove('active');
          }
        }
      });
    }
  });

  if (activeIdx !== lastActiveIdx) {
    lastActiveIdx = activeIdx;

    if (activeIdx >= 0) {
      const activeEl = document.getElementById(`line-${activeIdx}`);
      if (activeEl) {
        const target = activeEl.offsetTop - (viewport.clientHeight / 2) + (activeEl.clientHeight / 2);
        viewport.scrollTo({ top: Math.max(0, target), behavior: 'smooth' });
      }
    } else {
      viewport.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }

  requestAnimationFrame(updateLoop);
}

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

// ==============================================================================
// AUTENTICAZIONE SERVIZI DI PUBBLICAZIONE & IDENTITÀ UNISON
// ==============================================================================
let currentUnisonKeyId = "";
let currentUnisonCuratorUrl = "";

function openExtLink(url) {
  if (!url) return;
  window.pywebview.api.open_external_url(url);
}

function openCuratorProfile() {
  if (currentUnisonCuratorUrl) {
    window.pywebview.api.open_external_url(currentUnisonCuratorUrl);
  }
}

function openAuthSettings() {
  switchTab('settings');
  switchSettingsSubTab('auth');
}

function toggleSecretVisibility(inputId, btn) {
  const inp = document.getElementById(inputId);
  if (!inp) return;
  if (inp.type === 'password') {
    inp.type = 'text';
    if (btn) btn.innerText = '🙈';
  } else {
    inp.type = 'password';
    if (btn) btn.innerText = '👁️';
  }
}

function applyPublishingState(pub, identity) {
  if (pub) {
    const setV = (id, val) => {
      const el = document.getElementById(id);
      if (el && val !== undefined && val !== null) el.value = val;
    };
    setV('pubUnisonLabel', pub.unison_label || '');
    setV('pubUnisonUrl', pub.unison_url || 'https://unison.betterlyrics.org');
    setV('pubUnisonAuthMode', pub.unison_auth_mode || 'ecdsa_identity');
    setV('pubUnisonBearer', pub.unison_bearer_token || '');
    setV('pubUnisonClientId', pub.unison_client_id || '');
    setV('pubLrcUrl', pub.lrclib_url || 'https://lrclib.net');
    setV('pubLrcAuthMode', pub.lrclib_auth_mode || 'auto_pow');
    setV('pubLrcToken', pub.lrclib_token || '');
    setV('pubLrcClientId', pub.lrclib_client_id || '');
    updateAuthFieldVisibility();
  }

  if (identity) {
    currentUnisonKeyId = identity.valid ? (identity.key_id || '') : '';
    currentUnisonCuratorUrl = identity.valid ? (identity.curator_url || '') : '';
    const keyEl = document.getElementById('unisonKeyIdDisplay');
    const badgeEl = document.getElementById('unisonAuthBadge');
    const btnCopy = document.getElementById('btnCopyKeyId');
    const btnCurator = document.getElementById('btnOpenCurator');
    const btnExpN = document.getElementById('btnExportUnisonNative');
    const btnExpJ = document.getElementById('btnExportUnisonJwk');
    const btnDel = document.getElementById('btnDeleteUnisonId');

    if (identity.valid && identity.key_id) {
      if (keyEl) keyEl.innerText = identity.key_id;
      if (badgeEl) {
        const lbl = identity.label ? ` • ${identity.label}` : '';
        badgeEl.innerText = `🔑 Identità Attiva${lbl}`;
        badgeEl.classList.add('calibrated');
        badgeEl.classList.remove('disabled');
      }
      if (btnCopy) btnCopy.disabled = false;
      if (btnCurator) btnCurator.disabled = !currentUnisonCuratorUrl;
      if (btnExpN) btnExpN.disabled = false;
      if (btnExpJ) btnExpJ.disabled = false;
      if (btnDel) btnDel.disabled = false;
    } else {
      if (keyEl) {
        keyEl.innerText = identity.crypto_available === false
          ? "Modulo 'cryptography' non installato"
          : "Nessuna chiave locale (generata al primo invio o tramite Importa)";
      }
      if (badgeEl) {
        badgeEl.innerText = identity.exists && !identity.valid ? "⚠️ Chiave non valida" : "⚪ Non configurata";
        badgeEl.classList.remove('calibrated');
        badgeEl.classList.add('disabled');
      }
      if (btnCopy) btnCopy.disabled = true;
      if (btnCurator) btnCurator.disabled = true;
      if (btnExpN) btnExpN.disabled = true;
      if (btnExpJ) btnExpJ.disabled = true;
      if (btnDel) btnDel.disabled = !identity.exists;
    }
  }
}

function updateAuthFieldVisibility() {
  const lrcMode = document.getElementById('pubLrcAuthMode')?.value || 'auto_pow';
  const lrcWrap = document.getElementById('lrcTokenWrap');
  const lrcLbl = document.getElementById('lrcTokenLabel');
  if (lrcWrap) {
    lrcWrap.style.opacity = lrcMode === 'auto_pow' ? '0.55' : '1';
  }
  if (lrcLbl) {
    if (lrcMode === 'manual_token') lrcLbl.innerText = 'X-Publish-Token Manuale:';
    else if (lrcMode === 'bearer_auth') lrcLbl.innerText = 'Bearer Token / API Key:';
    else lrcLbl.innerText = 'Token LRCLIB (non necessario in PoW Auto):';
  }

  const uMode = document.getElementById('pubUnisonAuthMode')?.value || 'ecdsa_identity';
  const uWrap = document.getElementById('unisonBearerWrap');
  if (uWrap) {
    uWrap.style.opacity = uMode === 'ecdsa_identity' ? '0.55' : '1';
  }
}

function onPublishingAuthModeChange() {
  updateAuthFieldVisibility();
  savePublishingConfig(false);
}

function readPublishingForm() {
  return {
    unison_label: (document.getElementById('pubUnisonLabel')?.value || '').trim(),
    unison_url: (document.getElementById('pubUnisonUrl')?.value || '').trim(),
    unison_auth_mode: document.getElementById('pubUnisonAuthMode')?.value || 'ecdsa_identity',
    unison_bearer_token: (document.getElementById('pubUnisonBearer')?.value || '').trim(),
    unison_client_id: (document.getElementById('pubUnisonClientId')?.value || '').trim(),
    lrclib_url: (document.getElementById('pubLrcUrl')?.value || '').trim(),
    lrclib_auth_mode: document.getElementById('pubLrcAuthMode')?.value || 'auto_pow',
    lrclib_token: (document.getElementById('pubLrcToken')?.value || '').trim(),
    lrclib_client_id: (document.getElementById('pubLrcClientId')?.value || '').trim(),
  };
}

async function savePublishingConfig(showFeedback = false) {
  const res = await window.pywebview.api.save_publishing_settings(readPublishingForm());
  if (res && res.success) {
    applyPublishingState(res.publishing, res.identity);
    if (showFeedback) {
      const box = document.getElementById('pubTestResult');
      if (box) {
        box.className = 'pub-test-result ok';
        box.innerText = '✅ Configurazione servizi salvata.';
      }
    }
  }
}

async function resetPublishingConfig() {
  if (!confirm("Ripristinare gli URL, i client ID e le modalità di autenticazione predefiniti? (La tua chiave Unison locale non verrà eliminata)")) return;
  const res = await window.pywebview.api.reset_publishing_settings();
  if (res && res.success) {
    applyPublishingState(res.publishing, res.identity);
    const box = document.getElementById('pubTestResult');
    if (box) {
      box.className = 'pub-test-result ok';
      box.innerText = '↺ Impostazioni servizi ripristinate ai valori predefiniti.';
    }
  }
}

async function copyUnisonKeyId() {
  if (!currentUnisonKeyId) return;
  await navigator.clipboard.writeText(currentUnisonKeyId);
  const box = document.getElementById('pubTestResult');
  if (box) {
    box.className = 'pub-test-result ok';
    box.innerText = `📋 Key ID copiato negli appunti (${currentUnisonKeyId.slice(0, 12)}...).`;
  }
}

function togglePasteIdentityBox(forceState) {
  const box = document.getElementById('pasteIdentityBox');
  if (!box) return;
  const show = (forceState !== undefined) ? forceState : (box.style.display === 'none');
  box.style.display = show ? 'block' : 'none';
  scheduleWindowFit(true);
}

async function importUnisonIdentityFile() {
  const res = await window.pywebview.api.import_unison_identity_file();
  if (!res) return;
  if (res.success) {
    applyPublishingState(null, res.identity);
    window.appendLog(`🔑 Identità Unison importata: ${res.identity?.key_id?.slice(0, 14)}...`);
    alert(`🔑 Identità Unison importata con successo!\n\nKey ID:\n${res.identity.key_id}`);
  } else if (res.error) {
    alert(`Errore importazione identità:\n${res.error}`);
  }
}

async function applyPastedUnisonIdentity() {
  const raw = document.getElementById('pasteIdentityText')?.value || '';
  const lbl = document.getElementById('pubUnisonLabel')?.value || '';
  const res = await window.pywebview.api.import_unison_identity_text(raw, lbl);
  if (res && res.success) {
    document.getElementById('pasteIdentityText').value = '';
    togglePasteIdentityBox(false);
    applyPublishingState(res.publishing, res.identity);
    window.appendLog(`🔑 Identità Unison applicata: ${res.identity?.key_id?.slice(0, 14)}...`);
    alert(`🔑 Identità Unison applicata con successo!\n\nKey ID:\n${res.identity.key_id}`);
  } else {
    alert(`Errore chiave identità:\n${res?.error || 'Formato non valido'}`);
  }
}

async function exportUnisonIdentity(fmt = 'metrosync') {
  const res = await window.pywebview.api.export_unison_identity_file(fmt);
  if (!res) return;
  if (res.success) {
    alert(`📤 Identità Unison esportata in:\n${res.path}\n\n⚠️ Conserva questo file al sicuro: contiene la tua chiave privata.`);
  } else if (res.error) {
    alert(`Errore esportazione:\n${res.error}`);
  }
}

async function generateNewUnisonIdentity() {
  if (currentUnisonKeyId) {
    const ok = confirm("Generare una NUOVA chiave ECDSA sostituirà l'identità Unison attuale.\nLe future pubblicazioni avranno un nuovo Key ID.\n\nVuoi continuare?");
    if (!ok) return;
  }
  const lbl = document.getElementById('pubUnisonLabel')?.value || '';
  const res = await window.pywebview.api.generate_unison_identity(lbl);
  if (res && res.success) {
    applyPublishingState(res.publishing, res.identity);
    window.appendLog(`🔄 Nuova identità Unison generata: ${res.identity?.key_id?.slice(0, 14)}...`);
    alert(`🔄 Nuova identità Unison generata!\n\nKey ID:\n${res.identity.key_id}`);
  } else {
    alert(`Errore generazione chiave:\n${res?.error || 'Errore sconosciuto'}`);
  }
}

async function deleteUnisonIdentity() {
  if (!confirm("Rimuovere il file identità Unison locale? (Potrai sempre reimportarlo o rigenerarlo in seguito)")) return;
  const res = await window.pywebview.api.delete_unison_identity();
  if (res && res.success) {
    applyPublishingState(null, res.identity);
    window.appendLog("🗑️ Identità Unison locale rimossa.");
  } else if (res?.error) {
    alert(`Errore rimozione:\n${res.error}`);
  }
}

async function testPublishingAuth() {
  const btn = document.getElementById('btnTestPubAuth');
  const box = document.getElementById('pubTestResult');
  if (btn) btn.disabled = true;
  if (box) {
    box.className = 'pub-test-result';
    box.innerText = '⏳ Verifica connessione e autenticazione in corso...';
  }
  try {
    const res = await window.pywebview.api.test_publishing_services(readPublishingForm());
    if (res && res.success) {
      const lOk = res.lrclib?.ok;
      const uOk = res.unison?.ok;
      const lTxt = `${lOk ? '✅' : '❌'} LRCLIB: ${res.lrclib?.message || ''}`;
      const uTxt = `${uOk ? '✅' : '❌'} Unison: ${res.unison?.message || ''}`;
      if (box) {
        box.className = `pub-test-result ${(lOk && uOk) ? 'ok' : 'err'}`;
        box.innerText = `${lTxt}  |  ${uTxt}`;
      }
    }
  } catch (e) {
    if (box) {
      box.className = 'pub-test-result err';
      box.innerText = `❌ Errore test: ${e}`;
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

// ==============================================================================
// RICERCA ONLINE & DOWNLOAD LOCALE (METROSYNC / METROLIST)
// ==============================================================================
let searchResultsCache = [];
let selectedSearchIdx = -1;
let selectedSearchDetail = null;
let currentSearchPreviewFmt = 'lrc';

function openSearchLyricsModal() {
  const title = (document.getElementById('metaTitle')?.value || '').trim();
  const artist = (document.getElementById('metaArtist')?.value || '').trim();
  const yt = (document.getElementById('metaYoutube')?.value || '').trim();

  document.getElementById('searchTitle').value = title;
  document.getElementById('searchArtist').value = artist;
  document.getElementById('searchYoutube').value = yt;
  document.getElementById('searchResultsList').innerHTML = '';
  document.getElementById('searchPreviewCode').innerText = '';
  document.getElementById('searchDetailMeta').innerText = 'Seleziona un risultato a sinistra per visualizzare l\'anteprima convertita.';
  setDownloadButtonsEnabled(false);
  selectedSearchIdx = -1;
  selectedSearchDetail = null;

  document.getElementById('searchLyricsModal').classList.add('active');
  scheduleWindowFit(true);
  if (title || artist || yt) {
    runOnlineLyricsSearch();
  } else {
    setSearchStatus('Inserisci titolo, artista o YouTube ID e clicca su "Cerca Online".', '');
  }
}

function closeSearchLyricsModal() {
  document.getElementById('searchLyricsModal').classList.remove('active');
  scheduleWindowFit(true);
}

function setSearchStatus(msg, state = '') {
  const el = document.getElementById('searchStatusBanner');
  if (!el) return;
  el.className = `search-status-banner${state ? ' ' + state : ''}`;
  el.innerText = msg;
}

function setDownloadButtonsEnabled(enabled) {
  ['btnImportAsTxt', 'btnDlAuto', 'btnDlAs', 'btnDlStudioOnly'].forEach(id => {
    const b = document.getElementById(id);
    if (b) b.disabled = !enabled;
  });
}

async function runOnlineLyricsSearch() {
  const title = (document.getElementById('searchTitle')?.value || '').trim();
  const artist = (document.getElementById('searchArtist')?.value || '').trim();
  const youtube = (document.getElementById('searchYoutube')?.value || '').trim();
  const source = document.getElementById('searchSource')?.value || 'all';

  if (!title && !artist && !youtube) {
    setSearchStatus('Inserisci almeno un titolo, un artista o un Video ID YouTube.', 'err');
    return;
  }

  const btn = document.getElementById('btnRunSearch');
  if (btn) btn.disabled = true;
  setSearchStatus('🔍 Ricerca in corso su BetterLyrics / Unison e LRCLIB...', '');
  document.getElementById('searchResultsList').innerHTML = '';
  document.getElementById('searchPreviewCode').innerText = '';
  setDownloadButtonsEnabled(false);
  selectedSearchIdx = -1;
  selectedSearchDetail = null;

  try {
    const res = await window.pywebview.api.search_online_lyrics({
      title,
      artist,
      album: (document.getElementById('metaAlbum')?.value || '').trim(),
      youtube,
      source,
      duration: audioDuration || 0
    });

    if (!res || !res.success) {
      setSearchStatus(`❌ ${res?.error || 'Errore durante la ricerca.'}`, 'err');
      return;
    }

    searchResultsCache = res.results || [];
    renderSearchResults(searchResultsCache);

    const warnTxt = (res.errors && res.errors.length) ? ` (Avvisi: ${res.errors.join(' • ')})` : '';
    if (searchResultsCache.length === 0) {
      setSearchStatus(`Nessun risultato trovato.${warnTxt}`, 'err');
    } else {
      setSearchStatus(`✅ Trovati ${searchResultsCache.length} risultati.${warnTxt}`, 'ok');
      selectSearchResult(0);
    }
  } catch (err) {
    setSearchStatus(`❌ Errore ricerca: ${err}`, 'err');
  } finally {
    if (btn) btn.disabled = false;
  }
}

function renderSearchResults(results) {
  const list = document.getElementById('searchResultsList');
  list.innerHTML = '';

  results.forEach((item, idx) => {
    const card = document.createElement('div');
    card.className = 'search-result-item';
    card.id = `sr-item-${idx}`;
    card.onclick = () => selectSearchResult(idx);

    const top = document.createElement('div');
    top.className = 'sr-top';

    const left = document.createElement('div');
    const titleEl = document.createElement('div');
    titleEl.className = 'sr-title';
    titleEl.innerText = item.title || 'Senza Titolo';
    const artistEl = document.createElement('div');
    artistEl.className = 'sr-artist';
    artistEl.innerText = `${item.artist || 'Artista sconosciuto'}${item.album ? ' • ' + item.album : ''}`;
    left.appendChild(titleEl);
    left.appendChild(artistEl);
    top.appendChild(left);

    if (item.duration > 0) {
      const durEl = document.createElement('span');
      durEl.style.cssText = 'font-size:11px;font-family:JetBrains Mono,monospace;color:#94a3b8;white-space:nowrap;';
      durEl.innerText = formatTime(item.duration);
      top.appendChild(durEl);
    }
    card.appendChild(top);

    const metaRow = document.createElement('div');
    metaRow.className = 'sr-meta';

    const provPill = document.createElement('span');
    provPill.className = `sr-pill ${item.provider === 'unison' ? 'sr-pill-unison' : 'sr-pill-lrclib'}`;
    provPill.innerText = item.provider === 'unison' ? '🎵 Unison' : '🚀 LRCLIB';
    metaRow.appendChild(provPill);

    const syncPill = document.createElement('span');
    if (item.sync_level === 'word') {
      syncPill.className = 'sr-pill sr-pill-word';
      syncPill.innerText = '✨ Word-by-Word';
    } else if (item.sync_level === 'line') {
      syncPill.className = 'sr-pill sr-pill-line';
      syncPill.innerText = '⏱️ Line-Synced';
    } else {
      syncPill.className = 'sr-pill sr-pill-plain';
      syncPill.innerText = '📄 Solo Testo';
    }
    metaRow.appendChild(syncPill);

    if (item.exact_video_match) {
      const matchPill = document.createElement('span');
      matchPill.className = 'sr-pill sr-pill-match';
      matchPill.innerText = '🎯 Match YouTube';
      metaRow.appendChild(matchPill);
    } else if (audioDuration > 0 && item.duration > 0 && Math.abs(audioDuration - item.duration) <= 3.0) {
      const durPill = document.createElement('span');
      durPill.className = 'sr-pill sr-pill-match';
      durPill.innerText = '⏱️ Durata OK';
      metaRow.appendChild(durPill);
    }

    card.appendChild(metaRow);

    const extraParts = [];
    if (item.video_id) extraParts.push(`YT: ${item.video_id}`);
    if (item.submitter) extraParts.push(`Curator: ${item.submitter}`);
    if (item.preview_snippet) extraParts.push(item.preview_snippet);
    if (extraParts.length > 0) {
      const extraEl = document.createElement('div');
      extraEl.className = 'sr-extra';
      extraEl.innerText = extraParts.join(' • ');
      card.appendChild(extraEl);
    }

    list.appendChild(card);
  });
}

async function selectSearchResult(idx) {
  const item = searchResultsCache[idx];
  if (!item) return;

  selectedSearchIdx = idx;
  document.querySelectorAll('.search-result-item').forEach((el, i) => {
    el.classList.toggle('active', i === idx);
  });

  setDownloadButtonsEnabled(false);
  document.getElementById('searchDetailMeta').innerText = '⏳ Download e conversione nel formato MetroSync / Metrolist...';
  document.getElementById('searchPreviewCode').innerText = '';

  const targetFmt = document.querySelector('input[name="settingFormat"]:checked')?.value || 'metrolist';
  try {
    const detail = await window.pywebview.api.fetch_online_lyrics_detail({
      provider: item.provider,
      id: item.id,
      video_id: item.video_id,
      raw_item: item.raw_item,
      form_meta: getFormMetadata(),
      format: targetFmt
    });

    if (!detail || !detail.success) {
      document.getElementById('searchDetailMeta').innerText = `❌ ${detail?.error || 'Errore caricamento dettagli.'}`;
      return;
    }

    selectedSearchDetail = detail;
    const syncDesc = detail.sync_level === 'word'
      ? `✨ Word-by-Word (${detail.line_count} righe, ${detail.word_count} parole)`
      : (detail.sync_level === 'line'
        ? `⏱️ Line-Synced (${detail.line_count} righe)`
        : '📄 Testo Semplice (non sincronizzato)');

    document.getElementById('searchDetailMeta').innerText =
      `${item.provider_label} • ${syncDesc} • Formato locale: ${targetFmt.toUpperCase()}`;

    const dlSelect = document.getElementById('downloadFormatSelect');
    const isSynced = detail.sync_level !== 'plain' && Boolean(detail.lrc);
    if (dlSelect) {
      Array.from(dlSelect.options).forEach(opt => {
        if (opt.value !== 'txt') opt.disabled = !isSynced;
      });
      dlSelect.value = isSynced ? 'metrolist_bundle' : 'txt';
    }

    if (!isSynced) {
      const plainBtn = document.querySelector('.fmt-tab[onclick*="\'plain\'"]');
      switchSearchPreviewTab('plain', plainBtn);
    } else {
      const lrcBtn = document.querySelector('.fmt-tab[onclick*="\'lrc\'"]');
      switchSearchPreviewTab('lrc', lrcBtn);
    }

    setDownloadButtonsEnabled(true);
  } catch (err) {
    document.getElementById('searchDetailMeta').innerText = `❌ Errore: ${err}`;
  }
}

async function importSelectedLyricsAsTxt() {
  if (selectedSearchIdx < 0 || !selectedSearchDetail) return;
  const item = searchResultsCache[selectedSearchIdx];
  const plainText = (selectedSearchDetail.plain || '').trim();
  if (!plainText) {
    alert("Il risultato selezionato non contiene testo utilizzabile.");
    return;
  }

  setDownloadButtonsEnabled(false);
  try {
    const localFmt = document.querySelector('input[name="settingFormat"]:checked')?.value || 'metrolist';
    const res = await window.pywebview.api.download_and_save_online_lyrics({
      provider: item.provider,
      id: item.id,
      video_id: item.video_id,
      raw_item: item.raw_item,
      fetched: selectedSearchDetail.fetched,
      target_format: 'txt',
      save_mode: 'load_only',
      load_into_studio: true,
      form_meta: getFormMetadata(),
      format: localFmt
    });

    if (!res || !res.success) {
      alert(`Errore importazione testo:\n${res?.error || 'Errore sconosciuto'}`);
      setDownloadButtonsEnabled(true);
      return;
    }

    const labelTitle = res.meta?.title || item.title || 'Testo online';
    const labelArtist = res.meta?.artist || item.artist || '';
    const displayLabel = labelArtist
      ? `Testo importato (${item.provider.toUpperCase()}): ${labelArtist} - ${labelTitle} [.txt]`
      : `Testo importato (${item.provider.toUpperCase()}): ${labelTitle} [.txt]`;

    document.getElementById('lyricsPath').value = displayLabel;
    window.tempLyricsContent = res.plain || plainText;
    if (res.meta) applyPrefilledMetadata(res.meta, false);

    const btnExp = document.getElementById('btnExportFiles');
    if (btnExp) btnExp.disabled = false;

    const lineCount = (window.tempLyricsContent.split('\n').filter(l => l.trim().length > 0)).length;
    window.appendLog(`📝 Testo importato da ${item.provider_label} come .txt (${lineCount} righe) — pronto per '⚡ AVVIA SYNC'!`);
    closeSearchLyricsModal();
  } catch (err) {
    alert(`Errore importazione: ${err}`);
    setDownloadButtonsEnabled(true);
  }
}

function switchSearchPreviewTab(fmt, btnEl) {
  currentSearchPreviewFmt = fmt;
  document.querySelectorAll('.fmt-tab').forEach(b => b.classList.remove('active'));
  if (btnEl) btnEl.classList.add('active');

  const codeEl = document.getElementById('searchPreviewCode');
  if (!codeEl || !selectedSearchDetail) return;
  const content = selectedSearchDetail[fmt] || '';
  codeEl.innerText = content || '(Formato non disponibile per testo non sincronizzato: usa "Testo (.txt)" o esegui Sync AI)';
}

async function downloadSelectedLyrics(saveMode) {
  if (selectedSearchIdx < 0 || !selectedSearchDetail) return;
  const item = searchResultsCache[selectedSearchIdx];
  const targetFormat = document.getElementById('downloadFormatSelect')?.value || 'metrolist_bundle';
  const localFmt = document.querySelector('input[name="settingFormat"]:checked')?.value || 'metrolist';

  setDownloadButtonsEnabled(false);
  try {
    const res = await window.pywebview.api.download_and_save_online_lyrics({
      provider: item.provider,
      id: item.id,
      video_id: item.video_id,
      raw_item: item.raw_item,
      fetched: selectedSearchDetail.fetched,
      target_format: targetFormat,
      save_mode: saveMode,
      load_into_studio: true,
      form_meta: getFormMetadata(),
      format: localFmt
    });

    if (!res) return;
    if (res.cancelled) {
      setDownloadButtonsEnabled(true);
      return;
    }
    if (!res.success) {
      alert(`Errore download:\n${res.error || 'Errore sconosciuto'}`);
      setDownloadButtonsEnabled(true);
      return;
    }

    if (res.loaded_mode === 'synced') {
      applySyncedLyricsData(res);
    } else if (res.loaded_mode === 'plain') {
      document.getElementById('lyricsPath').value = res.path || 'Testo scaricato online';
      window.tempLyricsContent = res.plain || '';
      if (res.meta) applyPrefilledMetadata(res.meta, false);
      const btnExp = document.getElementById('btnExportFiles');
      if (btnExp) btnExp.disabled = false;
      window.appendLog(`📄 Testo semplice pronto per Sync AI: ${res.path}`);
    }

    if (res.saved_paths && res.saved_paths.length > 0) {
      window.appendLog(`💾 File salvati in locale:\n${res.saved_paths.map(p => '   • ' + p).join('\n')}`);
      alert(`✅ Testo scaricato con successo!\n\nFile salvati:\n${res.saved_paths.join('\n')}`);
    } else {
      alert(`✅ Testo caricato in Studio!`);
    }
    closeSearchLyricsModal();
  } catch (err) {
    alert(`Errore download: ${err}`);
    setDownloadButtonsEnabled(true);
  }
}

// ==============================================================================
// ESPORTAZIONE RAPIDA FILE LOCALI
// ==============================================================================
function openExportModal() {
  const isSynced = Boolean(window.generatedLrc);
  const sel = document.getElementById('exportFormatSelect');
  if (sel) {
    Array.from(sel.options).forEach(opt => {
      if (opt.value !== 'txt') opt.disabled = !isSynced;
    });
    sel.value = isSynced ? 'metrolist_bundle' : 'txt';
  }
  document.getElementById('exportModal').classList.add('active');
}

function closeExportModal() {
  document.getElementById('exportModal').classList.remove('active');
}

async function runExportCurrentLyrics() {
  const targetFormat = document.getElementById('exportFormatSelect')?.value || 'metrolist_bundle';
  const res = await window.pywebview.api.export_current_lyrics({
    target_format: targetFormat,
    form_meta: getFormMetadata()
  });
  if (!res || res.cancelled) return;
  if (res.success) {
    closeExportModal();
    window.appendLog(`💾 Esportati file locali:\n${res.saved_paths.map(p => '   • ' + p).join('\n')}`);
    alert(`✅ Esportazione completata!\n\n${res.saved_paths.join('\n')}`);
  } else {
    alert(`Errore esportazione:\n${res.error || 'Errore sconosciuto'}`);
  }
}

// ==============================================================================
// GESTORE & DOWNLOAD MODELLI WHISPER LOCALI
// ==============================================================================
let modelCatalogCache = [];
let modelDownloadActive = false;

function populateModelDropdown(availableModels, selectedPath) {
  const modelSelect = document.getElementById('settingModelSelect');
  if (!modelSelect) return;
  modelSelect.innerHTML = '';
  let matchedPreset = false;
  const curTarget = (selectedPath || document.getElementById('settingModel')?.value || '').toLowerCase();

  (availableModels || []).forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.path;
    opt.innerText = m.name;
    if (m.path.toLowerCase() === curTarget) {
      opt.selected = true;
      matchedPreset = true;
    }
    modelSelect.appendChild(opt);
  });

  const customOpt = document.createElement('option');
  customOpt.value = 'custom';
  customOpt.innerText = '📁 Altro file personalizzato (.pt)...';
  if (!matchedPreset) customOpt.selected = true;
  modelSelect.appendChild(customOpt);
}

function renderModelCatalog(catalog) {
  modelCatalogCache = catalog || [];
  const list = document.getElementById('modelDownloadList');
  if (!list) return;
  list.innerHTML = '';

  const activePath = (document.getElementById('settingModel')?.value || '').toLowerCase();
  let missingCount = 0;

  modelCatalogCache.forEach(item => {
    if (!item.installed) missingCount++;

    const row = document.createElement('div');
    row.className = 'model-dl-item';

    const info = document.createElement('div');
    info.className = 'model-dl-info';

    const titleRow = document.createElement('div');
    titleRow.className = 'model-dl-title-row';

    const nameEl = document.createElement('span');
    nameEl.className = 'model-dl-name';
    nameEl.innerText = item.label;
    titleRow.appendChild(nameEl);

    const badge = document.createElement('span');
    if (item.downloading) {
      badge.className = 'sr-pill sr-pill-word';
      badge.innerText = '⏳ Download...';
    } else if (item.installed) {
      badge.className = 'sr-pill sr-pill-match';
      badge.innerText = `✅ Scaricato (${item.size_gb} GB)`;
    } else {
      badge.className = 'sr-pill sr-pill-plain';
      badge.innerText = '⚪ Non presente';
    }
    titleRow.appendChild(badge);

    if (item.recommended) {
      const rec = document.createElement('span');
      rec.className = 'sr-pill sr-pill-word';
      rec.innerText = 'Consigliato';
      titleRow.appendChild(rec);
    }

    const descEl = document.createElement('div');
    descEl.className = 'model-dl-desc';
    descEl.innerText = item.description;

    info.appendChild(titleRow);
    info.appendChild(descEl);
    row.appendChild(info);

    const actions = document.createElement('div');
    actions.className = 'model-dl-actions';

    if (item.installed) {
      const isSelected = activePath === (item.path || '').toLowerCase();
      const useBtn = document.createElement('button');
      useBtn.className = isSelected ? 'btn-micro' : 'btn-cyan-outline';
      useBtn.disabled = isSelected;
      useBtn.innerText = isSelected ? '✅ In uso' : '🎯 Usa questo';
      useBtn.onclick = () => selectDownloadedModel(item.path);
      actions.appendChild(useBtn);
    } else {
      const dlBtn = document.createElement('button');
      dlBtn.className = 'btn-cyan-outline';
      dlBtn.disabled = modelDownloadActive;
      dlBtn.innerText = item.downloading ? '⏳ In corso...' : '⬇️ Scarica';
      dlBtn.onclick = () => downloadSingleModel(item.id);
      actions.appendChild(dlBtn);
    }

    row.appendChild(actions);
    list.appendChild(row);
  });

  const btnAll = document.getElementById('btnDownloadAllModels');
  if (btnAll) {
    btnAll.disabled = modelDownloadActive || missingCount === 0;
    btnAll.innerText = missingCount === 0
      ? '✅ Tutti i modelli scaricati'
      : `⬇️ Scarica Tutti i Mancanti (${missingCount})`;
  }
}

function selectDownloadedModel(modelPath) {
  document.getElementById('settingModel').value = modelPath;
  populateModelDropdown(appConfig?.available_models || [], modelPath);
  autoSaveSettings();
  renderModelCatalog(modelCatalogCache);
}

async function downloadSingleModel(modelId) {
  modelDownloadActive = true;
  renderModelCatalog(modelCatalogCache);
  const res = await window.pywebview.api.download_whisper_models([modelId]);
  if (res && !res.success && res.error) {
    modelDownloadActive = false;
    renderModelCatalog(modelCatalogCache);
    alert(`Errore avvio download:\n${res.error}`);
  }
}

async function downloadAllMissingModels() {
  modelDownloadActive = true;
  renderModelCatalog(modelCatalogCache);
  const res = await window.pywebview.api.download_whisper_models(null);
  if (res && res.nothing_to_do) {
    modelDownloadActive = false;
    renderModelCatalog(res.catalog || modelCatalogCache);
    alert("Tutti i modelli ufficiali sono già presenti in locale!");
  } else if (res && !res.success && res.error) {
    modelDownloadActive = false;
    renderModelCatalog(modelCatalogCache);
    alert(`Errore avvio download:\n${res.error}`);
  }
}

async function cancelModelDownload() {
  await window.pywebview.api.cancel_model_download();
}

window.modelDownloadProgress = function(ev) {
  if (!ev) return;
  const box = document.getElementById('modelDownloadProgressBox');
  const txt = document.getElementById('modelDlStatusText');
  const bar = document.getElementById('modelDlProgressBar');

  if (ev.available_models && appConfig) {
    appConfig.available_models = ev.available_models;
    const curVal = document.getElementById('settingModel')?.value || '';
    if ((!curVal || !curVal.endsWith('.pt')) && ev.path) {
      document.getElementById('settingModel').value = ev.path;
      autoSaveSettings();
    }
    populateModelDropdown(appConfig.available_models, document.getElementById('settingModel')?.value);
  }

  if (ev.status === 'downloading') {
    modelDownloadActive = true;
    if (box) box.style.display = 'block';
    if (bar) bar.style.width = `${ev.pct || 0}%`;
    if (txt) {
      txt.innerText = `⬇️ ${ev.filename}: ${ev.pct || 0}% (${ev.downloaded_mb || 0} / ${ev.total_mb || 0} MB • ${ev.speed_mb_s || 0} MB/s)`;
    }
  } else if (ev.status === 'verifying') {
    if (box) box.style.display = 'block';
    if (bar) bar.style.width = '100%';
    if (txt) txt.innerText = `🔐 Verifica integrità SHA-256 per ${ev.filename}...`;
  } else if (ev.status === 'completed') {
    if (txt) txt.innerText = `✅ ${ev.filename} scaricato e verificato!`;
  } else if (ev.status === 'cancelled') {
    modelDownloadActive = false;
    if (box) box.style.display = 'none';
  } else if (ev.status === 'error') {
    modelDownloadActive = false;
    if (box) box.style.display = 'none';
    alert(`Errore download ${ev.filename || 'modello'}:\n${ev.error || 'Errore sconosciuto'}`);
  } else if (ev.status === 'all_done') {
    modelDownloadActive = false;
    if (box) box.style.display = 'none';
  }

  if (ev.catalog) {
    renderModelCatalog(ev.catalog);
  }
};

function updateDemucsCatalogUI(demucsCatalog) {
  if (!demucsCatalog) return;
  if (appConfig) appConfig.demucs_catalog = demucsCatalog;
  const chosen = document.getElementById('settingDemucsModel')?.value || 'htdemucs';
  const curInfo = demucsCatalog[chosen];
  const ftInfo = demucsCatalog.htdemucs_ft;

  const badge = document.getElementById('demucsModelBadge');
  if (badge && curInfo) {
    if (curInfo.installed) {
      badge.className = 'sr-pill sr-pill-match';
      badge.innerText = `✅ ${chosen} pronto`;
    } else {
      badge.className = 'sr-pill sr-pill-plain';
      badge.innerText = `⚪ ${chosen} da scaricare`;
    }
  }
  const btnFt = document.getElementById('btnDownloadDemucsFt');
  if (btnFt && ftInfo) {
    btnFt.disabled = Boolean(ftInfo.installed);
    btnFt.innerText = ftInfo.installed ? '✅ htdemucs_ft scaricato' : '⬇️ Scarica htdemucs_ft';
  }
}

function onDemucsModelChange() {
  autoSaveSettings();
  if (appConfig?.demucs_catalog) {
    updateDemucsCatalogUI(appConfig.demucs_catalog);
  }
}

async function downloadDemucsModel(modelName) {
  const btn = document.getElementById('btnDownloadDemucsFt');
  if (btn) {
    btn.disabled = true;
    btn.innerText = '⏳ Download htdemucs_ft...';
  }
  await window.pywebview.api.download_demucs_model(modelName || 'htdemucs_ft');
}

window.demucsDownloadProgress = function(ev) {
  if (!ev) return;
  const btn = document.getElementById('btnDownloadDemucsFt');
  if (ev.status === 'downloading') {
    if (btn) {
      btn.disabled = true;
      btn.innerText = '⏳ Download in corso...';
    }
  } else if (ev.status === 'completed') {
    if (ev.demucs_catalog) updateDemucsCatalogUI(ev.demucs_catalog);
    alert(`✅ Modello Demucs '${ev.model}' scaricato e pronto per la massima precisione vocale!`);
  } else if (ev.status === 'error') {
    if (ev.demucs_catalog) updateDemucsCatalogUI(ev.demucs_catalog);
    alert(`❌ Errore download Demucs:\n${ev.error || 'Errore sconosciuto'}`);
  }
};

// ==============================================================================
// ADATTAMENTO DINAMICO FINESTRA & INTERFACCIA (ZERO BARRE DI SCROLL / NO WRAP)
// ==============================================================================
function fitPreviewLinesToViewport() {
  const modal = document.getElementById('previewModal');
  const viewport = document.getElementById('lyricsViewport');
  if (!modal || !viewport || !modal.classList.contains('active')) return;
  const availW = Math.max(220, (viewport.clientWidth - 64) / 1.08);
  const lines = viewport.querySelectorAll('.lyric-line');
  lines.forEach(lineEl => {
    const isBg = lineEl.classList.contains('bg-sub');
    const basePx = isBg ? 18 : 26;
    lineEl.style.fontSize = `${basePx}px`;
    lineEl.querySelectorAll('.word').forEach(w => { w.style.margin = ''; });
    const sw = lineEl.scrollWidth;
    if (sw > availW && sw > 0) {
      const ratio = availW / sw;
      const newPx = Math.max(10.5, Math.floor(basePx * ratio * 10) / 10);
      lineEl.style.fontSize = `${newPx}px`;
      if (ratio < 0.78) {
        const mPx = Math.max(1.5, Math.round(4 * ratio * 10) / 10);
        lineEl.querySelectorAll('.word').forEach(w => {
          w.style.margin = `0 ${mPx}px`;
        });
      }
    }
  });
}

function fitSingleLineRow(el, minFontPx = 9.0) {
  if (!el || el.offsetParent === null) return;
  el.style.fontSize = '';
  el.style.gap = '';
  const items = el.querySelectorAll('button, .btn, .btn-micro, .btn-project');
  items.forEach(b => {
    b.style.fontSize = '';
    b.style.paddingLeft = '';
    b.style.paddingRight = '';
  });

  const avail = el.clientWidth;
  if (avail <= 0 || el.scrollWidth <= avail) return;

  const baseFs = parseFloat(window.getComputedStyle(el).fontSize) || 11.5;
  const ratio = avail / el.scrollWidth;
  const nextFs = Math.max(minFontPx, Math.floor(baseFs * ratio * 10) / 10);
  el.style.fontSize = `${nextFs}px`;

  if (el.scrollWidth > avail && items.length > 0) {
    el.style.gap = '5px';
    items.forEach(b => {
      const bf = parseFloat(window.getComputedStyle(b).fontSize) || 11;
      const bNext = Math.max(minFontPx, Math.floor(bf * ratio * 10) / 10);
      b.style.fontSize = `${bNext}px`;
      b.style.paddingLeft = '8px';
      b.style.paddingRight = '8px';
    });
  }
}

function fitAllSingleLineRows() {
  [
    '.quickstart-strip',
    '.search-help-strip',
    '.actions-row',
    '.auth-btn-group',
    '.header-actions',
    '.preview-fmt-tabs',
    '.search-download-actions',
    '.settings-subnav'
  ].forEach(sel => {
    document.querySelectorAll(sel).forEach(el => fitSingleLineRow(el, 9.0));
  });
  fitPreviewLinesToViewport();
}

function measureActiveLayoutNeeds(probeWidth = 0) {
  const container = document.querySelector('.app-container');
  const header = document.querySelector('.app-header');
  const nav = document.querySelector('.nav-tabs');
  if (!container) return { minW: 1140, minH: 860, idealW: 1140, idealH: 860 };

  const tabSettings = document.getElementById('tabSettings');
  const tabStudio = document.getElementById('tabStudio');
  const activeTab = (tabSettings && tabSettings.classList.contains('active')) ? tabSettings : tabStudio;

  const prevZoom = container.style.zoom;
  const prevWidth = container.style.width;
  const prevHeight = container.style.height;
  const prevTabFlex = activeTab ? activeTab.style.flex : '';

  container.style.zoom = '';
  container.style.width = probeWidth > 0 ? `${probeWidth}px` : '100%';
  container.style.height = 'auto';
  if (activeTab) activeTab.style.flex = 'none';

  fitAllSingleLineRows();

  const headerBottomPad = 24;
  const topChromeH = (header ? header.offsetHeight : 44) + (nav ? nav.offsetHeight : 38) + 18 + headerBottomPad;

  let minW = Math.max(960, container.scrollWidth);
  let minH = 720;
  let idealW = Math.max(1140, container.scrollWidth);
  let idealH = 860;

  if (tabSettings && tabSettings.classList.contains('active')) {
    const subnav = tabSettings.querySelector('.settings-subnav');
    const subPane = tabSettings.querySelector('.settings-subtab-pane.active');
    const subnavH = subnav ? (subnav.offsetHeight + 12) : 0;
    const paneH = subPane ? Math.max(subPane.offsetHeight, subPane.scrollHeight) : tabSettings.scrollHeight;
    minW = Math.max(960, container.scrollWidth);
    minH = topChromeH + subnavH + paneH;
    idealW = Math.max(1140, minW);
    idealH = Math.max(700, minH);
  } else if (tabStudio && tabStudio.classList.contains('active')) {
    let fixedStudioH = 0;
    Array.from(tabStudio.children).forEach(ch => {
      if (ch.classList.contains('log-card')) {
        fixedStudioH += 92;
      } else {
        const cs = window.getComputedStyle(ch);
        fixedStudioH += ch.offsetHeight + (parseFloat(cs.marginBottom) || 0);
      }
    });
    minW = Math.max(980, container.scrollWidth);
    minH = topChromeH + fixedStudioH;
    idealW = Math.max(1140, minW);
    idealH = Math.max(850, topChromeH + fixedStudioH + 58);
  }

  if (activeTab) activeTab.style.flex = prevTabFlex;
  container.style.zoom = prevZoom;
  container.style.width = prevWidth;
  container.style.height = prevHeight;

  return {
    minW: Math.ceil(minW),
    minH: Math.ceil(minH),
    idealW: Math.ceil(idealW),
    idealH: Math.ceil(idealH)
  };
}

function applyViewportFit() {
  const container = document.querySelector('.app-container');
  if (!container) return;

  const vw = Math.max(320, window.innerWidth || document.documentElement.clientWidth || 1140);
  const vh = Math.max(320, window.innerHeight || document.documentElement.clientHeight || 860);

  const m = measureActiveLayoutNeeds(0);
  let zoom = 1;
  if (m.minW > vw || m.minH > vh) {
    zoom = Math.max(0.55, Math.min(vw / m.minW, vh / m.minH) - 0.004);
  }

  if (zoom < 0.996) {
    const zStr = zoom.toFixed(4);
    container.style.zoom = zStr;
    container.style.width = `${Math.floor(vw / zoom)}px`;
    container.style.height = `${Math.floor(vh / zoom)}px`;
  } else {
    container.style.zoom = '';
    container.style.width = '';
    container.style.height = '';
  }

  fitAllSingleLineRows();
}

let _isFittingWindow = false;
async function fitUiToWindow(expandWindow = false) {
  if (!expandWindow || !window.pywebview?.api?.fit_window_to_content) {
    applyViewportFit();
    return;
  }
  if (_isFittingWindow) return;
  _isFittingWindow = true;
  try {
    const searchModal = document.getElementById('searchLyricsModal');
    const availW = Math.max(760, (window.screen?.availWidth || 1920) - 24);
    const probeW = Math.min(1140, availW - 20);

    const m = measureActiveLayoutNeeds(probeW);
    let targetInnerW = m.idealW;
    let targetInnerH = m.idealH;

    if (searchModal && searchModal.classList.contains('active')) {
      targetInnerW = Math.max(targetInnerW, 1140);
      targetInnerH = Math.max(targetInnerH, 860);
    }

    const chromeW = Math.max(0, Math.min(48, (window.outerWidth || 0) - (window.innerWidth || 0))) || 16;
    const chromeH = Math.max(0, Math.min(80, (window.outerHeight || 0) - (window.innerHeight || 0))) || 39;

    await window.pywebview.api.fit_window_to_content(
      Math.ceil(targetInnerW + chromeW),
      Math.ceil(targetInnerH + chromeH)
    );
  } catch (e) {
    console.warn('Auto window fit error:', e);
  } finally {
    _isFittingWindow = false;
    requestAnimationFrame(() => applyViewportFit());
  }
}

let _fitWinTimer = null;
function scheduleWindowFit(expandWindow = false) {
  if (_fitWinTimer) clearTimeout(_fitWinTimer);
  _fitWinTimer = setTimeout(() => {
    fitUiToWindow(expandWindow);
  }, expandWindow ? 35 : 40);
}

let _resizeFitTimer = null;
window.addEventListener('resize', () => {
  if (_resizeFitTimer) clearTimeout(_resizeFitTimer);
  _resizeFitTimer = setTimeout(() => applyViewportFit(), 40);
});
window.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => applyViewportFit(), 50);
});