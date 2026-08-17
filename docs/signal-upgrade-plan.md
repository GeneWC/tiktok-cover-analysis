# Signal upgrade plan (candidates, not serving)

This pass adds extractors and tests only. It does **not** re-extract the
~1,793 files in `downloads/`, does **not** change
`backend/models/feature_schema.json` (still 70 columns, version 1), and does
**not** retrain committed `.pkl` files. Extra keys are dropped at inference
the same way camera extras already were.

Skills used as checklists (not Gemini video-watchers):

- `nlp-natural-language-processing` — tokenize, redact `@handles` / URLs /
  creator names; lexicon flags instead of embeddings.
- `music-analyzer` — HPSS, chroma *stability*, spectral contrast, speech-band
  energy. **Not** copied: genre, named key, Demucs stems, Whisper lyrics.

## Candidate columns

### Text (EAST geometry + optional EasyOCR)

Role (no reader required): `text_titlecard_ratio`, `text_caption_ratio`,
`text_corner_watermark_ratio`.

Semantics (null + `text_read_failed=1` if EasyOCR is not installed):
`text_char_count`, `text_unique_token_count`, `text_first_3s_char_count`,
`text_has_song_or_piece_cue`, `text_has_cta`, `text_has_question_hook`,
`text_has_social_handle`, `text_script_latin_ratio`, `text_has_cjk`,
`text_all_caps_ratio`.

Interaction: `opening_text_plus_face`.

EasyOCR is optional and **not** in `requirements.txt` (it pulls torch). Install
locally when you are ready to re-extract: `pip install easyocr`.

### Speech / MIR (same 22 kHz decode as existing audio)

`speech_ratio`, `speech_ratio_first_3s`, `music_after_speech_gap`,
`audio_harmonic_ratio`, `audio_percussive_ratio`, `audio_chroma_stability`,
`audio_spectral_contrast_mean`, `audio_vocal_energy`.

Audio-structure columns already in the training CSV remain the highest-evidence
unused family (val ROC-AUC 0.608 with framing+visual). Promoting them still
waits on an explicit retrain.

## Val protocol (when re-extract is allowed)

1. `python scripts/extract_features.py` (resumable into `data/video_features.csv`).
2. Rebuild `data/training_dataset.csv`.
3. Val-only ablation; refuse `--split test`.
4. Ship a group only if **both** val ROC-AUC and within-creator pairwise beat
   the current RF control (0.586 / 0.636). Framing+visual+structure is the
   first comparison, then text/speech/MIR.
5. If a group helps: bump `schema_version`, retrain with
   `fit_dataset=train+val`, leave frozen test untouched.

UI copy for new keys is observed-signal labels in
`frontend/src/lib/featureCatalog.ts`, not causal view claims.
