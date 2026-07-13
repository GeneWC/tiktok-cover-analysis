// Plain-language explanations for jargon that appears in the report, aimed at
// users without video/audio-editing background. Each entry lists aliases so we
// can look a definition up by feature key (e.g. "audio_silence_ratio") or by a
// human label / signal prefix (e.g. "Dead air").

interface GlossaryEntry {
  aliases: string[];
  definition: string;
}

const ENTRIES: GlossaryEntry[] = [
  // --- Audio ---
  {
    aliases: ["dead air", "silence ratio", "audio_silence_ratio"],
    definition:
      "Stretches of the video with little or no sound. Long silent gaps can make a clip feel empty and cause viewers to scroll away.",
  },
  {
    aliases: ["audio clipping", "clipping", "audio_clipping_ratio"],
    definition:
      "Distortion that happens when audio is recorded too loud — the loudest parts get 'cut off,' creating a harsh, crackly sound.",
  },
  {
    aliases: [
      "audio dynamics",
      "audio dynamic range",
      "dynamic range",
      "audio_dynamic_range",
    ],
    definition:
      "The gap between the quietest and loudest parts of the audio. Healthy range keeps music lively; over-compressing it makes everything sound flat and equally loud.",
  },
  {
    aliases: ["loudness (rms mean)", "loudness", "rms", "audio_rms_mean", "audio_rms_std"],
    definition:
      "The overall average volume of the audio (RMS = a way of measuring loudness that reflects how loud it actually sounds to people).",
  },
  {
    aliases: ["peak level", "audio_peak_level"],
    definition:
      "The loudest single moment in the audio. If it's too high the sound can distort (see audio clipping).",
  },
  {
    aliases: ["spectral centroid", "audio_spectral_centroid_mean"],
    definition:
      "A measure of how 'bright' or treble-heavy the sound is. Higher values mean more high-frequency content (crisp/tinny); lower means darker/bassier.",
  },
  {
    aliases: ["spectral bandwidth", "audio_spectral_bandwidth_mean"],
    definition:
      "How spread out the sound is across low and high frequencies — a rough sense of how 'full' or 'thin' the audio is.",
  },
  {
    aliases: ["spectral rolloff", "audio_spectral_rolloff_mean"],
    definition:
      "The frequency below which most of the sound's energy sits. It's another way to gauge how bright or bass-heavy the audio is.",
  },
  {
    aliases: ["zero crossing rate", "audio_zero_crossing_rate_mean"],
    definition:
      "How often the audio waveform flips between positive and negative. Higher values usually mean noisier or more percussive/hissy sounds.",
  },
  {
    aliases: [
      "onset strength",
      "audio_onset_strength_mean",
      "audio_onset_strength_std",
    ],
    definition:
      "How strongly new notes or beats 'kick in.' Higher onset strength means a more rhythmic, punchy performance.",
  },
  {
    aliases: [
      "audio energy",
      "audio_energy_full",
      "audio_energy_first_1s",
      "audio_energy_first_3s",
      "audio_energy_first_6s",
      "audio_energy_ratio_first_3s_to_full",
    ],
    definition:
      "How much sound is happening overall. The 'first 3s' versions measure whether the opening grabs attention with sound quickly.",
  },

  // --- Visual quality ---
  {
    aliases: ["footage sharpness", "sharpness", "sharpness (overall)", "sharpness_full", "sharpness_first_3s"],
    definition:
      "How crisp and in-focus the picture is. Blurry or soft footage looks lower quality on screen.",
  },
  {
    aliases: ["visual contrast", "contrast", "contrast (overall)", "contrast_full", "contrast_first_3s"],
    definition:
      "The difference between the light and dark areas of the image. Good contrast helps the performer stand out from the background.",
  },
  {
    aliases: [
      "color vibrancy",
      "colorfulness",
      "colorfulness (overall)",
      "colorfulness_full",
      "colorfulness_first_3s",
    ],
    definition:
      "How rich and saturated the colors are. Vibrant frames tend to catch the eye more than dull, washed-out ones.",
  },
  {
    aliases: ["blur", "blur (overall)", "blur_full", "blur_first_3s"],
    definition:
      "A measure of how out-of-focus or smeared the footage is. Lower blur means a sharper, cleaner image.",
  },
  {
    aliases: [
      "brightness",
      "brightness (overall)",
      "brightness_mean_full",
      "brightness_mean_first_1s",
      "brightness_mean_first_3s",
      "brightness_mean_first_6s",
    ],
    definition:
      "How light or dark the video is overall. Too dark hides detail; too bright washes it out — a balanced middle usually looks best.",
  },
  {
    aliases: ["bitrate"],
    definition:
      "How much data is used per second of video. Higher bitrate generally means better image quality (fewer compression artifacts).",
  },
  {
    aliases: ["resolution (pixels)", "resolution", "resolution_area"],
    definition:
      "The number of pixels in the frame (width × height). More pixels means a sharper, higher-definition picture.",
  },
  {
    aliases: ["aspect ratio", "aspect_ratio"],
    definition:
      "The shape of the frame (width compared to height). TikTok favors tall, vertical video (about 9:16).",
  },
  {
    aliases: ["frame rate", "fps"],
    definition:
      "How many still images are shown per second (frames per second). Higher frame rates make motion look smoother.",
  },

  // --- Framing & subject ---
  {
    aliases: [
      "subject centering",
      "subject centering score",
      "subject_centering_score",
    ],
    definition:
      "How close the performer is to the middle of the frame. Well-centered subjects are easier to focus on.",
  },
  {
    aliases: ["subject size in frame", "subject size ratio", "subject_size_ratio"],
    definition:
      "How much of the screen the performer fills. Filling more of the frame (getting closer or cropping tighter) usually reads better on a phone.",
  },
  {
    aliases: ["face size in frame", "face size ratio", "face_size_ratio"],
    definition:
      "How large the performer's face appears relative to the frame.",
  },
  {
    aliases: [
      "performer visibility",
      "performer visible ratio",
      "person_visible_ratio",
      "person_visible_ratio_first_3s",
    ],
    definition:
      "The share of the video where a person is actually visible on screen.",
  },
  {
    aliases: [
      "face visibility",
      "face visible ratio",
      "face_visible_ratio",
      "face_visible_ratio_first_3s",
    ],
    definition:
      "The share of the video where the performer's face is visible. Seeing a face helps viewers connect.",
  },
  {
    aliases: [
      "hands visible ratio",
      "hand_visible_ratio",
      "hand_visible_ratio_first_3s",
    ],
    definition:
      "The share of the video where hands are visible — useful for instrument covers where technique is part of the appeal.",
  },
  {
    aliases: ["upper body visible ratio", "upper_body_visible_ratio"],
    definition:
      "The share of the video where the performer's upper body is in frame.",
  },

  // --- Motion & stability ---
  {
    aliases: [
      "motion energy",
      "motion energy (overall)",
      "motion_energy_full",
      "motion_energy_first_1s",
      "motion_energy_first_3s",
      "motion_energy_first_6s",
      "motion_energy_ratio_first_3s_to_full",
    ],
    definition:
      "How much movement is happening in the video. Some motion keeps things lively; too much can feel chaotic.",
  },
  {
    aliases: ["motion steadiness", "motion consistency", "motion_consistency"],
    definition:
      "How even and controlled the movement is over time, rather than jerky or unpredictable.",
  },
  {
    aliases: ["camera stability", "camera stability score", "camera_stability_score"],
    definition:
      "How steady the camera is. Shaky footage (no tripod/gimbal) looks less polished and can be distracting.",
  },
  {
    aliases: [
      "hand motion",
      "hand_motion_energy_full",
      "hand_motion_energy_first_3s",
      "hand_motion_consistency",
    ],
    definition:
      "Movement specifically from the performer's hands — relevant for showing playing technique.",
  },

  // --- On-screen text ---
  {
    aliases: [
      "on-screen text present",
      "on-screen text",
      "text present",
      "text_present_anywhere",
      "text_present_first_1s",
      "text_present_first_3s",
    ],
    definition:
      "Whether captions or text overlays appear on the video. Optional for instrumental covers, but text can add context or hooks.",
  },
  {
    aliases: [
      "text area ratio",
      "text area ratio (overall)",
      "text_area_ratio_full",
      "text_area_ratio_first_3s",
      "average_text_area_ratio_when_present",
    ],
    definition:
      "How much of the screen is covered by on-screen text.",
  },
  {
    aliases: ["first text appears at", "first_text_timestamp"],
    definition:
      "How many seconds in before any on-screen text shows up.",
  },
  {
    aliases: ["ocr", "text detection failed", "ocr_failed"],
    definition:
      "OCR (optical character recognition) is the automatic reading of text in the video. 'Failed' just means the detector couldn't run — not that anything is wrong with your video.",
  },

  // --- Scores & predictions ---
  {
    aliases: ["overall presentation", "presentation score"],
    definition:
      "A 0-100 rank of how polished the video looks and sounds compared to similar covers — averaging the visual, audio, motion, and framing scores.",
  },
  {
    aliases: ["visual quality"],
    definition:
      "A 0-100 rank of picture quality (sharpness, contrast, color) versus comparable covers.",
  },
  {
    aliases: ["audio quality"],
    definition:
      "A 0-100 rank of sound quality (dynamics, clipping, silence) versus comparable covers. Shows 'Not available' when there's no audio track.",
  },
  {
    aliases: ["motion"],
    definition:
      "A 0-100 rank of how steady and controlled the movement and camera are versus comparable covers.",
  },
  {
    aliases: ["framing"],
    definition:
      "A 0-100 rank of how well the performer is positioned and sized in the frame versus comparable covers.",
  },
  {
    aliases: ["top-quartile", "top quartile", "top-quartile probability"],
    definition:
      "The estimated chance this video lands in the top 25% of comparable instrumental covers. It's a probability, not a guarantee.",
  },
  {
    aliases: ["view performance", "view performance tier"],
    definition:
      "A tier (Low to High) estimating how this video's views might compare to the creator's own typical results. Marked low-confidence because it doesn't generalize well.",
  },
  {
    aliases: ["engagement", "engagement tier"],
    definition:
      "A tier (Low to High) estimating likes/comments relative to views, compared to similar covers.",
  },
  {
    aliases: ["shareability", "shareability tier"],
    definition:
      "A tier (Low to High) estimating how often the video might be shared relative to views. Marked low-confidence.",
  },
  {
    aliases: ["percentile", "percentile rank"],
    definition:
      "Where this video ranks against the training set. A 70 means it scores higher than about 70% of comparable covers.",
  },
];

const LOOKUP: Map<string, string> = (() => {
  const map = new Map<string, string>();
  for (const entry of ENTRIES) {
    for (const alias of entry.aliases) {
      map.set(normalize(alias), entry.definition);
    }
  }
  return map;
})();

function normalize(text: string): string {
  return text.trim().toLowerCase().replace(/\s+/g, " ");
}

/** Look up a definition by feature key or human label. Returns null if none. */
export function getDefinition(term: string): string | null {
  if (!term) return null;
  return LOOKUP.get(normalize(term)) ?? null;
}

/**
 * Signal strings look like "Dead air: weaker than most comparable covers."
 * Return the term (before the colon) and its definition, if known.
 */
export function getSignalTerm(
  signal: string
): { term: string; rest: string; definition: string } | null {
  const colon = signal.indexOf(":");
  if (colon === -1) return null;
  const term = signal.slice(0, colon);
  const definition = getDefinition(term);
  if (!definition) return null;
  return { term, rest: signal.slice(colon), definition };
}

// Jargon phrases that may appear *inside* a free-text recommendation. Ordered
// longest-first so the most specific phrase wins.
const INLINE_PHRASES = [
  "dynamic range",
  "dead air",
  "clipping",
]
  .map((phrase) => ({ phrase, definition: getDefinition(phrase)! }))
  .filter((p) => p.definition)
  .sort((a, b) => b.phrase.length - a.phrase.length);

/** Find the first known jargon phrase inside a recommendation sentence. */
export function getInlineDefinition(
  text: string
): { phrase: string; definition: string } | null {
  const lower = text.toLowerCase();
  for (const { phrase, definition } of INLINE_PHRASES) {
    if (lower.includes(phrase)) return { phrase, definition };
  }
  return null;
}
