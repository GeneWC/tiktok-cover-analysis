# Genre Feature Profiles

Reference table for mapping audio features to genres. Use these ranges as guidelines,
not strict rules. Real music often blends genres.

## Feature Ranges by Genre

| Genre | BPM Range | Spectral Centroid | Energy (RMS) | Common Keys/Modes | Onset Density | ZCR |
|-------|-----------|-------------------|--------------|-------------------|---------------|-----|
| Rock | 110-140 | 2000-4000 (mid-high) | 0.15-0.30 (high) | E, A, G, D major | 3-6 (high) | 0.08-0.12 |
| Classic Rock | 100-135 | 1800-3500 (mid) | 0.12-0.25 | E, A, G major | 2.5-5 | 0.07-0.11 |
| Pop | 100-130 | 2000-4000 (mid-high) | 0.10-0.22 (medium-high) | C, G, D, A major | 2-4 (medium) | 0.06-0.10 |
| Electronic/Dance | 120-140 | 3000-5500 (high) | 0.18-0.35 (very high) | A, C, F minor | 4-8 (very high) | 0.10-0.18 |
| Hip-Hop/Rap | 80-115 | 1500-3000 (medium) | 0.12-0.25 (medium-high) | Minor keys dominant | 2-5 (medium) | 0.05-0.09 |
| R&B/Soul | 60-100 | 1500-3000 (medium) | 0.06-0.15 (low-medium) | Minor/major 7th chords | 1-3 (low) | 0.04-0.08 |
| Jazz | 80-200 | 1500-3000 (medium) | 0.04-0.15 (low-medium) | Complex, Bb, Eb, F | 2-4 (medium) | 0.05-0.09 |
| Classical | 40-180 | 500-2500 (low-medium) | 0.02-0.20 (variable) | All keys, complex | 1-3 (low-medium) | 0.02-0.06 |
| Metal | 120-200+ | 3000-6000 (very high) | 0.25-0.40 (very high) | E, D, C minor/chromatic | 5-10 (very high) | 0.12-0.20 |
| Blues | 70-130 | 1200-2800 (low-medium) | 0.06-0.18 (medium) | E, A, G major/minor blues | 1.5-3.5 | 0.05-0.09 |
| Country | 90-140 | 1500-3000 (medium) | 0.08-0.20 (medium) | G, C, D, A major | 2-4 (medium) | 0.06-0.10 |
| Folk | 80-140 | 1200-2500 (low-medium) | 0.04-0.14 (low-medium) | G, C, D, Em major/minor | 1.5-3 (low) | 0.04-0.08 |
| Reggae | 60-90 | 1000-2500 (low-medium) | 0.08-0.18 (medium) | Minor keys, Bb, Eb | 1-3 (low) | 0.04-0.07 |
| Punk | 150-200 | 2500-5000 (high) | 0.22-0.38 (very high) | E, A, D major | 5-8 (very high) | 0.10-0.16 |
| Funk | 95-120 | 1800-3500 (mid) | 0.12-0.25 (medium-high) | E, A minor/dominant 7th | 3-6 (high) | 0.06-0.10 |
| Ambient | 60-120 | 500-2000 (low) | 0.01-0.06 (very low) | Any, often no clear key | 0.5-2 (very low) | 0.02-0.05 |
| Indie | 100-140 | 1800-3500 (mid) | 0.08-0.20 (medium) | Various, often minor | 2-5 (medium) | 0.06-0.10 |

## Subgenre Indicators

### Rock Subgenres
- **Soft Rock**: BPM 80-110, lower energy, more acoustic features
- **Hard Rock**: BPM 120-150, high energy, high spectral centroid
- **Progressive Rock**: Variable BPM, high dynamic range, complex time signatures
- **Blues Rock**: BPM 80-130, blues scale emphasis in chroma, moderate energy
- **Folk Rock**: BPM 90-130, acoustic character, lower spectral centroid
- **Alternative Rock**: Variable, moderate-high energy, mid spectral centroid
- **Country Rock**: BPM 100-140, major keys, moderate energy

### Electronic Subgenres
- **House**: BPM 120-130, four-on-the-floor beat (regular onsets)
- **Techno**: BPM 125-150, minimal melodic content, high onset density
- **Drum & Bass**: BPM 160-180, very high onset density, syncopated
- **Trance**: BPM 125-145, building energy curve, high spectral centroid
- **Ambient Electronic**: BPM 60-100, very low onset density, low energy
- **Synthwave**: BPM 80-120, mid spectral centroid, retro synth MFCCs

### Metal Subgenres
- **Heavy Metal**: BPM 120-160, high energy, high spectral centroid
- **Thrash Metal**: BPM 150-220, very high onset density, extreme energy
- **Doom Metal**: BPM 60-90, high energy but slow, low spectral centroid
- **Power Metal**: BPM 130-180, high energy, major keys more common
- **Black Metal**: BPM 140-200+, extreme ZCR, very high spectral centroid

## Mood Mapping

| Audio Signal | Typical Mood |
|-------------|-------------|
| Major key + fast BPM + high energy | Energetic, Happy, Uplifting |
| Major key + slow BPM + low energy | Peaceful, Calm, Soothing |
| Minor key + fast BPM + high energy | Intense, Aggressive, Anxious |
| Minor key + slow BPM + low energy | Sad, Melancholic, Somber |
| High dynamic range + variable energy | Dramatic, Epic, Cinematic |
| Low dynamic range + steady energy | Hypnotic, Meditative, Droning |
| Building energy curve | Anticipatory, Hopeful, Triumphant |
| Declining energy curve | Winding down, Nostalgic, Reflective |
| High onset density + high BPM | Frantic, Chaotic, Exciting |
| Low onset density + low BPM | Spacious, Atmospheric, Ethereal |

## Instrument Detection Hints

| Audio Feature | Likely Instruments |
|--------------|-------------------|
| High spectral contrast in low bands (0-200Hz) | Bass guitar, kick drum, sub-bass synth |
| High spectral contrast in mid bands (200-2000Hz) | Guitar, piano, vocals, strings |
| High spectral contrast in high bands (2000-8000Hz) | Cymbals, hi-hat, shakers, brass |
| High onset density + percussive transients | Drums, percussion |
| Smooth spectral envelope, low ZCR | Strings, pads, sustained synths |
| Narrow spectral bandwidth | Solo instrument, clean tone |
| Wide spectral bandwidth | Full band, distorted guitar, orchestra |
| Accompaniment centroid < 1500 | Acoustic instruments, bass-heavy |
| Accompaniment centroid > 3000 | Electronic, bright instruments, distortion |

## Vocal Characteristics

| Vocal RMS | Description |
|-----------|-------------|
| > 0.08 | Strong, prominent vocals |
| 0.03-0.08 | Moderate vocal presence |
| 0.01-0.03 | Soft/background vocals |
| < 0.01 | Instrumental or very faint vocals |

Vocal pitch estimation from lyrics language + genre context:
- Male vocals typically in genres: Rock, Metal, Hip-Hop, Blues, Reggae
- Female vocals common in: Pop, R&B, Country, Dance
- Mixed/harmonies common in: Folk, Indie, Gospel, Pop
