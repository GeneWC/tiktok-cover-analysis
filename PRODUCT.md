# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Instrumental cover musicians (TikTok and similar short-video apps) who want a plain read on how a clip is filmed and recorded before they post. [Inferred from README and existing product copy.]

## Product Purpose

Zukover analyzes how an instrumental cover is presented — filming and sound — and returns a coaching-style report. Success is a clear, honest read the musician can act on, not a virality promise.

## Positioning

Scores are exploratory similarity to comparable covers, plus within-batch comparison of a creator’s own videos. Neighboring “will this go viral?” tools cannot truthfully claim that; this product does not either.

## Operating Context

Upload a short cover (mp4 / mov / m4v, about 1–120s). Wait while the job runs. Read presentation scores, feature details, and recommendations. Optional second path: upload 5+ of your own videos (with optional view counts) for within-batch diagnostics.

## Capabilities and Constraints

- Single-video analysis and channel-batch diagnostics
- Local filesystem uploads and SQLite job records
- Frontend: Vite + React + TypeScript + Tailwind
- Backend: FastAPI; analysis needs trained model artifacts
- Hashtags and instrument fields exist but are not used by the current model
- No guarantee of views, engagement, or ranking

## Brand Commitments

- Product name: Zukover (formerly CoverSignal)
- Credit: “made by @suibianmusic · formerly CoverSignal” in the footer
- Voice: short, plain descriptions. No marketing or AI-sounding copy
- Binding visual constraint (user): calm Fire Nation / Zuko atmosphere from Avatar: The Last Airbender — intense, still, inviting. Reference stills provided by the user (palace eaves at dusk, dancing-dragon gold light, maroon quote card, Zuko before the flame emblem)

## Evidence on Hand

- Working upload → processing → report flows for single video and channel batch
- User-supplied mood stills (not product photography)
- No testimonials, case studies, or performance benchmarks. Do not invent them.

## Product Principles

- Tell the musician what the video is doing, in ordinary words
- Never imply a forecast of views
- Keep optional fields optional and say so
- Preserve the two jobs: one video, or a batch of your own
- Atmosphere can be strong; the task stays obvious
