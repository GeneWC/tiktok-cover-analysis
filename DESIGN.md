---
name: Zukover
description: Dusk palace courtyard for reading how a cover is presented
colors:
  dusk: "oklch(0.16 0.038 32)"
  lacquer: "oklch(0.20 0.046 30)"
  panel: "oklch(0.245 0.05 28)"
  ridge: "oklch(0.30 0.052 28)"
  brick: "oklch(0.42 0.09 28)"
  wine: "oklch(0.36 0.08 25)"
  ember: "oklch(0.64 0.145 48)"
  gold: "oklch(0.78 0.10 78)"
  ivory: "oklch(0.93 0.028 82)"
  dust: "oklch(0.78 0.045 55)"
typography:
  display:
    fontFamily: "Kaisei Tokumin, Songti SC, PMingLiU, serif"
    fontSize: "2.25rem"
    fontWeight: 500
    lineHeight: 1.2
    letterSpacing: "-0.02em"
  body:
    fontFamily: "Zen Kaku Gothic New, Hiragino Sans, Yu Gothic, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.65
    letterSpacing: "normal"
rounded:
  sm: "2px"
spacing:
  sm: "8px"
  md: "16px"
  lg: "32px"
components:
  button-primary:
    backgroundColor: "{colors.gold}"
    textColor: "{colors.dusk}"
    rounded: "{rounded.sm}"
    padding: "12px 16px"
  button-primary-hover:
    backgroundColor: "oklch(0.84 0.10 80)"
    textColor: "{colors.dusk}"
  button-ghost:
    backgroundColor: "oklch(0.20 0.046 30 / 0.5)"
    textColor: "{colors.gold}"
    rounded: "{rounded.sm}"
    padding: "8px 16px"
  field:
    backgroundColor: "oklch(0.16 0.038 32 / 0.7)"
    textColor: "{colors.ivory}"
    rounded: "{rounded.sm}"
    padding: "8px 12px"
  panel:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ivory}"
    rounded: "{rounded.sm}"
    padding: "24px"
---

## Overview

Zukover is a dusk courtyard, not a light SaaS tool. The page sits in mahogany shadow with worn gold leaf. Atmosphere comes from a palace-eaves still at sunset; the work itself stays plain: upload, wait, read.

## Colors

Dusk and lacquer own the surface. Ivory is the reading color; dust is secondary text, tinted warm so it is never gray. Gold is rare — links, the mark, the primary action. Ember is for in-progress and mid scores. Brick is for failure and low scores.

## Typography

Headings use Kaisei Tokumin, a carved plaque face. Body and controls use Zen Kaku Gothic New. Five clear sizes: display ~2.25–2.5rem, section ~1.5rem, body 1rem, small 0.875rem, micro 0.75rem. No Inter, no system UI stack as the voice of the page.

## Layout

Centered column, max 64rem. Header is a thin lacquer bar with the flame mark and two links. The first viewport is title, one sentence, one panel. Reports mix open sections with a few lacquered panels — not a stack of identical cards. Footer is two quiet lines, including the author credit.

## Elevation & Depth

Depth is overlay and ridge, not glow. Panels use a faint horizontal ridge and a soft offset shadow. No neon halo. No glass blur on chrome.

## Shapes

Corners stay nearly square (`2px`). Borders are thin gold at low opacity, sometimes read as a double edge against the dusk ground. The mark is a flame in a thin circle.

## Components

Primary button is gold on dusk. Ghost button is gold line on lacquer. Fields are recessed dusk with gold focus. Drop zones use a dashed gold edge. Errors are wine/brick panels with ivory text. Tooltips are dusk with a gold ring.

## Do's and Don'ts

- Do keep copy short and literal
- Do tint neutrals toward red-brown
- Do leave gold scarce
- Don't use indigo, purple gradients, or Inter
- Don't wrap every block in a card
- Don't promise views
