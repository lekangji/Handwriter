---
name: AI Handwriter
description: A warm, simple workspace for handwritten documents.
colors:
  ink: "#342b25"
  paper: "#f6f0e7"
  card: "#fffdf8"
  accent: "#9b4b32"
  muted: "#746357"
  line: "#dfd3c4"
  warn-bg: "#f6e5d6"
  warn-ink: "#7a3824"
typography:
  display:
    fontFamily: "Georgia, 'Times New Roman', serif"
    fontSize: "clamp(2rem, 3.2vw, 2.9rem)"
    fontWeight: 400
    lineHeight: 1.15
    letterSpacing: "-0.025em"
  body:
    fontFamily: "'Segoe UI', sans-serif"
    lineHeight: 1.6
rounded:
  control: "6px"
  canvas: "12px"
spacing:
  field-gap: "20px"
  desktop-gutter: "40px"
  desktop-columns: "54px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.card}"
    rounded: "{rounded.control}"
    padding: "13px 22px"
---

# Design System: AI Handwriter

## Overview

Warm and simple, as requested. The interface pairs an open writing workspace with a paper preview. Form sections use thin separators rather than enclosing every group in a card.

## Colors

Cream paper and warm ink set the quiet background. Terracotta identifies actions, active navigation, ranges, and focus. Preview pages remain white to represent the physical export.

## Typography

Georgia gives headings and the document editor a familiar printed voice. Segoe UI keeps controls and instructions clear. Code and measurement values retain conventional formatting.

## Layout

The desktop workspace has two columns in a 1320px container. Below 900px, the preview follows the editor. At 580px, gutters reduce to 20px and navigation wraps. Capture uses a responsive canvas with coordinate scaling.

## Elevation & Depth

Most surfaces stay flat. Only paper previews use a soft offset shadow: `0 12px 36px #49351c12`.

## Shapes

Controls use modest corners. Writing canvases use 12px corners. Separators and borders use warm neutral tones.

## Components

Primary actions use terracotta with cream text. Hover darkens the fill. Visible focus uses a terracotta outline. Disabled actions remain legible. Progress uses native accessible bars and live status text. Settings use native details controls. Error messages include recovery guidance. Reduced-motion preferences disable transitions.

## Do's and Don'ts

Keep the handwriting preview central. Maintain the warm palette across capture, training, and writing. Keep optional model and calibration settings behind clear disclosure controls. Do not imply perfect transcription or guaranteed matching from a small dataset.
