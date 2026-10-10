# AI Handwriter
<!-- impeccable:product-schema 1 -->

## Platform
web

## Product Purpose
Turn typed documents into handwriting, preview the exact strokes, and export SVG, PNG, and pen-plotter G-code. Personalize the model from captured handwriting samples.

## Operating Context
Existing local Flask application. Capture uses a mouse or stylus. Physical writing requires numeric paper and printer calibration and manual page changes.

## Capabilities and Constraints
Preserve existing export formats and printer validation. Do not touch `.backups`. Prioritize readable multi-page generation, responsive progress, and reliable capture. Existing pretrained weights and saved profiles must remain usable.

## Brand Commitments
The user requests a warm, simple theme.

## Open Decisions
The primary audience is inferred from the current product: people creating handwritten documents with a pen plotter. No deployment change is requested.
