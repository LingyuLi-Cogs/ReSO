# Vendored OpenRT package

This directory contains the pinned OpenRT Python package required by
`openrt32_runner.py` and the fixed Mousetrap prompt generator. Its revision and
AGPL-3.0 license are documented in `../THIRD_PARTY_NOTICES.md` and
`../LICENSE.OpenRT`.

The package is intentionally kept intact for import compatibility. Only the
methods selected by `--attacks paper` plus the fixed Mousetrap adapter enter the
project metric.
