# BIDS_minimize

Schema-driven BIDS filename minimizer.

This tool scans a BIDS tree recursively, loads the latest schema from `https://jsr.io/@bids/schema` (with a GitHub fallback when JSR is unavailable), and removes non-mandatory filename entities while preserving unique filenames.

## Usage

```bash
python bids_minimize.py /path/to/bids_dataset
python bids_minimize.py /path/to/bids_dataset --dry-run
```
