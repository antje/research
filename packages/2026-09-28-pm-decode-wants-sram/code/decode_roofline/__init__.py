"""decode_roofline: what one LLM decode step costs, and which memory it wants.

Modules: model (shapes from a config.json), chips (spec-sheet numbers with sources),
roofline (bytes, FLOPs, intensity, ridge, fit), plot (the package diagram).
Run `python -m decode_roofline` to print the tables and write results/.
"""
