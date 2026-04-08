#!/usr/bin/env python3
"""Test decode_modes string parsing."""

# Test the string parsing logic
decode_modes_str = "gaussian,mesh"
decode_modes_list = ["gaussian", "mesh"]

# Simulate the parsing
if isinstance(decode_modes_str, str):
    parsed = [m.strip() for m in decode_modes_str.split(",")]
    print(f"String input: {decode_modes_str!r}")
    print(f"Parsed: {parsed}")
    print(f"Match: {parsed == decode_modes_list}")
else:
    print(f"Already a list: {decode_modes_str}")

# Test with list input
if isinstance(decode_modes_list, str):
    parsed = [m.strip() for m in decode_modes_list.split(",")]
    print(f"List input parsed: {parsed}")
else:
    print(f"List input: {decode_modes_list}")
