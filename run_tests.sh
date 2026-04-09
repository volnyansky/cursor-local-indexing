#!/bin/bash
set -e

source .venv/bin/activate
pytest tests/ -v
