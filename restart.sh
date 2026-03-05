#!/bin/bash
# Setup script to configure Poetry for local virtual environment
docker compose down
docker compose build
docker compose up -d