#!/bin/bash

curl -X POST http://localhost:3000/tip \
  -H "Content-Type: application/json" \
  -d '{"text": "Here is a tip to setup python virtual env: in your shell rc file, set an alias:\nalias penv='\''python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt'\''"}'
