#!/bin/bash

curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{"text": "Please explain the anyhow Rust crate"}'

