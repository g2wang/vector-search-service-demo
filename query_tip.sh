#!/bin/bash

curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{"text": "How to deal with snow"}'

