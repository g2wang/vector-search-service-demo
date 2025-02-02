#!/bin/bash

curl -X POST http://localhost:3000/tip \
  -H "Content-Type: application/json" \
  -d '{"text": "Remember to drink water regularly"}'
