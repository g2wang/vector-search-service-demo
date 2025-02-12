#!/bin/bash

curl -X POST http://localhost:3000/tip \
  -H "Content-Type: application/json" \
  -d '{"text": "To embed a piece of test, first choose a embedding model, then break the text into tokens, then run the model to embed the tokens"}'
