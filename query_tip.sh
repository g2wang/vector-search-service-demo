#!/bin/bash

curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{"text": "民营企业座谈会"}'

