#!/bin/bash

curl -X POST http://localhost:3000/tip \
  -H "Content-Type: application/json" \
  -d '{"text": "Build Scripts\nSome packages need to compile third-party non-Rust code, for example C libraries. Other packages need to link to C libraries which can either be located on the system or possibly need to be built from source. Others still need facilities for functionality such as code generation before building (think parser generators).\n\nCargo does not aim to replace other tools that are well-optimized for these tasks, but it does integrate with them with custom build scripts. Placing a file named build.rs in the root of a package will cause Cargo to compile that script and execute it just before building the package."}'
