#!/bin/sh

set -e
cargo clean -q
rm -f submission.zip
zip -q -r submission.zip * 
echo "Generated file submission.zip. Upload it to Gradescope."