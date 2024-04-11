#!/bin/bash -e

# TWINE_USERNAME - (Requried) Username for the publisher's account on PyPI
# TWINE_PASSWORD - (Required, Secret) API Token for the publisher's account on PyPI

cat <<EOF >> ~/.pypirc
[distutils]
index-servers=pypi

[pypi]
username=$TWINE_USERNAME
password=$TWINE_PASSWORD
EOF

# Deploy to pip
python -m build
twine upload dist/* --verbose
