#!/bin/bash
# Add the DALE project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH has been updated to include $(pwd)"
echo "You can now run your test script"