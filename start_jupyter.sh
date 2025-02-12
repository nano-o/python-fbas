#!/bin/sh
# exec needed to enable signals (like when using ctrl-C) to work
exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/app/python_fbas/constellation/notebooks
