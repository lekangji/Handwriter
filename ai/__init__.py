"""Handwriting models and training. Keep small recurrent matvecs single-threaded."""
import os

# Spawning a BLAS thread team at every pen point costs more than the matvec.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
