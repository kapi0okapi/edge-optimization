"""Keep ONNX Runtime inside the CPUs Slurm gave the job.

ONNX Runtime sizes its thread pool by the cores of the whole node, not by the job's allocation.
On a shared 64-core node with 8 allocated CPUs that means 64 threads fighting for 8 cores (and
failed affinity calls). After ``limit_onnxruntime_threads()``, every session, including the ones
Ultralytics and the quantisation tool create, uses the allocated number of threads.
"""

from __future__ import annotations

import os


def limit_onnxruntime_threads() -> None:
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if not cpus:
        return
    import onnxruntime as ort

    if getattr(ort.InferenceSession, "_threads_limited", False):
        return
    base = ort.InferenceSession

    class InferenceSession(base):
        _threads_limited = True

        def __init__(self, path_or_bytes, sess_options=None, *args, **kwargs):
            options = sess_options or ort.SessionOptions()
            if options.intra_op_num_threads == 0:  # 0 = ONNX Runtime's default (all cores of the node)
                options.intra_op_num_threads = int(cpus)
                options.inter_op_num_threads = 1
            super().__init__(path_or_bytes, options, *args, **kwargs)

    ort.InferenceSession = InferenceSession
