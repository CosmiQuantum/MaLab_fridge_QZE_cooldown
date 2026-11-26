"""Hardware-free, best-effort status publication for timed SSF acquisitions.

Raw HDF5/plot persistence remains the acquisition's responsibility. Status
publication must never turn a Windows reader lock into lost acquisition time.
"""
import json
import os
import tempfile
import time


def publish_status(path, state, attempts=3):
    """Atomically publish JSON; retry transient I/O and report failure, not raise.

    Unique temporary files avoid sharing a fixed .tmp with readers. A failed
    update retains the previous complete JSON. Only supplemental I/O errors
    are caught; serialization/programming errors remain visible.
    """
    payload = json.dumps(state, indent=2)
    temporary = None
    try:
        for attempt in range(attempts):
            try:
                if temporary is None:
                    descriptor, temporary = tempfile.mkstemp(
                        prefix=os.path.basename(path) + '.', suffix='.tmp',
                        dir=os.path.dirname(os.path.abspath(path)))
                    with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
                        output.write(payload)
                os.replace(temporary, path)
                return True
            except OSError as error:
                if attempt + 1 == attempts:
                    print('WARNING: supplemental SSF status unavailable at %s: %s'
                          % (path, error), flush=True)
                    return False
                time.sleep(0.05 * (attempt + 1))
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def publish_ssf_status(path, state):
    """Write the optional local watchdog mirror before the network sidecar."""
    local_path = os.environ.get('PUCQ4_SSF_LOCAL_STATUS')
    if local_path:
        publish_status(local_path, state)
    return publish_status(path, state)
