"""Locate the transport bundled alongside the frozen FLIR runtime."""
import os
import sys
from pathlib import Path

transport = Path(sys._MEIPASS) / 'flir' / 'Spinnaker_GenTL.cti'
if transport.is_file():
    os.environ['SPINNAKER_GENTL64_CTI'] = str(transport)
