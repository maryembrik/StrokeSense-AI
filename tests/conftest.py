import os
import tempfile

os.environ["STROKESENSE_LOG"] = os.path.join(tempfile.mkdtemp(), "predictions.jsonl")  # tests never touch the real log
