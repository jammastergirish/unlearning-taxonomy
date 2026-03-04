# Kill everything Python (nuclear option)
pkill -u $USER python
# Or more targeted — kill only unlearn.py jobs
pkill -u $USER -f unlearn.py
# Or to also catch the uv runners spawning them
pkill -u $USER -f "uv run"