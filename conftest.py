import sys
from pathlib import Path

# Add api directory to path so "from api.app..." imports work
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "api"))
