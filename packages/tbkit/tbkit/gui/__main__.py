import sys

from . import main

# The guard matters: calculations run in "spawn" child processes, which import this module
# again under another name; without it each child would open a window of its own.
if __name__ == "__main__":
    sys.exit(main())
