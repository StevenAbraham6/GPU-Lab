import signal
import sys
from .runner import work
from .store import Store

def interrupted(signum,frame):
    raise InterruptedError(f"Worker received signal {signum}")

if __name__=="__main__":
    signal.signal(signal.SIGTERM,interrupted)
    work(Store(sys.argv[1]),sys.argv[2])
