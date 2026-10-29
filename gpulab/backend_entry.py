"""Load only GPU Lab's pure Python code into another backend interpreter.

Never put the application's entire site-packages on a CUDA/ROCm interpreter's
PYTHONPATH: that could import NumPy extensions compiled for another Python ABI.
"""
import importlib.util
import runpy
import sys
from pathlib import Path

package=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gpulab',package/'__init__.py',submodule_search_locations=[str(package)])
module=importlib.util.module_from_spec(spec)
sys.modules['gpulab']=module
spec.loader.exec_module(module)
sys.argv=sys.argv[1:]
runpy.run_module('gpulab.'+sys.argv[0],run_name='__main__')
