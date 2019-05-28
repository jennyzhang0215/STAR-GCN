def load_c_plugin():
    import os
    import shutil
    from sys import platform

    _BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Copy the graph_sampler python lib to the destination
    if platform.lower().startswith("win"):
        _VALID_DLL_PATHS = [os.path.join(_BASE_PATH, 'GraphSampler', 'build', 'Release', 'graph_sampler.dll')]
        _TARGET_PATH = os.path.join(_BASE_PATH, 'mxgraph', '_graph_sampler.pyd')
    else:
        _VALID_DLL_PATHS = [os.path.join(_BASE_PATH, 'GraphSampler', 'build', 'libgraph_sampler.dylib'),
                            os.path.join(_BASE_PATH, 'GraphSampler', 'build', 'libgraph_sampler.so')]
        _TARGET_PATH = os.path.join(_BASE_PATH, 'mxgraph', '_graph_sampler.so')

    found = False
    for p in _VALID_DLL_PATHS:
        if os.path.exists(p):
            found = True
            print("Found python extension for graph sampling, path=%s. Copy to %s" % (p, _TARGET_PATH))
            shutil.copy(p, _TARGET_PATH)
            break
    if not found:
        raise RuntimeError(
            "Graph sampling extensions not found! Please check these paths: %s" % (str(_VALID_DLL_PATHS)))
load_c_plugin()
