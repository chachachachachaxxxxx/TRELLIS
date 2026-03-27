import importlib

__attributes = {
    "Strivec": ("radiance_field", "Strivec"),
    "Octree": ("octree", "DfsOctree"),
    "Gaussian": ("gaussian", "Gaussian"),
    "MeshExtractResult": ("mesh", "MeshExtractResult"),
}

__all__ = list(__attributes.keys())


def __getattr__(name: str):
    if name in __attributes:
        module_name, attr_name = __attributes[name]
        module = importlib.import_module(f".{module_name}", __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
