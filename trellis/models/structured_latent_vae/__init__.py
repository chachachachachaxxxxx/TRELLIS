import importlib

__attributes = {
    "SLatEncoder": "encoder",
    "ElasticSLatEncoder": "encoder",
    "SLatGaussianDecoder": "decoder_gs",
    "ElasticSLatGaussianDecoder": "decoder_gs",
    "SLatRadianceFieldDecoder": "decoder_rf",
    "ElasticSLatRadianceFieldDecoder": "decoder_rf",
    "SLatMeshDecoder": "decoder_mesh",
    "ElasticSLatMeshDecoder": "decoder_mesh",
}

__all__ = list(__attributes.keys())


def __getattr__(name: str):
    if name in __attributes:
        module = importlib.import_module(f".{__attributes[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
