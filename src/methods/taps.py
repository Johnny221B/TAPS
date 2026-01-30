from src.methods.adapters.llada_generate import generate as llada_generate
from src.methods.adapters.trado_generate import generate as trado_generate


def generate_ids(backbone: str, **kwargs):
    backbone = backbone.lower()
    if backbone == "llada":
        return llada_generate(**kwargs)
    elif backbone == "trado":
        return trado_generate(**kwargs)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
