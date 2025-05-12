from src.algorithms import BaseAlgorithm

def get_layer_name(layer: BaseAlgorithm) -> str:
    layer_name = layer.__class__.__name__

    return layer_name
