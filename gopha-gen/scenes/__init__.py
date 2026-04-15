from .procedural_discs import ProceduralDiscsScene
from .slate import SlateScene

SCENE_REGISTRY = {
    "procedural_discs": ProceduralDiscsScene,
    "slate": SlateScene,
}


def create_scene(name, width, height, config):
    try:
        scene_cls = SCENE_REGISTRY[name]
    except KeyError as e:
        valid = ", ".join(sorted(SCENE_REGISTRY.keys()))
        raise ValueError(f"Unknown scene '{name}'. Valid scenes: {valid}") from e
    return scene_cls(width, height, config)