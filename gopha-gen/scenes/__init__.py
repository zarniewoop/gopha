from .procedural_discs import ProceduralDiscsScene
from .slate import SlateScene
from .procedural_discs_v2 import ProceduralDiscsSceneV2

SCENE_REGISTRY = {
    "procedural_discs": ProceduralDiscsScene,
    "slate": SlateScene,
    "procedural_discs_v2": ProceduralDiscsSceneV2
}


def create_scene(name, width, height, config):
    try:
        scene_cls = SCENE_REGISTRY[name]
    except KeyError as e:
        valid = ", ".join(sorted(SCENE_REGISTRY.keys()))
        raise ValueError(f"Unknown scene '{name}'. Valid scenes: {valid}") from e
    return scene_cls(width, height, config)