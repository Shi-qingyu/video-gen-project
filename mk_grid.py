from torchvision.io import read_image, write_png
from torchvision.utils import make_grid

from pathlib import Path


root = Path("attention_map/A_dog_is_playing_with_a_puppy/attn_maps")

object_ids = set([image.stem.split("_")[0] for image in root.iterdir()])
for object_id in object_ids:
    images = [
        image.as_posix() for image in root.iterdir() if image.name.startswith(object_id)
    ]
    bank = []
    func = lambda x: int(x.split("_")[-1].split(".")[0])
    iter = sorted(images, key=func)
    for image in iter:
        bank.append(read_image(image))
    image = make_grid(bank, nrow=13)
    write_png(image, f"test{object_id}.png")