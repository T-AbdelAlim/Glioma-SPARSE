from PIL import Image
import random


class Patch(object):
    """
    Patch shuffle augmentation.

    Splits image into grid, optionally applies per-tile transform,
    shuffles tiles, and reconstructs image.

    Input: PIL Image
    Output: PIL Image
    """

    def __init__(
        self,
        grid_size=8,
        tile_transform=None,
        shuffle=True,
        seed=None
    ):
        self.grid_size = grid_size
        self.tile_transform = tile_transform
        self.shuffle = shuffle
        self.seed = seed

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("Input must be PIL Image")

        w, h = img.size

        # enforce exact divisibility
        if w % self.grid_size != 0 or h % self.grid_size != 0:
            raise ValueError(
                "Image size ({}, {}) not divisible by grid_size {}".format(
                    w, h, self.grid_size
                )
            )

        tile_w = w // self.grid_size
        tile_h = h // self.grid_size

        tiles = []

        # --------------------------------------------------
        # Extract tiles
        # --------------------------------------------------
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                left = j * tile_w
                upper = i * tile_h
                right = left + tile_w
                lower = upper + tile_h

                tile = img.crop((left, upper, right, lower))

                if self.tile_transform is not None:
                    tile = self.tile_transform(tile)

                    if not isinstance(tile, Image.Image):
                        raise TypeError(
                            "tile_transform must return PIL Image"
                        )

                tiles.append(tile)

        # --------------------------------------------------
        # Shuffle tiles
        # --------------------------------------------------
        if self.shuffle:
            if self.seed is not None:
                rng = random.Random(self.seed)
                rng.shuffle(tiles)
            else:
                random.shuffle(tiles)

        # --------------------------------------------------
        # Reconstruct image
        # --------------------------------------------------
        new_img = Image.new("RGB", (w, h))

        idx = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                left = j * tile_w
                upper = i * tile_h

                new_img.paste(tiles[idx], (left, upper))
                idx += 1

        return new_img