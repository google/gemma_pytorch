# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pan and scan image cropping implementation."""

from collections.abc import Sequence

import numpy as np
from PIL import Image


def pan_and_scan(
    img: Image.Image,
    *,
    min_crop_size: int = 256,
    max_num_crops: int = 4,
) -> Sequence[Image.Image]:
    return _pan_and_scan_os(
        img,
        min_crop_size=min_crop_size,
        max_num_crops=max_num_crops,
    )[0]


def pan_and_scan_os_with_crop_positions(
    img: Image.Image,
    *,
    min_crop_size: int = 256,
    max_num_crops: int = 4,
) -> tuple[Sequence[Image.Image], Sequence[tuple[int, int, int, int]]]:
    return _pan_and_scan_os(
        img,
        min_crop_size=min_crop_size,
        max_num_crops=max_num_crops,
    )


def _pan_and_scan_os(
    img: Image.Image,
    *,
    min_crop_size: int,
    max_num_crops: int,
) -> tuple[Sequence[Image.Image], Sequence[tuple[int, int, int, int]]]:
    """Pan and scan an image for open source.

    If the image is landscape, the crops are made horizontally and if the image is
    portrait, the crops are made vertically. The longer side of the image is split
    into [2 - max_num_crops] crops.

    Args:
        img: PIL Image object.
        min_crop_size: The minimum size of each crop.
        max_num_crops: The maximum desired number of crops to be generated.

    Returns:
        List of cropped PIL Image objects and a list of crop positions.
    """
    w, h = img.size

    # Square or landscape image.
    if w >= h:
        if w / h < 1.5:
            return [img], [(0, 0, h, w)]

        # Select ideal number of crops close to the image aspect ratio and such that
        # crop_size > min_crop_size.
        num_crops_w = int(np.floor(w / h + 0.5))    # Half round up rounding.
        num_crops_w = min(
            int(np.floor(w / min_crop_size)),
            num_crops_w,
        )

        # Make sure the number of crops is in range [2, max_num_crops].
        num_crops_w = max(2, num_crops_w)
        num_crops_w = min(max_num_crops, num_crops_w)
        num_crops_h = 1

    # Portrait image.
    else:
        if h / w < 1.5:
            return [img], [(0, 0, h, w)]

        num_crops_h = int(np.floor(h / w + 0.5))
        num_crops_h = min(int(np.floor(h / min_crop_size)), num_crops_h)
        num_crops_h = max(2, num_crops_h)
        num_crops_h = min(max_num_crops, num_crops_h)
        num_crops_w = 1

    crop_size_w = int(np.ceil(w / num_crops_w))
    crop_size_h = int(np.ceil(h / num_crops_h))

    # Don't apply pan and scan if crop size is too small.
    if min(crop_size_w, crop_size_h) < min_crop_size:
        return [img], [(0, 0, h, w)]

    crop_positions_w = [crop_size_w * i for i in range(num_crops_w)]
    crop_positions_h = [crop_size_h * i for i in range(num_crops_h)]

    # Generate crops.
    crops = []
    crop_positions = []
    for pos_h in crop_positions_h:
        for pos_w in crop_positions_w:
            crops.append(
                img.crop((
                    pos_w,
                    pos_h,
                    pos_w + crop_size_w,
                    pos_h + crop_size_h,
                ))
            )
            crop_positions.append(
                (pos_h, pos_w, pos_h + crop_size_h, pos_w + crop_size_w)
            )

    return crops, crop_positions
