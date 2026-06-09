"""Object detection augmentations with line/ellipse support."""

from __future__ import annotations

import math
import random
from typing import Dict, Optional, Tuple

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms
from torchvision.transforms import functional as F
from torchvision.transforms import RandomAffine, RandomResizedCrop


_EPS = 1e-8


def _get_hw(image: torch.Tensor) -> Tuple[int, int]:
    return int(image.shape[-2]), int(image.shape[-1])


def _ensure_float_image(image: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(image):
        return image
    return image.to(dtype=torch.float32).div(255.0)


def _clamp_unit_interval(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    values = values.clone()
    values = torch.where(values.abs() < _EPS, torch.zeros_like(values), values)
    values = torch.where((1.0 - values).abs() < _EPS, torch.ones_like(values), values)
    return values.clamp(0.0, 1.0)


def _apply_affine_to_points(
    points: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    if points.numel() == 0:
        return points.reshape(0, 2)
    x = points[:, 0]
    y = points[:, 1]
    x2 = matrix[0] * x + matrix[1] * y + matrix[2]
    y2 = matrix[3] * x + matrix[4] * y + matrix[5]
    return torch.stack([x2, y2], dim=1)


def _apply_affine_to_boxes(
    boxes: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    corners = torch.stack(
        [
            boxes[:, [0, 1]],
            boxes[:, [2, 1]],
            boxes[:, [2, 3]],
            boxes[:, [0, 3]],
        ],
        dim=1,
    ).reshape(-1, 2)
    warped = _apply_affine_to_points(corners, matrix).reshape(-1, 4, 2)
    x_min = warped[:, :, 0].min(dim=1).values
    y_min = warped[:, :, 1].min(dim=1).values
    x_max = warped[:, :, 0].max(dim=1).values
    y_max = warped[:, :, 1].max(dim=1).values
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)


def _apply_affine_to_lines(
    line_points: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    if line_points.numel() == 0:
        return line_points.reshape(0, 4)
    points = line_points.reshape(-1, 2)
    warped = _apply_affine_to_points(points, matrix)
    return warped.reshape(-1, 4)


def _rotate_ellipses(ellipses: torch.Tensor, angle_deg: float) -> torch.Tensor:
    if ellipses.numel() == 0:
        return ellipses.reshape(0, 6)
    theta = math.radians(angle_deg)
    cos_2 = math.cos(2.0 * theta)
    sin_2 = math.sin(2.0 * theta)
    cos2 = ellipses[:, 4]
    sin2 = ellipses[:, 5]
    ellipses[:, 4] = cos2 * cos_2 - sin2 * sin_2
    ellipses[:, 5] = sin2 * cos_2 + cos2 * sin_2
    return ellipses


def _scale_ellipses(ellipses: torch.Tensor, scale_x: float, scale_y: float) -> torch.Tensor:
    if ellipses.numel() == 0:
        return ellipses.reshape(0, 6)
    ellipses[:, 0] *= scale_x
    ellipses[:, 1] *= scale_y
    ellipses[:, 2] += math.log(scale_x)
    ellipses[:, 3] += math.log(scale_y)
    return ellipses


def _clip_line_to_bounds(
    p0: torch.Tensor,
    p1: torch.Tensor,
    width: float,
    height: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    x_min, y_min = 0.0, 0.0
    x_max, y_max = float(width), float(height)

    def _code(x: float, y: float) -> int:
        code = 0
        if x < x_min - _EPS:
            code |= 1
        elif x > x_max + _EPS:
            code |= 2
        if y < y_min - _EPS:
            code |= 4
        elif y > y_max + _EPS:
            code |= 8
        return code

    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    code0 = _code(x0, y0)
    code1 = _code(x1, y1)

    while True:
        if code0 == 0 and code1 == 0:
            c0 = torch.tensor([x0, y0], dtype=p0.dtype, device=p0.device)
            c1 = torch.tensor([x1, y1], dtype=p1.dtype, device=p1.device)
            c0 = torch.stack([c0[0].clamp(x_min, x_max), c0[1].clamp(y_min, y_max)])
            c1 = torch.stack([c1[0].clamp(x_min, x_max), c1[1].clamp(y_min, y_max)])
            if torch.dist(c0, c1) <= _EPS:
                return None
            return c0, c1
        if code0 & code1:
            return None
        out_code = code0 if code0 != 0 else code1

        if out_code & 8:
            denom = y1 - y0
            if abs(denom) <= _EPS:
                return None
            x = x0 + (x1 - x0) * (y_max - y0) / denom
            y = y_max
        elif out_code & 4:
            denom = y1 - y0
            if abs(denom) <= _EPS:
                return None
            x = x0 + (x1 - x0) * (y_min - y0) / denom
            y = y_min
        elif out_code & 2:
            denom = x1 - x0
            if abs(denom) <= _EPS:
                return None
            y = y0 + (y1 - y0) * (x_max - x0) / denom
            x = x_max
        else:
            denom = x1 - x0
            if abs(denom) <= _EPS:
                return None
            y = y0 + (y1 - y0) * (x_min - x0) / denom
            x = x_min

        if out_code == code0:
            x0, y0 = x, y
            code0 = _code(x0, y0)
        else:
            x1, y1 = x, y
            code1 = _code(x1, y1)


def _clip_lines(line_points: torch.Tensor, width: float, height: float) -> torch.Tensor:
    if line_points.numel() == 0:
        return line_points.reshape(0, 4)
    clipped = []
    for line in line_points:
        p0 = line[:2]
        p1 = line[2:]
        result = _clip_line_to_bounds(p0, p1, width, height)
        if result is None:
            continue
        c0, c1 = result
        if torch.dist(c0, c1) <= _EPS:
            continue
        clipped.append(torch.cat([c0, c1]))
    if not clipped:
        return line_points.new_empty((0, 4))
    return torch.stack(clipped, dim=0)


def _hflip_targets(target: Dict[str, torch.Tensor], width: float) -> Dict[str, torch.Tensor]:
    if target.get("boxes") is not None:
        boxes = target["boxes"].clone()
        x_min = boxes[:, 0].clone()
        x_max = boxes[:, 2].clone()
        boxes[:, 0] = width - x_max
        boxes[:, 2] = width - x_min
        boxes[:, 0::2] = _clamp_unit_interval(boxes[:, 0::2])
        boxes[:, 1::2] = _clamp_unit_interval(boxes[:, 1::2])
        target["boxes"] = boxes
    if target.get("line_points") is not None and target["line_points"].numel() > 0:
        lines = target["line_points"].clone()
        lines[:, [0, 2]] = width - lines[:, [0, 2]]
        target["line_points"] = _clip_lines(lines, 1.0, 1.0)
    if target.get("ellipses") is not None and target["ellipses"].numel() > 0:
        ellipses = target["ellipses"].clone()
        ellipses[:, 0] = width - ellipses[:, 0]
        ellipses[:, 5] = -ellipses[:, 5]
        ellipses[:, 0] = _clamp_unit_interval(ellipses[:, 0])
        ellipses[:, 1] = _clamp_unit_interval(ellipses[:, 1])
        target["ellipses"] = ellipses
    return target


def _resize_targets(
    target: Dict[str, torch.Tensor],
    old_size: Tuple[int, int],
    new_size: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    return target


def _crop_targets(
    target: Dict[str, torch.Tensor],
    top: int,
    left: int,
    height: int,
    width: int,
    img_height: int,
    img_width: int,
) -> Dict[str, torch.Tensor]:
    if target.get("boxes") is not None:
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]]*img_width - float(left))/width
        boxes[:, [1, 3]] = (boxes[:, [1, 3]]*img_height - float(top))/height
        target["boxes"] = boxes

    if target.get("line_points") is not None and target["line_points"].numel() > 0:
        lines = target["line_points"].clone()
        lines[:, [0, 2]] = (lines[:, [0, 2]]*img_width - float(left))/width
        lines[:, [1, 3]] = (lines[:, [1, 3]]*img_height - float(top))/height
        target["line_points"] = lines

    if target.get("ellipses") is not None and target["ellipses"].numel() > 0:
        ellipses = target["ellipses"].clone()
        ellipses[:, 0] = (ellipses[:, 0]*img_width - float(left))/width
        ellipses[:, 1] = (ellipses[:, 1]*img_height - float(top))/height
        ellipses[:, 2] += math.log(img_width / float(width))
        ellipses[:, 3] += math.log(img_height / float(height))
        target["ellipses"] = ellipses

    if target.get("line_points") is not None:
        target["line_points"] = _clip_lines(target["line_points"], 1.0, 1.0)

    if target.get("boxes") is not None and target["boxes"].numel() > 0:
        boxes = target["boxes"]
        boxes[:, 0::2] = _clamp_unit_interval(boxes[:, 0::2])
        boxes[:, 1::2] = _clamp_unit_interval(boxes[:, 1::2])
        target["boxes"] = boxes

    if target.get("ellipses") is not None and target["ellipses"].numel() > 0:
        target["ellipses"][:, 0] = _clamp_unit_interval(target["ellipses"][:, 0])
        target["ellipses"][:, 1] = _clamp_unit_interval(target["ellipses"][:, 1])

    return target


def apply_affine_to_target(
    image: torch.Tensor,
    target: Dict[str, torch.Tensor],
    angle: float,
    translate: Tuple[int, int],
    scale: float,
    shear: Tuple[float, float],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    height, width = _get_hw(image)
    image = F.affine(
        image,
        angle=angle,
        translate=list(translate),
        scale=scale,
        shear=list(shear),
        interpolation=interpolation,
        fill=[*fill],
    )

    translate_norm = [translate[0] / float(width), translate[1] / float(height)]
    matrix = F._get_inverse_affine_matrix([0.5, 0.5], angle, translate_norm, scale, list(shear), inverted=False)
    matrix = torch.tensor(matrix, dtype=target["boxes"].dtype if target.get("boxes") is not None else torch.float32)

    if target.get("boxes") is not None:
        boxes = target["boxes"].to(dtype=matrix.dtype)
        boxes = _apply_affine_to_boxes(boxes, matrix)
        boxes[:, 0::2] = _clamp_unit_interval(boxes[:, 0::2])
        boxes[:, 1::2] = _clamp_unit_interval(boxes[:, 1::2])
        target["boxes"] = boxes

    if target.get("line_points") is not None:
        lines = target["line_points"].to(dtype=matrix.dtype)
        lines = _apply_affine_to_lines(lines, matrix)
        target["line_points"] = _clip_lines(lines, 1.0, 1.0)

    if target.get("ellipses") is not None:
        ellipses = target["ellipses"].to(dtype=matrix.dtype)
        points = ellipses[:, :2]
        warped = _apply_affine_to_points(points, matrix)
        ellipses[:, :2] = warped
        ellipses = _rotate_ellipses(ellipses, angle)
        ellipses = _scale_ellipses(ellipses, scale, scale)
        ellipses[:, 0] = _clamp_unit_interval(ellipses[:, 0])
        ellipses[:, 1] = _clamp_unit_interval(ellipses[:, 1])
        target["ellipses"] = ellipses

    return image, target


def apply_hflip(
    image: torch.Tensor,
    target: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    image = F.hflip(image)
    target = _hflip_targets(target, 1.0)
    return image, target


class ODTrainTransforms:
    def __init__(
        self,
        image_size: int = 224,
        p_affine: float = 1, #0.7,
        p_hflip: float = 0.5,
        p_crop: float = 0.2,
        affine_degrees: Tuple[float, float] = (-5.0, 5.0),
        affine_translate: Tuple[float, float] = (0.15, 0.15),
        affine_scale: Tuple[float, float] = (0.95, 1.05),
        crop_scale: Tuple[float, float] = (0.85, 1.0),
    ) -> None:
        self.image_size = image_size
        self.p_affine = p_affine
        self.p_hflip = p_hflip
        self.p_crop = p_crop
        self.affine_degrees = affine_degrees
        self.affine_translate = affine_translate
        self.affine_scale = affine_scale
        self.crop_scale = crop_scale
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)
        self.random_grayscale = transforms.RandomGrayscale(p=0.05)
        self.random_blur = transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.1)
        self.random_sharpness = transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.1)
        self.random_autocontrast = transforms.RandomAutocontrast(p=0.1)
        self.random_erasing = transforms.RandomErasing(p=0.1, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _random_resized_crop(
        self,
        image: torch.Tensor,
        target: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        height, width = _get_hw(image)
        top, left, crop_h, crop_w = RandomResizedCrop.get_params(
            image, scale=list(self.crop_scale), ratio=[1., 1.]#ratio=[0.8235, 0.8235]
        )
        image = F.resized_crop(
            image,
            top,
            left,
            crop_h,
            crop_w,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        target = _crop_targets(target, top, left, crop_h, crop_w, height, width)
        return image, target

    def _resize_to_output(
        self,
        image: torch.Tensor,
        target: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image = F.resize(
            image,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return image, _resize_targets(target, _get_hw(image), (self.image_size, self.image_size))

    def __call__(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None):
        image = _ensure_float_image(image)
        has_target = target is not None
        target_dict: Dict[str, torch.Tensor] = {} if target is None else target

        if random.random() < self.p_crop:
            image, target_dict = self._random_resized_crop(image, target_dict)
        else:
            image, target_dict = self._resize_to_output(image, target_dict)

        if random.random() < self.p_affine:
            height, width = _get_hw(image)
            angle, translate, scale, shear = RandomAffine.get_params(
                degrees=[*self.affine_degrees],
                translate=[*self.affine_translate],
                scale_ranges=[*self.affine_scale],
                shears=None,
                img_size=[width, height],
            )
            image, target_dict = apply_affine_to_target(
                image,
                target_dict,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=(0.0, 0.0),
            )

        if random.random() < self.p_hflip:
            image, target_dict = apply_hflip(image, target_dict)

        image = self.color_jitter(image)
        image = self.random_autocontrast(image)
        image = self.random_grayscale(image)
        image = self.random_blur(image)
        image = self.random_sharpness(image)
        image = self.normalize(image) # Only deactivate for viszualizations?

        # image = self.random_erasing(image) # (i dont want this anymore)
        if has_target:
            return image, target_dict
        return image


class ODValTransforms:
    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None):
        image = _ensure_float_image(image)
        height, width = _get_hw(image)
        image = F.resize(
            image,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if target is None:
            return self.normalize(image)
        target = _resize_targets(target, (height, width), (self.image_size, self.image_size))
        return self.normalize(image), target
