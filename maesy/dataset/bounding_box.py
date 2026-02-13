from typing import Tuple, Optional


class BoundingBox:
    """Bounding Box representation stored internally as [x_min, y_min, x_max, y_max].

    Use factory methods if you have a different representation (for example
    `BoundingBox.from_xywh(...)`).
    """

    def __init__(self, cls_id: int, x_min: float, y_min: float, x_max: float, y_max: float, normalized=False):
        """
        Initialize bounding box using corner coordinates.

        Args:
            :param cls_id: Class ID
            :param x_min: Minimum x coordinate (left)
            :param y_min: Minimum y coordinate (top)
            :param x_max: Maximum x coordinate (right)
            :param y_max: Maximum y coordinate (bottom)
            :param normalized: Whether the coordinates are already normalized to [0,1] (default False)
        """
        if x_max < x_min or y_max < y_min:
            raise ValueError("x_max must be >= x_min and y_max must be >= y_min")

        self.cls_id = cls_id
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.normalized = normalized

    @classmethod
    def from_xywh(cls, cls_id: int, center_x: float, center_y: float, width: float, height: float, normalized=False) -> "BoundingBox":
        """Create a BoundingBox from center-format (cx, cy, w, h).

        Args:
            :param cls_id: Class ID
            :param center_x: center x coordinate
            :param center_y: center y coordinate
            :param width: box width
            :param height: box height
            :param normalized: Whether the coordinates are already normalized to [0,1] (default False)

        Returns:
            BoundingBox instance
        """
        half_w = width / 2.0
        half_h = height / 2.0
        x_min = center_x - half_w
        y_min = center_y - half_h
        x_max = center_x + half_w
        y_max = center_y + half_h
        return cls(cls_id, x_min, y_min, x_max, y_max, normalized)

    @classmethod
    def from_str(cls, yolo_row: str, normalized: bool=True) -> "BoundingBox":
        """
        Create a BoundingBox from string format "cls cx cy w h" as used in YOLO annotations.
        Assumes normalized values by default, but can be set to False if the values in the string are in pixel coordinates.

        Args:
            :param yolo_row: String in format "cls cx cy w h"
            :param normalized: Whether the coordinates in the string are normalized to [0,1] (default True)

        Returns:
            BoundingBox instance
        """
        splits = yolo_row.split()
        if len(splits) != 5:
            raise ValueError(f"Invalid YOLO annotation format: {yolo_row}. Expected format: 'cls cx cy w h'")
        cls_id = int(splits[0])
        cx = float(splits[1])
        cy = float(splits[2])
        w = float(splits[3])
        h = float(splits[4])

        return cls(cls_id, cx, cy, w, h, normalized)

    def cls(self):
        return self.cls_id

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        """Return bounding box in [x_min, y_min, x_max, y_max] format."""
        return self.x_min, self.y_min, self.x_max, self.y_max

    def as_xywh(self) -> Tuple[float, float, float, float]:
        """Return bounding box in [center_x, center_y, width, height] format."""
        center_x = (self.x_min + self.x_max) / 2.0
        center_y = (self.y_min + self.y_max) / 2.0
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        return center_x, center_y, width, height

    def as_xyxy_normalized(self, image_width: float, image_height: float) -> Tuple[float, float, float, float]:
        """
        Return xyxy coordinates normalized to [0,1] by image size.
        If the bounding box is already normalized, this will just return .as_xyxy()
        """
        if self.normalized:
            return self.as_xyxy()
        return self.x_min / image_width, self.y_min / image_height, self.x_max / image_width, self.y_max / image_height,

    def as_xywh_normalized(self, image_width: float, image_height: float) -> Tuple[float, float, float, float]:
        """
        Return xywh (center) coordinates normalized to [0,1] by image size.
        If the bounding box is already normalized, this will just return .as_xywh()
        """
        cx, cy, w, h = self.as_xywh()
        if self.normalized:
            return cx, cy, w, h
        return cx / image_width, cy / image_height, w / image_width, h / image_height

    def clip(self, image_width: Optional[int], image_height: Optional[int]) -> None:
        """
        Clip the bounding box to image boundaries in-place.
        If the bounding box is normalized, image_width and image_height are ignored.
        Otherwise, they must be specified

        Args:
        :param image_width: Image width (required if not normalized)
        :param image_height: Image height (required if not normalized)
        """

        if self.normalized:
            image_width = 1
            image_height = 1
        else:
            if image_width is None or image_height is None:
                raise ValueError("image_width and image_height must be specified for non-normalized bounding boxes")

        self.x_min = max(0.0, min(self.x_min, float(image_width)))
        self.y_min = max(0.0, min(self.y_min, float(image_height)))
        self.x_max = max(0.0, min(self.x_max, float(image_width)))
        self.y_max = max(0.0, min(self.y_max, float(image_height)))
        # ensure valid after clipping
        if self.x_max < self.x_min:
            self.x_max = self.x_min
        if self.y_max < self.y_min:
            self.y_max = self.y_min

    def area(self) -> float:
        """Return the area of the bounding box (in pixels if coords are pixels)."""
        w = max(0.0, self.x_max - self.x_min)
        h = max(0.0, self.y_max - self.y_min)
        return w * h

    def iou(self, other: "BoundingBox") -> float:
        """Compute Intersection over Union (IoU) with another bounding box."""
        ix_min = max(self.x_min, other.x_min)
        iy_min = max(self.y_min, other.y_min)
        ix_max = min(self.x_max, other.x_max)
        iy_max = min(self.y_max, other.y_max)

        iw = max(0.0, ix_max - ix_min)
        ih = max(0.0, iy_max - iy_min)
        inter = iw * ih
        union = self.area() + other.area() - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    def __repr__(self) -> str:
        return f"BoundingBox(cls_id={self.cls_id}, x_min={self.x_min:.3f}, y_min={self.y_min:.3f}, x_max={self.x_max:.3f}, y_max={self.y_max:.3f})"

