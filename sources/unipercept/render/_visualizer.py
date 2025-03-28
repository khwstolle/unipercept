# Adapted from:  ``detectron2/utils/visualizer.py``
from __future__ import annotations

import colorsys
import enum
import math
import typing as T

import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import numpy as np
import pycocotools.mask as mask_util
import torch
from matplotlib import patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

from ._colormap import colormap, random_color, random_jitters

if T.TYPE_CHECKING:
    from unipercept.data.types import Metadata

__all__ = ["ColorMode", "VisImage", "Visualizer", "LabelFlags"]


_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_KEYPOINT_THRESHOLD = 0.05


class ColorMode(enum.StrEnum):
    """Enum of different color modes to use for instance visualizations."""

    IMAGE = enum.auto()
    """Picks a random color for every instance and overlay segmentations with low opacity."""
    SEGMENTATION = enum.auto()
    """
    Let instances of the same category have similar colors
    (from metadata["thing_colors"]), and overlay them with
    high opacity. This provides more attention on the quality of segmentation.
    """
    IMAGE_BW = enum.auto()
    """
    Same as IMAGE, but convert all areas without masks to gray-scale.
    Only available for drawing per-instance mask predictions.
    """


class GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask.
    """

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons
        if isinstance(m, dict):
            # RLEs
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):  # uncompressed RLEs
                h, w = m["size"]
                assert h == height and w == width
                m = mask_util.frPyObjects(m, h, w)
            self._mask = mask_util.decode(m)[:, :]
            return

        if isinstance(m, list):  # list[ndarray]
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # assumed to be a binary mask
            assert m.shape[1] != 2, m.shape
            assert m.shape == (
                height,
                width,
            ), f"mask shape: {m.shape}, target dims: {height}, {width}"
            self._mask = m.astype("uint8")
            return

        raise ValueError(f"GenericMask cannot handle object {m} of type '{type(m)}'")

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = (
                    False  # if original format is polygon, does not have holes
                )
        return self._has_holes

    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(
            mask
        )  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(
            mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0  # type: ignore
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]  # type: ignore
        return res, has_holes

    def polygons_to_mask(self, polygons):
        rle = mask_util.frPyObjects(polygons, self.height, self.width)
        rle = mask_util.merge(rle)
        return mask_util.decode(rle)[:, :]

    def area(self):
        return self.mask.sum()

    def bbox(self):
        p = mask_util.frPyObjects(self.polygons, self.height, self.width)
        p = mask_util.merge(p)
        bbox = mask_util.toBbox(p)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox


class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3) in range [0, 255].
            scale (float): scale the input image.
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        """
        Args:
            img: same as in __init__.
        """
        img = img.astype("uint8")
        self.ax.imshow(
            img, extent=(0, self.width, self.height, 0), interpolation="nearest"
        )

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")

    def get_pil(self):
        import PIL.Image as pil_image

        out = self.get_image()
        return pil_image.fromarray(out).convert("RGB")


DEFAULT_JITTERMAP = random_jitters()


class LabelFlags(enum.IntFlag):
    NONE = 0
    ID = enum.auto()
    SCORE = enum.auto()
    CLASS = enum.auto()


DEFAULT_LABEL_FLAGS = LabelFlags.CLASS


class Visualizer:
    def __init__(
        self,
        img_rgb: torch.Tensor | np.ndarray,
        info: Metadata | None,
        scale=1.0,
        instance_mode=ColorMode.IMAGE,
        jittermap=DEFAULT_JITTERMAP,
        font_family: str = "sans-serif",
        font_weight: str = "normal",
    ):
        self.font_family = font_family
        self.font_weight = font_weight

        if isinstance(img_rgb, torch.Tensor):
            from torchvision.transforms.functional import to_pil_image

            img_rgb = to_pil_image(img_rgb)
        elif isinstance(img_rgb, np.ndarray):
            if img_rgb.ndim == 2:
                img_rgb = np.repeat(img_rgb[:, :, None], 3, axis=2)
            elif img_rgb.shape[2] == 4:
                img_rgb = img_rgb[:, :, :3]

        self._jittermap = jittermap
        self._colormap = colormap(rgb=True, maximum=255)
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.metadata = info
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")

        # too small texts are useless, therefore clamp to 9
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 100, 6 // scale
        )
        self._instance_mode = instance_mode
        self.keypoint_threshold = _KEYPOINT_THRESHOLD

    def get_mask_color(self, sem_id: int, ins_id=0):
        if self.metadata:
            clr = np.asarray(self.metadata.semantic_classes[sem_id].color)
        else:
            clr = self._colormap[sem_id % len(self._colormap)]

        assert isinstance(
            clr, np.ndarray
        ), f"Color of semantic class {sem_id} must be an ndarray, got {type(clr)}"

        clr = mplc.to_rgb(clr / 255.0)
        clr = self._jittermap.apply(ins_id, clr)

        # TODO: instance ID?
        return clr

    def get_mask_label(
        self, sem_id, ins_id=0, labels: LabelFlags = DEFAULT_LABEL_FLAGS
    ):
        lbl = []

        if LabelFlags.CLASS & labels != 0:
            if self.metadata:
                cls_name = self.metadata.semantic_classes[sem_id].name.title()
            else:
                cls_name = f"[{sem_id}]"
            lbl.append(cls_name)

        if LabelFlags.ID & labels != 0:
            if ins_id > 0:
                lbl.append(f"{ins_id}")

        return "".join(lbl)

    def draw_boxes(
        self,
        boxes: torch.Tensor,
        labels: T.Sequence[T.Any] | None = None,
        **kwargs,
    ):
        if labels is None:
            labels = [None] * len(boxes)

        for box, lbl in zip(boxes.unbind(0), labels, strict=False):
            assert box.ndim == 1, box.shape
            assert box.numel() == 4, box.shape
            self.draw_box(box, **kwargs)
            if lbl is not None:
                self.draw_text(f"{lbl}", (box[0], box[1]))

        return self.output

    def draw_segmentation(
        self,
        pan: torch.Tensor,
        *,
        alpha=1.0,
        labels: LabelFlags = DEFAULT_LABEL_FLAGS,
        **kwargs,
    ):
        from unipercept.tensors import PanopticTensor

        pan = pan.as_subclass(PanopticTensor)

        assert pan.ndim == 2, pan.shape

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(self._create_grayscale_image(pan.get_nonempty()))

        for sem_id, ins_id, mask in pan.get_masks():
            mask_color = self.get_mask_color(sem_id, ins_id)
            mask_label = self.get_mask_label(sem_id, ins_id, labels)

            self.draw_binary_mask(
                mask,
                color=mask_color,
                text=mask_label,
                alpha=alpha,
                **kwargs,
            )

        # draw mask for all instances second
        # all_instances = list(pan.get_instance_masks())
        # if len(all_instances) == 0:
        #     return self.output
        # ins_ids, mask = list(zip(*all_instances))
        # category_ids = [x["category_id"] for x in sinfo]

        # try:
        #     scores = [x["score"] for x in sinfo]
        # except KeyError:
        #     scores = None
        # labels = _create_text_labels(
        #     category_ids, scores, self.metadata["thing_classes"], [x.get("iscrowd", 0) for x in sinfo]
        # )

        # try:
        #     colors = [self._jitter([x / 255 for x in self.metadata["thing_colors"][c]]) for c in category_ids]
        # except AttributeError:
        #     colors = None
        # self.overlay_instances(masks=masks, labels=labels, assigned_colors=colors, alpha=alpha)

        return self.output

    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5,
    ):
        num_instances = 0
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = self._convert_keypoints(keypoints)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [
                random_color(rgb=True, maximum=1) for _ in range(num_instances)
            ]
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    # skip small mask without polygon
                    if len(masks[i].polygons) == 0:
                        continue

                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                    or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(
                    self.output.height * self.output.width
                )
                lighter_color = self._change_color_brightness(
                    color, brightness_factor=0.7
                )
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1, 2.5)
                    * 0.5
                    * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        # draw keypoints
        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                self.draw_and_connect_keypoints(keypoints_per_instance)

        return self.output

    def overlay_rotated_instances(self, boxes=None, labels=None, assigned_colors=None):
        num_instances = len(boxes)

        if assigned_colors is None:
            assigned_colors = [
                random_color(rgb=True, maximum=1) for _ in range(num_instances)
            ]
        if num_instances == 0:
            return self.output

        # Display in largest to smallest order to reduce occlusion.
        if boxes is not None:
            areas = boxes[:, 2] * boxes[:, 3]

        sorted_idxs = np.argsort(-areas).tolist()
        # Re-order overlapped instances in descending order.
        boxes = boxes[sorted_idxs]
        labels = [labels[k] for k in sorted_idxs] if labels is not None else None
        colors = [assigned_colors[idx] for idx in sorted_idxs]

        for i in range(num_instances):
            self.draw_rotated_box_with_label(
                boxes[i],
                edge_color=colors[i],
                label=labels[i] if labels is not None else None,
            )

        return self.output

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
        alpha=1.0,
    ):
        if not font_size:
            font_size = self._default_font_size

        # Fix contrast
        # color = np.maximum(list(mplc.to_rgb(color)), 0.33)
        # color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family=self.font_family,  # "sans-serif",
            fontweight=self.font_weight,  # "bold",
            bbox={
                "facecolor": "black",
                "alpha": 0.99 * alpha,
                "boxstyle": "round,pad=0.05",
                "edgecolor": "none",
            },
            verticalalignment="center",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
            alpha=alpha,
        )
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        if isinstance(box_coord, torch.Tensor):
            box_coord = box_coord.cpu().numpy()
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    def draw_rotated_box_with_label(
        self, rotated_box, alpha=0.5, edge_color="g", line_style="-", label=None
    ):
        """
        Draw a rotated box with label on its top-left corner.

        Args:
            rotated_box (tuple): a tuple containing (cnt_x, cnt_y, w, h, angle),
                where cnt_x and cnt_y are the center coordinates of the box.
                w and h are the width and height of the box. angle represents how
                many degrees the box is rotated CCW with regard to the 0-degree box.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.
            label (string): label for rotated box. It will not be rendered when set to None.

        Returns:
            output (VisImage): image object with box drawn.
        """
        cnt_x, cnt_y, w, h, angle = rotated_box
        area = w * h
        # use thinner lines when the box is small
        linewidth = self._default_font_size / (
            6 if area < _SMALL_OBJECT_AREA_THRESH * self.output.scale else 3
        )

        theta = angle * math.pi / 180.0
        c = math.cos(theta)
        s = math.sin(theta)
        rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
        # x: left->right ; y: top->down
        rotated_rect = [
            (s * yy + c * xx + cnt_x, c * yy - s * xx + cnt_y) for (xx, yy) in rect
        ]
        for k in range(4):
            j = (k + 1) % 4
            self.draw_line(
                [rotated_rect[k][0], rotated_rect[j][0]],
                [rotated_rect[k][1], rotated_rect[j][1]],
                color=edge_color,
                linestyle="--" if k == 1 else line_style,
                linewidth=linewidth,
            )

        if label is not None:
            text_pos = rotated_rect[1]  # topleft corner

            height_ratio = h / np.sqrt(self.output.height * self.output.width)
            label_color = self._change_color_brightness(
                edge_color, brightness_factor=0.7
            )
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                * 0.5
                * self._default_font_size
            )
            self.draw_text(
                label, text_pos, color=label_color, font_size=font_size, rotation=angle
            )

        return self.output

    def draw_circle(self, circle_coord, color, radius=3):
        """
        Args:
            circle_coord (list(int) or tuple(int)): contains the x and y coordinates
                of the center of the circle.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            radius (int): radius of the circle.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x, y = circle_coord
        self.output.ax.add_patch(
            patches.Circle(circle_coord, radius=radius, fill=True, color=color)
        )
        return self.output

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        """
        Args:
            x_data (list[int]): a list containing x values of all the points being drawn.
                Length of list should match the length of y_data.
            y_data (list[int]): a list containing y values of all the points being drawn.
                Length of list should match the length of x_data.
            color: color of the line. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            linestyle: style of the line. Refer to `matplotlib.lines.Line2D`
                for a full list of formats that are accepted.
            linewidth (float or None): width of the line. When it's None,
                a default value will be computed and used.

        Returns:
            output (VisImage): image object with line drawn.
        """
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        self.output.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=linewidth * self.output.scale,
                color=color,
                linestyle=linestyle,
            )
        )
        return self.output

    def draw_binary_mask(
        self,
        binary_mask,
        color=None,
        *,
        edge_size: float = 0.0,
        edge_color=None,
        text=None,
        alpha=0.5,
        area_threshold=10,
        depth_map: np.ndarray | torch.Tensor | None = None,
        depth_label_threshold: float = 0.5,
        depth_label_alpha_modulation: float = 0.9,
        force_polygons: bool = True,
    ) -> VisImage:
        binary_mask = np.asarray(binary_mask)
        if color is None:
            color = random_color(rgb=True, maximum=1)
        color = mplc.to_rgb(color)

        if edge_color is None:
            # edge_color = self._change_color_brightness(color, brightness_factor=0.7)
            edge_color = (1.0, 1.0, 1.0)
        edge_color = mplc.to_rgb(edge_color)

        has_valid_segment = False
        binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
        mask = GenericMask(binary_mask, self.output.height, self.output.width)
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])

        if not mask.has_holes or force_polygons:
            # draw polygons for regular masks
            for segment in mask.polygons:
                rle_mask = mask_util.frPyObjects([segment], shape2d[0], shape2d[1])
                area = mask_util.area(rle_mask)
                if area < (area_threshold or 0):
                    continue
                has_valid_segment = True
                segment = segment.reshape(-1, 2)
                self.draw_polygon(
                    segment,
                    color=color,
                    edge_color=edge_color,
                    edge_size=edge_size,
                    alpha=alpha,
                )
        else:
            # draw mask contour for masks with holes

            # TODO: Use Path/PathPatch to draw vector graphics:
            # https://stackoverflow.com/questions/8919719/how-to-plot-a-complex-polygon
            rgba = np.zeros(shape2d + (4,), dtype="float32")
            rgba[:, :, :3] = color
            rgba[:, :, 3] = (mask.mask == 1).astype("float32") * alpha
            has_valid_segment = True
            self.output.ax.imshow(
                rgba, extent=(0, self.output.width, self.output.height, 0)
            )

        label_alpha = 1.0

        if has_valid_segment:
            label_alpha = 1.0

            # Depth modulation
            if depth_map is not None:
                depth_map = np.asarray(depth_map, dtype=np.float32)
                if depth_map.max() > 1.0:
                    depth_map = depth_map / depth_map.max()
                assert depth_map is not None
                depth_valid = (binary_mask > 0) & (depth_map > 0)
                if depth_valid.any():
                    depth_mean = depth_map[depth_valid].mean()
                    if depth_mean > depth_label_threshold:
                        text = None
                    if depth_label_alpha_modulation > 0.0:
                        label_alpha *= np.clip(
                            1 - depth_mean * depth_label_alpha_modulation, 0.0, 1.0
                        )

            # Draw text
            if text is not None:
                lighter_color = "white"  # self._change_color_brightness(color, brightness_factor=0.9)
                self._draw_text_in_mask(
                    binary_mask, text, lighter_color, alpha=label_alpha
                )
        return self.output

    def draw_soft_mask(self, soft_mask, color=None, *, text=None, alpha=0.5):
        """
        Args:
            soft_mask (ndarray): float array of shape (H, W), each value in [0, 1].
            color: color of the mask. Refer to `matplotlib.colors` for a full list of
                formats that are accepted. If None, will pick a random color.
            text (str): if None, will be drawn on the object
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with mask drawn.
        """
        soft_mask = np.asarray(soft_mask)

        if color is None:
            color = random_color(rgb=True, maximum=1)
        color = mplc.to_rgb(color)

        shape2d = (soft_mask.shape[0], soft_mask.shape[1])
        rgba = np.zeros(shape2d + (4,), dtype="float32")
        rgba[:, :, :3] = color
        rgba[:, :, 3] = soft_mask * alpha
        self.output.ax.imshow(
            rgba, extent=(0, self.output.width, self.output.height, 0)
        )

        if text is not None:
            lighter_color = (
                "white"  # self._change_color_brightness(color, brightness_factor=0.7)
            )
            binary_mask = (soft_mask > 0.5).astype("uint8")
            self._draw_text_in_mask(binary_mask, text, lighter_color)
        return self.output

    def draw_polygon(self, segment, color, edge_color=None, edge_size=0.0, alpha=0.5):
        """
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted. If not provided, a darker shade
                of the polygon color will be used instead.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with polygon drawn.
        """
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(
                    color, brightness_factor=-0.7
                )
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color if edge_size > 0.0 else None,
            linewidth=max(
                self._default_font_size // 12 * edge_size * self.output.scale, 0.5
            ),
        )
        self.output.ax.add_patch(polygon)
        return self.output

    """
    Internal methods:
    """

    def _jitter(self, color):
        """
        Randomly modifies given color to produce a slightly different color than the color given.

        Args:
            color (tuple[double]): a tuple of 3 elements, containing the RGB values of the color
                picked. The values in the list are in the [0.0, 1.0] range.

        Returns:
            jittered_color (tuple[double]): a tuple of 3 elements, containing the RGB values of the
                color after being jittered. The values in the list are in the [0.0, 1.0] range.
        """
        color = mplc.to_rgb(color)
        vec = np.random.rand(3)
        # better to do it in another color space
        vec = vec / np.linalg.norm(vec) * 0.5
        res = np.clip(vec + color, 0, 1)
        return tuple(res)

    def _create_grayscale_image(self, mask=None):
        """
        Create a grayscale version of the original image.
        The colors in masked area, if given, will be kept.
        """
        img_bw = self.img.astype("f4").mean(axis=2)
        img_bw = np.stack(arrays=[img_bw] * 3, axis=2)
        if mask is not None:
            mask = np.asarray(mask)
            img_bw[mask] = self.img[mask]
        return img_bw

    def _change_color_brightness(self, color, brightness_factor):
        """
        Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
        less or more saturation than the original color.

        Args:
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
                0 will correspond to no change, a factor in [-1.0, 0) range will result in
                a darker color and a factor in (0, 1.0] range will result in a lighter color.

        Returns:
            modified_color (tuple[double]): a tuple containing the RGB values of the
                modified color. Each value in the tuple is in the [0.0, 1.0] range.
        """
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = max(modified_lightness, 0.0)
        modified_lightness = min(modified_lightness, 1.0)
        modified_color = colorsys.hls_to_rgb(
            polygon_color[0], modified_lightness, polygon_color[2]
        )
        return tuple(np.clip(modified_color, 0.0, 1.0))

    def _convert_boxes(self, boxes):
        """Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension."""
        if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
            return boxes.tensor.detach().numpy()
        return np.asarray(boxes)

    def _convert_masks(self, masks_or_polygons):
        """
        Convert different format of masks or polygons to a tuple of masks and polygons.

        Returns:
            list[GenericMask]:
        """

        m = masks_or_polygons
        if isinstance(m, PolygonMasks):
            m = m.polygons
        if isinstance(m, BitMasks):
            m = m.tensor.numpy()
        if isinstance(m, torch.Tensor):
            m = m.numpy()
        ret = []
        for x in m:
            if isinstance(x, GenericMask):
                ret.append(x)
            else:
                ret.append(GenericMask(x, self.output.height, self.output.width))
        return ret

    def _draw_text_in_mask(self, binary_mask, text, color, **kwargs):
        """Find proper places to draw text given a binary mask."""
        # TODO sometimes drawn on wrong objects. the heuristics here can improve.
        _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, 8
        )
        if stats[1:, -1].size == 0:
            return
        largest_component_id = np.argmax(stats[1:, -1]) + 1

        # draw text on the largest component, as well as other very large components.
        for cid in range(1, _num_cc):
            if cid == largest_component_id or stats[cid, -1] > _LARGE_MASK_AREA_THRESH:
                # median is more stable than centroid
                # center = centroids[largest_component_id]
                center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
                self.draw_text(text, center, color=color, **kwargs)
