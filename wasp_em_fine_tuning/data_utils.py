import os
import yaml
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Callable, Iterator, Self

import cv2
import h5py
import numpy as np
import torch
from numpy.typing import ArrayLike
from scipy.ndimage import center_of_mass, distance_transform_edt

from sam2.utils.misc import get_connected_components
from sam2.utils.transforms import SAM2Transforms

RngInitType = int | str | np.random.Generator


def _get_config(config: dict | str) -> dict:
    if isinstance(config, str):
        if os.path.exists(config):
            with open(config) as f:
                config = yaml.safe_load(f)
        else:
            config = yaml.safe_load(config)
    return config


def _load_h5_dataset(file, dataset: str | tuple[str, ...]):
    if isinstance(dataset, str):
        dset = file[dataset]
    else:
        dset = file
        for name in dataset:
            dset = dset[name]
    return dset


def get_rng(rng: RngInitType | None = None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        rng = rng
    else:
        if isinstance(rng, str):  # convert to int with hash function
            rng = abs(hash(rng))
        rng = np.random.default_rng(rng)
    return rng


def to_rgb(image: ArrayLike) -> np.ndarray:
    if image.max() > 1:
        image = image.astype(float) / 255
    if image.shape[-1] != 3:
        image = np.stack([image] * 3, axis=-1)
    return image


class ImagePreprocessor:
    def __init__(self, transforms: SAM2Transforms | None):
        self.transforms = transforms

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2 or (image.ndim == 3 and image.shape[0] == image.shape[1] and image.shape[2] == 3):
            image = image[np.newaxis, ...]
            is_image = True
        else:
            is_image = False
        if image.ndim == 3:
            image = to_rgb(image)
        if self.transforms:
            frames = [self.transforms(frame).permute(1, 2, 0) for frame in image]
            if is_image:
                image = frames[0]
            else:
                image = torch.stack(frames)
        return image


@dataclass
class HDF5DataIndex:
    path: str
    dataset: str | tuple[str, ...]
    frame_axis: int | None = None
    slice: tuple | None = None
    shape: tuple | None = None
    preprocessor: ImagePreprocessor | None = None

    def load(self, preprocess=True) -> np.ndarray:
        with h5py.File(self.path, 'r') as f:
            dset = _load_h5_dataset(f, self.dataset)
            if self.slice is not None:
                data = dset[*self.slice]
            else:
                data = dset[...]
        data = np.asarray(data)
        if data.ndim > 2 and self.frame_axis is not None:
            assert data.ndim == 3
            axes_order = [0, 1, 2]
            axes_order = [self.frame_axis] + axes_order[:self.frame_axis] + axes_order[(self.frame_axis + 1):]
            data = np.permute_dims(data, axes=axes_order)
            if data.shape[0] == 1:
                data = data[0]
        if preprocess and self.preprocessor:
            data = self.preprocessor(data)
        return data


def _int_slice(slc) -> int:
    if isinstance(slc, slice):
        step = slc.step if slc.step is not None else 1
        n = (slc.stop - slc.start) // step
        if n > 1:
            raise ValueError(f"Not a singular-value slice: {slc}")
        else:
            assert slc.start is not None
            return slc.start
    else:
        return int(slc)


@dataclass
class SegmentationDataIndex:
    image: HDF5DataIndex
    label: HDF5DataIndex

    @classmethod
    def from_frame_index(cls, frame_index: Self, num_frames: int) -> Self:
        image_index = frame_index.image
        axis = image_index.frame_axis
        if axis is None:
            raise ValueError("frame_index must have its frame_axis set")
        image_shape = image_index.shape
        if image_shape is None:
            with h5py.File(image_index.path, 'r') as f:
                dset = _load_h5_dataset(f, image_index.dataset)
                image_shape = dset.shape
        frame_start = _int_slice(image_index.slice[axis])
        frame_end = frame_start + num_frames
        if frame_end > image_shape[axis]:
            raise ValueError(f"num_frames of {num_frames} is out of bounds for frame start of {frame_start} and only "
                             f"{image_shape[axis]} frames in the video")
        new_slice = list(image_index.slice)
        new_slice[axis] = slice(frame_start, frame_end)
        image_params = asdict(image_index)
        image_params["slice"] = new_slice
        label_params = asdict(frame_index.label)
        label_params["slice"] = new_slice
        return cls(HDF5DataIndex(**image_params), HDF5DataIndex(**label_params))


class SegmentationDataIndexer(ABC):
    def __init__(
            self,
            data_files: dict[str, SegmentationDataIndex],
            axis: int | None = None,
            preprocessor: ImagePreprocessor | None = None,
    ):
        self.data_files = data_files
        self.axis = axis
        self.image_shapes: dict[str, tuple[int, int, int]] = {}
        self.preprocessor = preprocessor
        self._build_indexer_data()

    @classmethod
    def from_config(cls, config: dict | str, **kwargs):
        config = _get_config(config)
        if kwargs:
            config = config.copy()
            config.update(kwargs)
        data_files = parse_data_files(config)
        return cls(
            data_files,
            axis=config.get("axis"),
            preprocessor=config.get("preprocessor"),
        )

    def __repr__(self):
        params = [f"{len(self.data_files)} volumes"]
        params.extend([f"{k}={v}" for k, v in self.__dict__.items() if k not in ["data_files", "image_shapes"]])
        return f"{self.__class__.__name__}({', '.join(params)})"

    def list_volumes(self):
        return list(self.data_files.keys())

    def _build_indexer_data(self):
        for key, data in self.data_files.items():
            with h5py.File(data.image.path, 'r') as f:
                dset = _load_h5_dataset(f, data.image.dataset)
                image_shape = dset.shape
            with h5py.File(data.label.path, 'r') as f:
                dset = _load_h5_dataset(f, data.label.dataset)
                label_shape = dset.shape
            assert image_shape == label_shape, \
                f"image and label shapes do not match ({image_shape} vs. {label_shape})" \
                f" for {data.image.path} and {data.label.path}"
            # Add slices for image patches or whole images, along each axis
            self.image_shapes[key] = image_shape

    def subset(self, volumes: str | list, **kwargs) -> Self:
        if isinstance(volumes, str):
            volumes = [volumes]
        data_files = {k: v for k, v in self.data_files.items() if k in volumes}
        params = {"axis": self.axis}
        params.update(kwargs)
        return type(self)(data_files, **params)

    @abstractmethod
    def _get_index(self, *args, **kwargs) -> SegmentationDataIndex:
        pass

    def get_index(self, *args, **kwargs):
        index = self._get_index(*args, **kwargs)
        index.image.preprocessor = self.preprocessor
        return index


def _resolve_index(index, size) -> int | tuple[int, ...]:
    try:
        index = int(index)
        if index < 0:
            index += size
        return index
    except TypeError:
        try:
            return tuple(_resolve_index(i, s) for i, s in zip(index, size))
        except TypeError:
            return tuple(_resolve_index(i, size) for i in index)


class SegmentationVideoIndexer(SegmentationDataIndexer):
    def _get_index(
            self,
            volume_key: str,
            frame_start: int,
            frame_end: int,
            patch_start: tuple[int, int] | None = None,
            patch_end: tuple[int, int] | None = None,
            axis: int | None = None,
    ) -> SegmentationDataIndex:
        if axis is None:
            if self.axis is None:
                raise ValueError("Axis must be specified")
            axis = self.axis
        axis = _resolve_index(axis, 3)

        img_shape = self.image_shapes[volume_key]
        img_frames = img_shape[axis]
        frame_start = _resolve_index(frame_start, img_frames)
        frame_end = _resolve_index(frame_end, img_frames)

        img_shape_2dim = img_shape[:axis] + img_shape[axis+1:]
        if patch_start is None:
            patch_start = [0, 0]
        else:
            patch_start = _resolve_index(patch_start, img_shape_2dim)
        if patch_end is None:
            patch_end = img_shape_2dim
        else:
            patch_end = _resolve_index(patch_end, img_shape_2dim)

        if axis == 0:
            slice_ = (slice(frame_start, frame_end),
                      slice(patch_start[0], patch_end[0]),
                      slice(patch_start[1], patch_end[1]))
        elif axis == 1:
            slice_ = (slice(patch_start[0], patch_end[0]),
                      slice(frame_start, frame_end),
                      slice(patch_start[1], patch_end[1]))
        else:
            slice_ = (slice(patch_start[0], patch_end[0]),
                      slice(patch_start[1], patch_end[1]),
                      slice(frame_start, frame_end))

        data = self.data_files[volume_key]
        return SegmentationDataIndex(
            HDF5DataIndex(data.image.path, data.image.dataset, axis, slice_, img_shape),
            HDF5DataIndex(data.label.path, data.label.dataset, axis, slice_, img_shape),
        )


class SegmentationVideoSampler(SegmentationVideoIndexer):
    def __init__(
            self,
            data_files: dict[str, SegmentationDataIndex],
            axis: int | None = None,
            preprocessor: ImagePreprocessor | None = None,
            num_frames: int | None = None,
            patch_size: tuple[int, int] = None,
            rng: RngInitType | None = None,
    ):
        super().__init__(data_files=data_files, axis=axis, preprocessor=preprocessor)
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.rng = get_rng(rng)

    @classmethod
    def from_config(cls, config: dict | str, **kwargs):
        config = _get_config(config)
        if kwargs:
            config = config.copy()
            config.update(kwargs)
        data_files = parse_data_files(config)
        return cls(
            data_files,
            axis=config.get("axis"),
            preprocessor=config.get("preprocessor"),
            num_frames=config.get("num_frames"),
            patch_size=config.get("patch_size"),
            rng=config.get("rng"),
        )

    def subset(self, volumes: str | list, **kwargs) -> Self:
        if isinstance(volumes, str):
            volumes = [volumes]
        data_files = {k: v for k, v in self.data_files.items() if k in volumes}
        params = {"axis": self.axis, "num_frames": self.num_frames, "patch_size": self.patch_size, "rng": self.rng}
        params.update(kwargs)
        return type(self)(data_files, **params)

    def get_sample(
            self,
            volume_key: str | None = None,
            frame_start: int | None = None,
            patch_start: tuple[int, int] | None = None,
            axis: int | None = None,
    ) -> SegmentationDataIndex:
        if axis is None:
            axis = self.axis if self.axis is not None else self.rng.choice(3)

        if volume_key is None:
            volume_idx = self.rng.choice(len(self.data_files))
            volume_key = list(self.data_files.keys())[volume_idx]

        img_shape = self.image_shapes[volume_key]
        img_frames = img_shape[axis]
        num_frames = self.num_frames if self.num_frames is not None else img_frames
        img_shape_2dim = img_shape[:axis] + img_shape[axis+1:]

        if frame_start is None:
            frame_start = self.rng.choice(img_frames - num_frames + 1)
        else:
            frame_start = _resolve_index(frame_start, img_frames)
        frame_end = frame_start + num_frames
        if frame_start + num_frames > img_frames:
            raise ValueError(f"Range {frame_start}:{frame_end} is out of bounds for {img_frames} image frames")

        patch_size = self.patch_size or img_shape_2dim
        if any(p > s for p, s in zip(patch_size, img_shape_2dim)):
            raise ValueError(f"Patch size {patch_size} must be smaller than image shape {img_shape_2dim}")

        if patch_start is None:
            patch_start = (self.rng.choice(img_shape_2dim[0] - patch_size[0] + 1),
                           self.rng.choice(img_shape_2dim[1] - patch_size[1] + 1))
        patch_end = (patch_start[0] + patch_size[0], patch_start[1] + patch_size[1])

        return super().get_index(volume_key, frame_start, frame_end, patch_start, patch_end, axis)

    def __next__(self) -> SegmentationDataIndex:
        return self.get_sample()

    def __iter__(self) -> Iterator[SegmentationDataIndex]:
        return self


class SegmentationImageIndexer(SegmentationVideoIndexer):
    def _get_index(
            self,
            volume_key: str,
            frame_index: int,
            patch_start: tuple[int, int] | None = None,
            patch_end: tuple[int, int] | None = None,
            axis: int | None = None,
            frame_end: int = None,
    ) -> SegmentationDataIndex:
        if frame_end is None:
            frame_end = frame_index + 1
        return super().get_index(volume_key, frame_index, frame_end, patch_start, patch_end, axis)

    def get_frames_through_point(
            self,
            volume_key: str,
            point: tuple[int, int, int],
    ) -> tuple[tuple[SegmentationDataIndex, tuple[int, int]], ...]:
        frames = []
        for axis, frame_idx in enumerate(point):
            frame_coord = point[:axis] + point[axis+1:]
            frames.append((self.get_index(volume_key, frame_idx, axis=axis), frame_coord))
        return tuple(frames)


class SegmentationImageSampler(SegmentationVideoSampler):
    def __init__(
            self,
            data_files: dict[str, SegmentationDataIndex],
            axis: int | None = None,
            preprocessor: ImagePreprocessor | None = None,
            patch_size: tuple[int, int] | None = None,
            rng: RngInitType | None = None,
            num_frames: int = 1,
    ):
        super().__init__(
            data_files=data_files,
            axis=axis,
            preprocessor=preprocessor,
            num_frames=num_frames,
            patch_size=patch_size,
            rng=rng,
        )

    @classmethod
    def from_config(cls, config: dict | str, **kwargs):
        config = _get_config(config)
        if kwargs:
            config = config.copy()
            config.update(kwargs)
        data_files = parse_data_files(config)
        return cls(
            data_files,
            axis=config.get("axis"),
            preprocessor=config.get("preprocessor"),
            patch_size=config.get("patch_size"),
            rng=config.get("rng"),
        )

    def subset(self, volumes: str | list, **kwargs) -> Self:
        if isinstance(volumes, str):
            volumes = [volumes]
        data_files = {k: v for k, v in self.data_files.items() if k in volumes}
        params = {"axis": self.axis, "patch_size": self.patch_size, "rng": self.rng}
        params.update(kwargs)
        return type(self)(data_files, **params)

    def get_sample(
            self,
            volume_key: str | None = None,
            frame_idx: int | None = None,
            patch_start: tuple[int, int] | None = None,
            axis: int | None = None,
    ) -> SegmentationDataIndex:
        return super().get_sample(volume_key, frame_idx, patch_start, axis)


def read_single_image(
        data_indexer: SegmentationImageSampler,
        exclude_background: int | bool | None = 0,
        rng: RngInitType | None = None,
        retry_limit: int | None = 10,
        retry_counter: int = 0,
        min_size: int | float | None = None,
        max_size: int | float | None = None,
        min_distance_from_edge: int | float | None = 0.05,
):
    if rng is None:
        rng = data_indexer.rng
    else:
        rng = get_rng(rng)

    if isinstance(exclude_background, bool):
        exclude_background = 0 if exclude_background else None

    retry_kwargs = {
        "data_indexer": data_indexer,
        "exclude_background": exclude_background,
        "rng": rng,
        "retry_limit": retry_limit,
        "retry_counter": retry_counter + 1,
        "min_size": min_size,
        "max_size": max_size,
        "min_distance_from_edge": min_distance_from_edge,
    }

    # Select image
    data_idx = next(data_indexer)
    image = data_idx.image.load()
    annotation_map = data_idx.label.load()

    # Images are grayscale, convert to RGB
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Get binary masks and points
    object_ids = np.unique(annotation_map)
    if exclude_background is not None:
        object_ids = sorted(set(object_ids) - {exclude_background})
    if len(object_ids) == 0:  # No non-background objects, try again if not at retry limit
        if retry_counter == retry_limit:
            raise RuntimeError("No non-background objects found in the dataset after retry limit reached")
        else:
            return read_single_image(**retry_kwargs)

    obj_id = object_ids[rng.integers(len(object_ids))]  # pick single segment
    mask = (annotation_map == obj_id).astype(np.uint8)  # make binary mask corresponding to index ind

    # Get connected components and select the largest component
    mask_all = torch.tensor(mask).reshape(1, 1, *mask.shape).cuda()  # get_connected_components expects a batched tensor
    mask_cc = get_connected_components(mask_all)[0][0, 0, :, :].cpu().numpy()
    mask_cc_counts = np.unique(mask_cc, return_counts=True)
    main_component_id, main_component_size = mask_cc_counts[0][1], mask_cc_counts[1][1]
    mask = (mask_cc == main_component_id).astype(np.uint8)

    if min_size is not None:
        if min_size < 1:
            min_size = min_size * mask.size
        if main_component_size < min_size:
            mask = None
    if max_size is not None:
        if max_size < 1:
            max_size = max_size * mask.size
        if main_component_size > max_size:
            mask = None
    if min_distance_from_edge is not None:
        if min_distance_from_edge < 1:
            min_distance_from_edge = min_distance_from_edge * np.min(mask.shape)
        mask_com = center_of_mass(mask)
        dist_from_edge = np.min([mask_com[0], mask_com[1], mask.shape[0] - mask_com[0], mask.shape[1] - mask_com[1]])
        if dist_from_edge < min_distance_from_edge:
            mask = None

    if mask is None:
        return read_single_image(**retry_kwargs)
    else:
        return image, mask, data_idx



def get_point_prompt(mask: np.ndarray, num_points: int = 1, sample_mode="edt", rng: RngInitType | None = None):
    rng = get_rng(rng)
    coords = np.argwhere(mask)  # get all coordinates in mask

    if sample_mode.lower() == "edt":
        p = distance_transform_edt(mask)
        p = p / p.sum()
    elif sample_mode.lower() == "uniform":
        p = np.ones_like(mask)
        p = p / p.sum()
    else:
        raise ValueError(f"Invalid sample mode: {sample_mode}")
    yx = rng.choice(coords, size=num_points, p=p[np.nonzero(mask)])  # choose random point/coordinate
    return yx[:, ::-1]  # convert to (x, y) format


# def get_box_prompt(mask: np.ndarray, buffer: int = 0) -> np.ndarray:
#     coords = np.argwhere(mask > 0)  # get all coordinates in mask
#     min_y, min_x = coords.min(axis=0)
#     max_y, max_x = coords.max(axis=0)
#     return np.array([min_x - buffer, min_y - buffer, max_x + buffer, max_y + buffer])


def get_batch_with_prompts(
        data_indexer: SegmentationImageSampler,
        batch_size: int = 4,
        exclude_background: int | bool | None = 0,
        rng: RngInitType | None = None,
        retry_limit: int | None = 10,
):
    if rng is None:
        rng = data_indexer.rng
    else:
        rng = get_rng(rng)

    images = []
    masks = []
    prompt_points = []
    data_files = []
    for _ in range(batch_size):
        image, mask, dfiles = read_single_image(data_indexer, exclude_background, rng, retry_limit)
        point_prompt = get_point_prompt(mask, rng=rng)
        images.append(image)
        masks.append(mask)
        prompt_points.append(point_prompt)
        data_files.append(dfiles)

    return images, np.array(masks), np.array(prompt_points), np.ones([batch_size, 1]), data_files


def parse_data_files(config: dict) -> dict[str, SegmentationDataIndex]:
    data_files = {}
    for key, data_cfg in config["data_files"].items():
        image = HDF5DataIndex(data_cfg["image"]["path"], data_cfg["image"]["dataset"])
        label = HDF5DataIndex(data_cfg["label"]["path"], data_cfg["label"]["dataset"])
        data_files[key] = SegmentationDataIndex(image, label)
    return data_files


def postprocess_mask_logits(mask, resolution: int | None = None):
    if mask.ndim == 3:
        mask = mask[0]
    mask = (mask > 0.0).cpu().numpy().astype(float) * 255
    if resolution:
        mask = cv2.resize(mask, [resolution, resolution])
    mask[mask > 0] = 1
    mask = mask.astype(bool)
    return mask


# This class is pasted from another package of mine, may want to set that other package as a dependency and import it
class OneToOneMap(dict):
    def __init__(self, *args, inverse=None, **kwargs):
        super().__init__(*args, **kwargs)
        if inverse is not None:
            self.inverse = inverse
        else:
            self.inverse = None
            self.validate()

    def validate(self):
        if len(set(self.values())) != len(self.keys()):
            raise ValueError('One-to-one mapping constraint violated')
        if self.inverse is None:
            self.inverse = OneToOneMap(((v, k) for k, v in self.items()), inverse=self)
        if self.inverse.inverse is not self:
            raise ValueError('Inverse mapping does not point back to original')

    def __setitem__(self, key, value):
        current_val = self.pop(key, None)
        if current_val != value:
            self.inverse.pop(current_val, None)
        super().__setitem__(key, value)
        super(self.inverse.__class__, self.inverse).__setitem__(value, key)
        self.validate()
        self.inverse.validate()

    def update(self, m, /, **kwargs):
        d = dict(m, **kwargs)
        for k, v in d.items():
            self[k] = v


def iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    return 2 * intersection / (mask1.sum() + mask2.sum())


def match_gt_pred_masks(
        gt_labels: np.ndarray,
        pred_labels: np.ndarray,
        match_fn: str | Callable = 'iou',
        match_threshold: float = None,
) -> OneToOneMap:
    if isinstance(match_fn, str):
        if match_fn == 'iou':
            match_fn = iou
        elif match_fn == 'dice':
            match_fn = dice
        else:
            raise ValueError(f"Invalid match mode: {match_fn}")

    remaining_gt_ids = set(np.unique(gt_labels))
    remaining_pred_ids = set(np.unique(pred_labels))

    remaining_gt_ids.discard(0)
    remaining_pred_ids.discard(0)

    gt_pred_map = OneToOneMap()
    while remaining_gt_ids and remaining_pred_ids:
        gt_id = remaining_gt_ids.pop()
        gt_mask = (gt_labels == gt_id)
        candidate_pred_ids = set(np.unique(pred_labels[gt_mask])).intersection(remaining_pred_ids)
        best_score = None
        best_id = None
        for pred_id in candidate_pred_ids:
            pred_mask = (pred_labels == pred_id)
            pred_score = match_fn(gt_mask, pred_mask)
            if match_threshold and pred_score < match_threshold:
                continue
            alt_gt_ids = set(np.unique(gt_labels[pred_mask])).intersection(remaining_gt_ids)
            alt_scores = [match_fn(gt_labels == alt_gt_id, pred_mask) for alt_gt_id in alt_gt_ids]
            if alt_scores and max(alt_scores) > pred_score:
                continue
            if best_score is None or pred_score > best_score:
                best_score = pred_score
                best_id = pred_id
        if best_id is not None:
            pred_mask = (pred_labels == best_id)
            alt_gt_ids = set(np.unique(gt_labels[pred_mask])).intersection(remaining_gt_ids)
            alt_scores = [match_fn(gt_labels == alt_gt_id, pred_mask) for alt_gt_id in alt_gt_ids]
            if alt_scores and max(alt_scores) > best_score:
                remaining_gt_ids.add(gt_id)
            else:
                gt_pred_map[gt_id] = best_id
                remaining_pred_ids.remove(best_id)

    return gt_pred_map
