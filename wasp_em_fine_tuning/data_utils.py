import os
import yaml
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Iterator, Self, Sequence

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import center_of_mass, distance_transform_edt

from sam2.utils.misc import get_connected_components

RngInitType = int | str | np.random.Generator


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


@dataclass
class HDF5DataIndex:
    path: str
    dataset: str | tuple[str, ...]
    frame_axis: int | None = None
    slice: tuple | None = None
    shape: tuple | None = None

    def load(self) -> NDArray:
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
    ):
        self.data_files = data_files
        self.axis = axis
        self.image_shapes: dict[str, tuple[int, int, int]] = {}
        self._build_indexer_data()

    @classmethod
    def from_config(cls, config: dict | str, **kwargs):
        if isinstance(config, str):
            if os.path.exists(config):
                with open(config) as f:
                    config = yaml.safe_load(f)
            else:
                config = yaml.safe_load(config)
        if kwargs:
            config = config.copy()
            config.update(kwargs)
        data_files = parse_data_files(config)
        return cls(
            data_files,
            axis=config.get("axis"),
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
    def get_index(self, *args, **kwargs) -> SegmentationDataIndex:
        pass


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
    def get_index(
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
            num_frames: int | None = None,
            patch_size: tuple[int, int] = None,
            rng: RngInitType | None = None,
    ):
        super().__init__(data_files=data_files, axis=axis)
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.rng = get_rng(rng)

    @classmethod
    def from_config(cls, config: dict, **kwargs):
        if kwargs:
            config = config.copy()
            config.update(kwargs)
        data_files = parse_data_files(config)
        return cls(
            data_files,
            axis=config.get("axis"),
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
    def get_index(
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
            patch_size: tuple[int, int] | None = None,
            rng: RngInitType | None = None,
            num_frames: int = 1,
    ):
        super().__init__(data_files=data_files, axis=axis, num_frames=num_frames, patch_size=patch_size, rng=rng)

    @classmethod
    def from_config(cls, config: dict, **kwargs):
        if kwargs:
            config = config.copy()
            config.update(kwargs)
        data_files = parse_data_files(config)
        return cls(
            data_files,
            axis=config.get("axis"),
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



def get_point_prompt(mask: NDArray, num_points: int = 1, sample_mode="edt", rng: RngInitType | None = None):
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


# def get_box_prompt(mask: NDArray, buffer: int = 0) -> NDArray:
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
