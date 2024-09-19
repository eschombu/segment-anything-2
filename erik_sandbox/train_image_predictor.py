""" Fine-tune a video predictor model on microwasp volume EM data.

Reference: https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code/blob/13d1bdf523cc0d7ce66b9b335e9d21a0c2f74672/TRAIN_multi_image_batch.py
"""

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import cv2

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _load_h5_dataset(file, dataset: Union[str, Tuple[str, ...]]):
    if isinstance(dataset, str):
        dset = file[dataset]
    else:
        dset = file
        for name in dataset:
            dset = dset[name]
    return dset


def _get_rng(rng: Optional[Union[int, np.random.Generator]] = None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        rng = rng
    else:
        rng = np.random.default_rng(rng)
    return rng


@dataclass
class HDF5DataIndex:
    filename: str
    dataset: Union[str, Tuple[str,...]]
    slice: tuple

    def load(self):
        with h5py.File(self.filename, 'r') as f:
            return _load_h5_dataset(f, self.dataset)[*self.slice]


class HDF5ImageIndexer:
    def __init__(
            self,
            image_files: Sequence[str],
            label_files: Sequence[str],
            image_dataset: Union[str, Tuple[str,...]],
            label_dataset: Union[str, Tuple[str,...]],
            patch_size: Tuple[int, int] = None,
            axis: Optional[int] = None,
            rng: Optional[Union[int, np.random.Generator]] = None,
    ):
        if len(image_files) != len(label_files):
            raise ValueError("Number of image files must match number of label files")

        self.image_files = image_files
        self.label_files = label_files
        self.image_dataset = image_dataset
        self.label_dataset = label_dataset
        self.patch_size = patch_size
        self.axis = axis
        self.rng = _get_rng(rng)

        self.image_shapes: List[Tuple[int, int, int]] = []
        self._build_indexer_data()

    def _build_indexer_data(self):
        for image_file, label_file in zip(self.image_files, self.label_files):
            with h5py.File(image_file, 'r') as f:
                dset = _load_h5_dataset(f, self.image_dataset)
                image_shape = dset.shape
            with h5py.File(image_file, 'r') as f:
                dset = _load_h5_dataset(f, self.label_dataset)
                label_shape = dset.shape
            assert image_shape == label_shape, \
                f"image and label shapes do not match ({image_shape} vs. {label_shape})" \
                f" for {image_file} and {label_file}"

            # Add slices for image patches or whole images, along each axis
            self.image_shapes.append(image_shape)

    def get_image_index(
            self,
            img_idx: Optional[int] = None,
            start: Optional[Tuple[int, int]] = None,
            axis: Optional[int] = None,
            file_idx: Optional[int] = None,
    ) -> Tuple[HDF5DataIndex, HDF5DataIndex]:
        if axis is None:
            axis = self.axis if self.axis is not None else self.rng.integers(3)
        if file_idx is None:
            file_idx = self.rng.integers(len(self.image_files))
        img_shape = self.image_shapes[file_idx]
        if img_idx is None:
            img_idx = self.rng.integers(img_shape[axis])
        img_shape_2dim = img_shape[:axis] + img_shape[axis+1:]
        if self.patch_size[0] > img_shape_2dim[0] or self.patch_size[1] > img_shape_2dim[1]:
            raise ValueError("Patch size must be smaller than image shape")
        if start is None:
            start = [self.rng.integers(img_shape_2dim[0] - self.patch_size[0] + 1),
                     self.rng.integers(img_shape_2dim[1] - self.patch_size[1] + 1)]
        stop = [start[0] + self.patch_size[0], start[1] + self.patch_size[1]]
        if axis == 0:
            slice_ = (img_idx, slice(start[0], stop[0]), slice(start[1], stop[1]))
        elif axis == 1:
            slice_ = (slice(start[0], stop[0]), img_idx, slice(start[1], stop[1]))
        else:
            slice_ = (slice(start[0], stop[0]), slice(start[1], stop[1]), img_idx)
        return (HDF5DataIndex(self.image_files[file_idx], self.image_dataset, slice_),
                HDF5DataIndex(self.label_files[file_idx], self.label_dataset, slice_))

    def __next__(self) -> Tuple[HDF5DataIndex, HDF5DataIndex]:
        return self.get_image_index()

    def __iter__(self) -> Iterator[Tuple[HDF5DataIndex, HDF5DataIndex]]:
        return self



def read_single(
        data_indexer: HDF5ImageIndexer,
        exclude_background: Optional[Union[int, bool]] = 0,
        rng: Optional[Union[int, np.random.Generator]] = None,
        retry_limit: Optional[int] = 10,
        retry_counter: int = 0,
):
    rng = _get_rng(rng)
    if isinstance(exclude_background, bool):
        exclude_background = 0 if exclude_background else None

    # Select image
    img_idx, ann_idx = next(data_indexer)
    image = img_idx.load()
    annotation_map = ann_idx.load()

    # Get binary masks and points
    object_ids = np.unique(annotation_map)
    if exclude_background is not None:
        object_ids = sorted(set(object_ids) - {exclude_background})
    if len(object_ids) == 0:  # No non-background objects, try again if not at retry limit
        if retry_counter == retry_limit:
            raise RuntimeError("No non-background objects found in the dataset after retry limit reached")
        else:
            return read_single(data_indexer, exclude_background, rng, retry_limit, (retry_counter + 1))

    obj_id = object_ids[rng.integers(len(object_ids))]  # pick single segment
    mask = (annotation_map == obj_id).astype(np.uint8)  # make binary mask corresponding to index ind
    coords = np.argwhere(mask > 0)  # get all coordinates in mask

    # TODO: bias coordinates away from the mask boundary???
    yx = np.array(coords[rng.integers(len(coords))])  # choose random point/coordinate

    return image, mask, [[yx[1], yx[0]]]


def read_batch(
        data_indexer: HDF5ImageIndexer,
        batch_size: int = 4,
        exclude_background: Optional[Union[int, bool]] = 0,
        rng: Optional[Union[int, np.random.Generator]] = None,
        retry_limit: Optional[int] = 10,
):
    rng = _get_rng(rng)
    images = []
    masks = []
    input_points = []
    for i in range(batch_size):
        image, mask, input_point = read_single(data_indexer, exclude_background, rng, retry_limit)
        images.append(image)
        masks.append(mask)
        input_points.append(input_point)

    return np.array(images), np.array(masks), np.array(input_points), np.ones([batch_size, 1])


def train():
    # Load dataset
    data = get_data("path/to/images", "path/to/labels")

    # Load model
    sam2_checkpoint = "sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # Set training parameters
    predictor.model.sam_mask_decoder.train(True)  # enable training of mask decoder
    predictor.model.sam_prompt_encoder.train(True)  # enable training of prompt encoder
    # *** NOTE: For the following line to work you need to scan the SAM2 code for "no_grad" and remove them all ***
    predictor.model.image_encoder.train(True)  # enable training of image encoder
    optimizer=torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
    scaler = torch.cuda.amp.GradScaler()  # mixed precision

    # Loop through batches
    for itr in range(100000):
        with torch.cuda.amp.autocast():  # cast to mix precision
            image,mask, input_point, input_label = read_batch(data, batch_size=4)
            if mask.shape[0] == 0:
                continue # ignore empty batches

            predictor.set_image_batch(image) # apply SAM image encoder to the image
            # predictor.get_image_embedding()

            # prompt encoding
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None)

            # mask decoder
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"],
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            # Upscale the masks to the original image resolution
            pred_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

            # Segmentation loss calculation
            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            pred_mask = torch.sigmoid(pred_masks[:, 0])  # Turn logit map to probability map
            seg_loss = (-gt_mask * torch.log(pred_mask + 0.00001) +
                        -1 * (1 - gt_mask) * torch.log((1 - pred_mask) + 0.00001)
                        ).mean()  # cross entropy loss

            # Score loss calculation (intersection over union) IOU
            inter = (gt_mask * (pred_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (pred_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss=seg_loss+score_loss*0.05  # mix losses

            # apply backpropagation
            predictor.model.zero_grad()  # empty gradient
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  # Mixed precision

            if itr % 1000 == 0:
                torch.save(predictor.model.state_dict(), "model.torch")  # save checkpoint

            # Display results
            if itr == 0:
                mean_iou = 0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print(f"step {itr}, Accuracy(IOU) = {mean_iou}")
