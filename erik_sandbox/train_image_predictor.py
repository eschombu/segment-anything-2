""" Fine-tune a video predictor model on microwasp volume EM data.

Reference: https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code/blob/13d1bdf523cc0d7ce66b9b335e9d21a0c2f74672/TRAIN_multi_image_batch.py
"""

import argparse
import os
import pdb
import sys
import traceback
import yaml
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

RngInitType = Union[int, str, np.random.Generator]

this_dir = Path(__file__).resolve().parent
checkpoint_dir = this_dir.parent / "checkpoints"
assert checkpoint_dir.is_dir(), f"Checkpoints directory not found: {checkpoint_dir}"


def _load_h5_dataset(file, dataset: Union[str, Tuple[str, ...]]):
    if isinstance(dataset, str):
        dset = file[dataset]
    else:
        dset = file
        for name in dataset:
            dset = dset[name]
    return dset


def _get_rng(rng: Optional[RngInitType] = None) -> np.random.Generator:
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
    dataset: Union[str, Tuple[str,...]]
    slice: Optional[tuple] = None

    def load(self):
        with h5py.File(self.path, 'r') as f:
            dset = _load_h5_dataset(f, self.dataset)
            if self.slice is not None:
                return dset[*self.slice]
            else:
                return dset[...]


@dataclass
class DataFiles:
    image: HDF5DataIndex
    label: HDF5DataIndex


class HDF5ImageIndexer:
    def __init__(
            self,
            data_files: Dict[str, DataFiles],
            patch_size: Tuple[int, int] = None,
            axis: Optional[int] = None,
            rng: Optional[RngInitType] = None,
    ):
        self.data_files = data_files
        self.patch_size = patch_size
        self.axis = axis
        self.rng = _get_rng(rng)

        self.image_shapes: Dict[str, Tuple[int, int, int]] = {}
        self._build_indexer_data()

    @classmethod
    def from_config(cls, config: dict):
        data_files = parse_data_files(config)
        return cls(
            data_files,
            patch_size=config.get("patch_size"),
            axis=config.get("axis"),
            rng=config.get("rng"),
        )

    def rand_int(self, *args, dtype=int, **kwargs):
        return self.rng.integers(*args, dtype=dtype, **kwargs)

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

    def get_image_index(
            self,
            img_idx: Optional[int] = None,
            start: Optional[Tuple[int, int]] = None,
            axis: Optional[int] = None,
            volume_key: Optional[int] = None,
    ) -> Tuple[HDF5DataIndex, HDF5DataIndex]:

        if axis is None:
            axis = self.axis if self.axis is not None else self.rand_int(3)

        if volume_key is None:
            volume_idx = self.rand_int(len(self.data_files))
            volume_key = list(self.data_files.keys())[volume_idx]

        data = self.data_files[volume_key]
        img_shape = self.image_shapes[volume_key]

        if img_idx is None:
            img_idx = self.rand_int(img_shape[axis])

        img_shape_2dim = img_shape[:axis] + img_shape[axis+1:]
        if self.patch_size[0] > img_shape_2dim[0] or self.patch_size[1] > img_shape_2dim[1]:
            raise ValueError("Patch size must be smaller than image shape")
        if start is None:
            start = [self.rand_int(img_shape_2dim[0] - self.patch_size[0] + 1),
                     self.rand_int(img_shape_2dim[1] - self.patch_size[1] + 1)]
        stop = [start[0] + self.patch_size[0], start[1] + self.patch_size[1]]

        if axis == 0:
            slice_ = (img_idx, slice(start[0], stop[0]), slice(start[1], stop[1]))
        elif axis == 1:
            slice_ = (slice(start[0], stop[0]), img_idx, slice(start[1], stop[1]))
        else:
            slice_ = (slice(start[0], stop[0]), slice(start[1], stop[1]), img_idx)

        return (HDF5DataIndex(data.image.path, data.image.dataset, slice_),
                HDF5DataIndex(data.label.path, data.label.dataset, slice_))

    def __next__(self) -> Tuple[HDF5DataIndex, HDF5DataIndex]:
        return self.get_image_index()

    def __iter__(self) -> Iterator[Tuple[HDF5DataIndex, HDF5DataIndex]]:
        return self


def read_single(
        data_indexer: HDF5ImageIndexer,
        exclude_background: Optional[Union[int, bool]] = 0,
        rng: Optional[RngInitType] = None,
        retry_limit: Optional[int] = 10,
        retry_counter: int = 0,
):
    if rng is None:
        rng = data_indexer.rng
    else:
        rng = _get_rng(rng)

    if isinstance(exclude_background, bool):
        exclude_background = 0 if exclude_background else None

    # Select image
    img_idx, ann_idx = next(data_indexer)
    image = img_idx.load()
    annotation_map = ann_idx.load()

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
            return read_single(data_indexer, exclude_background, rng, retry_limit, (retry_counter + 1))

    obj_id = object_ids[rng.integers(len(object_ids))]  # pick single segment
    # TODO: Split apart object, selecting largest connected component. Discard and re-select a new object if it is too
    #  small or too near the edge of the image

    mask = (annotation_map == obj_id).astype(np.uint8)  # make binary mask corresponding to index ind
    return image, mask, DataFiles(img_idx, ann_idx)


def get_point_prompt(mask: NDArray, num_points: int = 1, sample_mode="edt", rng: Optional[RngInitType] = None):
    rng = _get_rng(rng)
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


def get_box_prompt(mask: NDArray, buffer: int = 0) -> NDArray:
    coords = np.argwhere(mask > 0)  # get all coordinates in mask
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    return np.array([min_x - buffer, min_y - buffer, max_x + buffer, max_y + buffer])


def read_batch(
        data_indexer: HDF5ImageIndexer,
        batch_size: int = 4,
        exclude_background: Optional[Union[int, bool]] = 0,
        rng: Optional[RngInitType] = None,
        retry_limit: Optional[int] = 10,
):
    if rng is None:
        rng = data_indexer.rng
    else:
        rng = _get_rng(rng)

    images = []
    masks = []
    prompt_points = []
    for i in range(batch_size):
        image, mask = read_single(data_indexer, exclude_background, rng, retry_limit)
        point_prompt = get_point_prompt(mask, rng=rng)
        images.append(image)
        masks.append(mask)
        prompt_points.append(point_prompt)

    return images, np.array(masks), np.array(prompt_points), np.ones([batch_size, 1])


def parse_data_files(config: dict) -> Dict[str, DataFiles]:
    data_files = {}
    for key, data_cfg in config["data_files"].items():
        image = HDF5DataIndex(data_cfg["image"]["path"], data_cfg["image"]["dataset"])
        label = HDF5DataIndex(data_cfg["label"]["path"], data_cfg["label"]["dataset"])
        data_files[key] = DataFiles(image, label)
    return data_files


@dataclass
class ModelContainer:
    predictor: SAM2ImagePredictor
    optimizer: torch.optim.Optimizer
    scaler: torch.cuda.amp.GradScaler


def prepare_model(config: dict) -> ModelContainer:
    sam2_ckpt_path = checkpoint_dir / config["sam2_checkpoint"]
    assert sam2_ckpt_path.exists(), f"Checkpoint not found: {sam2_ckpt_path}"
    sam2_ckpt_path = str(sam2_ckpt_path)
    sam2_model = build_sam2(config["model_cfg"], sam2_ckpt_path, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # Set training parameters
    predictor.model.sam_mask_decoder.train(True)  # enable training of mask decoder
    predictor.model.sam_prompt_encoder.train(True)  # enable training of prompt encoder
    # *** NOTE: For the following line to work you need to scan the SAM2 code for "no_grad" and remove them all ***
    predictor.model.image_encoder.train(True)  # enable training of image encoder
    optimizer = torch.optim.AdamW(
        params=predictor.model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler()  # mixed precision
    return ModelContainer(predictor, optimizer, scaler)


def train(config: dict, save_model: bool = True, display: bool = False):
    start_time = datetime.now()
    ts = start_time.strftime("%Y%m%dT%H%M%S")
    output_path = f"tuned_sam_model_{ts}.pt"

    # Load dataset
    data_indexer = HDF5ImageIndexer.from_config(config["train_data_config"])

    # Load model
    model = prepare_model(config)

    # Loop through batches
    print("Starting training...")
    if save_model:
        print(f"Saving model to {output_path}")
    iou_cache = []
    pbar = tqdm(range(config["num_batches"]), desc="Accuracy(IOU) = _")
    for batch_idx in pbar:
        with torch.cuda.amp.autocast():  # cast to mix precision
            image, mask, input_point, input_label = read_batch(data_indexer, batch_size=config["batch_size"])
            if mask.shape[0] == 0:
                continue # ignore empty batches

            model.predictor.set_image_batch(image) # apply SAM image encoder to the image
            # model.predictor.get_image_embedding()

            # prompt encoding
            mask_input, unnorm_coords, labels, unnorm_box = model.predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = model.predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None)

            # mask decoder
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in model.predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = model.predictor.model.sam_mask_decoder(
                image_embeddings=model.predictor._features["image_embed"],
                image_pe=model.predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            # Upscale the masks to the original image resolution
            pred_masks = model.predictor._transforms.postprocess_masks(low_res_masks, model.predictor._orig_hw[-1])

            # Segmentation loss calculation
            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            pred_mask = torch.sigmoid(pred_masks[:, 0])  # Turn logit map to probability map
            seg_loss = (-gt_mask * torch.log(pred_mask + 0.00001) +
                        -1 * (1 - gt_mask) * torch.log((1 - pred_mask) + 0.00001)
                        ).mean()  # cross entropy loss

            # Score loss calculation (intersection over union) IOU
            intersection = (gt_mask * (pred_mask > 0.5)).sum(1).sum(1)
            union = (gt_mask.sum(1).sum(1) + (pred_mask > 0.5).sum(1).sum(1) - intersection)
            iou = intersection / union
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + (score_loss * 0.05)  # mix losses

            # apply backpropagation
            model.predictor.model.zero_grad()  # empty gradient
            model.scaler.scale(loss).backward()
            model.scaler.step(model.optimizer)
            model.scaler.update()  # Mixed precision

            if save_model and batch_idx % config["checkpoint_interval"] == 0:
                torch.save(model.predictor.model.state_dict(), output_path)  # save checkpoint

            # Display results
            if len(iou_cache) == 100:
                iou_cache.pop(0)
            iou_cache.append(iou.cpu().detach().numpy())
            mean_iou = np.mean(iou_cache)
            pbar.set_description(f"Accuracy(IOU) = {mean_iou}")

            if display or (config["display_interval"] and batch_idx % config["display_interval"] == 0):
                i = 0
                img = image[i]
                m = mask[i]
                pm = pred_mask[i, 0].cpu().detach().numpy()
                pt = input_point[i][0]
                arrow_len = len(img) // 10
                arrow_side = np.sqrt(0.5 * arrow_len ** 2)
                arrow = tuple(pt - np.ones(2) * arrow_side) + tuple(pt)
                print(arrow)
                plt.figure(figsize=(12, 12))
                plt.subplot(2, 2, 1)
                plt.imshow(img)
                plt.imshow(m, cmap="Reds", alpha=0.3)
                plt.arrow(pt[0] - arrow_side, pt[1] - arrow_side, arrow_side, arrow_side, color="g",
                          head_width=(arrow_len / 2), length_includes_head=True)
                plt.title("Ground Truth")
                plt.subplot(2, 2, 3)
                plt.imshow(m, cmap="gray")
                plt.subplot(2, 2, 2)
                plt.imshow(img)
                plt.imshow(pm, cmap="Blues", alpha=0.3)
                plt.arrow(pt[0] - arrow_side, pt[1] - arrow_side, arrow_side, arrow_side, color="g",
                          head_width=(arrow_len / 2), length_includes_head=True)
                plt.title("Prediction")
                plt.subplot(2, 2, 4)
                plt.imshow(pm, cmap="gray")
                plt.show()
                r = input("Press Enter to continue, 'q' to quit, or 'd' to drop into debugger: ")
                if r.strip().lower() == "q":
                    sys.exit(1)
                elif r.strip().lower() == "d":
                    pdb.set_trace()

    if save_model:
        torch.save(predictor.model.state_dict(), output_path)  # save final model


def display_example(image, mask, pred_mask=None, prompt=None):
    def plot_prompt(p):
        if p is not None:
            if p.size == 2:
                arrow_len = image.shape[0] / 10
                arrow_side = np.sqrt(0.5 * arrow_len ** 2)
                plt.arrow(prompt[0] - arrow_side, p[1] - arrow_side, arrow_side, arrow_side,
                          color="g", head_width=(arrow_len / 2), length_includes_head=True)
            elif p.ndim == 1 and p.size == 4:
                rect = Rectangle((p[0], p[1]), p[2] - p[0], p[3] - p[1],
                                 edgecolor="g", facecolor="none")
                plt.gca().add_patch(rect)
            else:
                for pp in p:
                    plot_prompt(pp)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Ground truth
    plt.sca(axs[0, 0])
    plt.imshow(image)
    plt.imshow(mask, cmap="Reds", alpha=0.4)
    plot_prompt(prompt)
    plt.title("Ground Truth")

    # Show the mask as a binary image
    plt.sca(axs[1, 0])
    plt.imshow(mask, cmap="gray")

    # Predictions
    if pred_mask is not None:
        plt.sca(axs[0, 1])
        plt.imshow(image)
        plt.imshow(pred_mask, cmap="Reds", alpha=0.3)
        plot_prompt(prompt)
        plt.title("Prediction")

        # Show the predicted mask probabilities, and thresholded
        plt.sca(axs[1, 1])
        thresh_mask = (pred_mask > 0.5).astype(float)
        thresh_mask[thresh_mask == 0] = np.nan
        plt.imshow(pred_mask, cmap="gray")
        plt.imshow(thresh_mask, cmap="Reds", alpha=0.4)

    plt.show()


def main(args):
    config_path = os.path.join(this_dir, "train_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    train(config, save_model=(not args.no_save), display=args.display)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", "-d", action="store_true", help="display the example images and masks")
    parser.add_argument("--no-save", action="store_true", help="do not save the model")
    parser.add_argument("--debug", action="store_true", help="drop into debugger on exception")
    args = parser.parse_args()

    try:
        main(args)
    except (KeyboardInterrupt, pdb.bdb.BdbQuit):
        sys.exit(1)
    except Exception as e:
        if args.debug:
            traceback.print_exc()
            pdb.post_mortem()
        else:
            raise e
