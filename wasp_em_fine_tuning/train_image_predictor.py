""" Fine-tune a video predictor model on microwasp volume EM data.

Reference: https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code/blob/13d1bdf523cc0d7ce66b9b335e9d21a0c2f74672/TRAIN_multi_image_batch.py
"""

import argparse
# import logging
import os
import pdb
import sys
import traceback
import yaml
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.training import enable_training, disable_training

from data_utils import SegmentationImageSampler, get_batch_with_prompts

RngInitType = int | str | np.random.Generator

this_dir = Path(__file__).resolve().parent
checkpoint_dir = this_dir.parent / "checkpoints"
assert checkpoint_dir.is_dir(), f"Checkpoints directory not found: {checkpoint_dir}"

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler(sys.stdout))


@dataclass
class ModelSpec:
    size: str
    checkpoint: str
    config: str

    @staticmethod
    def _get_model_size_from_checkpoint(model_ckpt: str) -> str | None:
        size_options = ["tiny", "small", "base", "large"]
        ckpt_split = os.path.splitext(model_ckpt)[0].split("_")
        for size in size_options:
            if size in ckpt_split:
                return size
        return None

    @classmethod
    def from_config(cls, config: dict, allow_model_size_discrepancy=False) -> Self:
        model_size = config.get("sam2_model_size")
        model_ckpt = config.get("model_checkpoint")
        model_cfg = config.get("model_cfg")
        if model_ckpt is None:
            if model_size is None:
                raise ValueError("Either 'sam2_model_size' or 'model_checkpoint' must be provided in the config")
            model_ckpt = f"sam2_hiera_{model_size}.pt"
        if model_cfg is None:
            if model_size is None:
                raise ValueError("Either 'sam2_model_size' or 'model_cfg' must be provided in the config")
            if model_size == "base_plus":
                model_cfg = f"sam2_hiera_b+.yaml"
            else:
                model_cfg = f"sam2_hiera_{model_size[0]}.yaml"

        ckpt_model_size = cls._get_model_size_from_checkpoint(model_ckpt)
        if model_size is None:
            if ckpt_model_size is None:
                model_size = "size-missing"
            else:
                model_size = ckpt_model_size
        elif ckpt_model_size is not None and model_size != ckpt_model_size:
            if not allow_model_size_discrepancy:
                raise ValueError(
                    f"specified model size '{model_size}' and checkpoint model size '{ckpt_model_size}' do not match")
            else:
                warn(f"specified model size '{model_size}' and checkpoint model size '{ckpt_model_size}' do not match")

        return cls(model_size, model_ckpt, model_cfg)


@dataclass
class ModelContainer:
    spec: ModelSpec
    predictor: SAM2ImagePredictor
    optimizer: torch.optim.Optimizer
    scaler: torch.cuda.amp.GradScaler


def prepare_model(config: dict) -> ModelContainer:
    model_spec = ModelSpec.from_config(config)
    sam2_ckpt_path = checkpoint_dir / model_spec.checkpoint
    assert sam2_ckpt_path.exists(), f"Checkpoint not found: {sam2_ckpt_path}"
    sam2_ckpt_path = str(sam2_ckpt_path)
    sam2_model = build_sam2(model_spec.config, sam2_ckpt_path, device="cuda")
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
    return ModelContainer(model_spec, predictor, optimizer, scaler)


def train(config: dict, save_model: bool = True, display_interval: int = 0):
    start_time = datetime.now()
    ts = start_time.strftime("%Y%m%dT%H%M%S")
    enable_training()

    # Load dataset
    image_sampler = SegmentationImageSampler.from_config(config["train_data_config"])

    # Load model
    model = prepare_model(config)
    save_path_template = f"tuned_sam2_{model.spec.size}_{ts}.{{batch}}.pt"

    # Loop through batches
    print(f"Starting training at {start_time.strftime('%Y/%m/%d %H:%M:%S %Z')}...")
    if save_model:
        print(f"Model will be saved at {save_path_template}")
    iou_cache = []
    pbar = tqdm(range(config["num_batches"]), desc="Accuracy(IOU) = _")
    for batch_idx in pbar:
        with torch.cuda.amp.autocast():  # cast to mix precision
            image, mask, input_point, input_label, data_files = \
                get_batch_with_prompts(image_sampler, batch_size=config["batch_size"])
            if mask.shape[0] == 0:
                continue  # ignore empty batches

            model.predictor.set_image_batch(image)  # apply SAM image encoder to the image
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

            if save_model and batch_idx > 0 and (batch_idx + 1) % config["checkpoint_interval"] == 0:
                ckpt_step = batch_idx + 1
                save_path = save_path_template.format(batch=ckpt_step)
                print(f"Saving model checkpoint to {save_path} at {datetime.now().strftime('%Y/%m/%d %H:%M:%S %Z')}")
                torch.save({"model": model.predictor.model.state_dict()}, save_path)  # save checkpoint

            # Display results
            if len(iou_cache) == 100:
                iou_cache.pop(0)
            iou_cache.append(iou.cpu().detach().numpy())
            mean_iou = np.mean(iou_cache)
            pbar.set_description(f"Accuracy(IOU) = {mean_iou}")

            if (display_interval > 0) and (batch_idx % config["display_interval"] == 0):
                display_idx = 0
                img = image[display_idx]
                m = mask[display_idx]
                pm = pred_mask[display_idx].cpu().detach().numpy()
                pt = input_point[display_idx][0]
                display_example(img, m, pm, pt)

                r = input("Press Enter to continue, 'q' to quit, or 'd' to drop into debugger: ")
                if r.strip().lower() == "q":
                    sys.exit(1)
                elif r.strip().lower() == "d":
                    pdb.set_trace()

    print(f"Training complete. Total time: {datetime.now() - start_time}")
    disable_training()
    if save_model:
        save_path = save_path_template.format(batch=config["num_batches"])
        print(f"Saving model checkpoint to {save_path} at {datetime.now().strftime('%Y/%m/%d %H:%M:%S %Z')}")
        torch.save({"model": model.predictor.model.state_dict()}, save_path)  # save final model


def display_example(image, mask, pred_mask=None, prompt=None):
    def plot_prompt(p):
        if p is not None:
            if p.size == 2:
                arrow_len = image.shape[0] / 10
                arrow_side = np.sqrt(0.5 * arrow_len ** 2)
                if p[0] < arrow_side:
                    arrow_xstart = p[0] + arrow_side
                    arrow_xside = -arrow_side
                else:
                    arrow_xstart = p[0] - arrow_side
                    arrow_xside = arrow_side
                if p[1] < arrow_side:
                    arrow_ystart = p[1] + arrow_side
                    arrow_yside = -arrow_side
                else:
                    arrow_ystart = p[1] - arrow_side
                    arrow_yside = arrow_side
                arrow_coords = (arrow_xstart, arrow_ystart, arrow_xside, arrow_yside)
                plt.arrow(*arrow_coords, color="g", head_width=(arrow_len / 2), head_length=(arrow_len / 2),
                          length_includes_head=True)
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
    train(config, save_model=(not args.no_save), display_interval=args.display_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-interval", "-d", type=int, default=0,
                        help="Display first example and prompt immediately and after each of this number of batches")
    parser.add_argument("--no-save", action="store_true", help="Do not save the model")
    parser.add_argument("--debug", action="store_true", help="Drop into debugger on exception")
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
