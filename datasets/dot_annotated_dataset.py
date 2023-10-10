import itertools
import pandas as pd

from pathlib import Path

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from datasets.patched_dataset import PatchedMultiImageDataset, PatchedImageDataset
from methods.detection.target_builder import DetectionTargetBuilder
from methods.density.target_builder import DensityTargetBuilder
from methods.segmentation.target_builder import SegmentationTargetBuilder


class DotAnnotatedDataset(PatchedMultiImageDataset):
    """ Dot-annotated dataset that provides per-patch iteration of bunch of big image files,
        implemented as a concatenation of single-file datasets. """

    def __init__(
        self,
        root='data/sticky-trap-insects',
        split='all',
        patch_size=640,
        overlap=0,
        random_offset=None,
        target=None,
        target_params={},
        transforms=None,
        as_gray=False,
        in_memory=True,
    ):
        assert target in (None, 'segmentation', 'detection', 'density'), f'Unsupported target type: {target}'
        assert split in ('train-half1', 'train-half2', 'test', 'all'), \
            "split must be one of ('train-half1', 'train-half2', 'test', 'all')"
        
        self.root = Path(root)
        
        self.patch_size = patch_size
        self.overlap = overlap
        self.random_offset = random_offset if random_offset is not None else patch_size // 2
        
        self.transforms = transforms
        self.as_gray = as_gray

        self.target = target
        self.target_params = target_params
        
        if target == 'segmentation':
            target_builder = SegmentationTargetBuilder
        elif target == 'detection':
            target_builder = DetectionTargetBuilder
        elif target == 'density':
            target_builder = DensityTargetBuilder
        
        target_builder = target_builder(**target_params) if target else None

        # TODO check when test (specially for pest24)
        self.split = split
        if split == 'all':
            annot_path = self.root / 'annotations.csv'
            imgs_path = self.root / 'fullFrames'
        else:
            split_dir = split.split('-')[0]
            annot_path = self.root / split_dir / 'annotations.csv'
            imgs_path = self.root / split_dir / 'fullFrames'

        # load pandas dataframe containing dot annotations
        all_annot = pd.read_csv(annot_path, index_col=0)
        if not 'class' in all_annot.columns:
            all_annot['class'] = 0
        num_classes = all_annot['class'].nunique()
        
        # get list of images in the given split
        self.image_paths = sorted((imgs_path).glob('*.[pjbpjJ][npmgpP][ggpmeG]*'))
        assert len(self.image_paths) > 0, "No images found"

        splits = ('all',)
        if self.split == 'train-half1':
            splits = ('left', 'right')
        elif self.split == 'train-half2':
            splits = ('right', 'left')
        elif self.split == 'test':
            pass
        splits = itertools.cycle(splits)

        stride = patch_size - self.overlap if patch_size else None
        kwargs = dict(
            patch_size=patch_size,
            stride=stride,
            random_offset=self.random_offset,
            annotations=all_annot,
            target_builder=target_builder,
            transforms=transforms,
            num_classes=num_classes,
            as_gray=as_gray,
            in_memory=in_memory,
        )
        image_ids = [i.name for i in self.image_paths]
        datasets = [
            PatchedImageDataset(image_path, split=s, image_id=i, **kwargs)
            for image_path, s, i in zip(self.image_paths, splits, image_ids)
        ]

        super().__init__(datasets)
        
    
    def __str__(self):
        s = f'{self.__class__.__name__}: ' \
            f'{self.split} split, ' \
            f'{len(self.datasets)} images, ' \
            f'{len(self)} patches ({self.patch_size}x{self.patch_size})'
        return s



# Test code
def main():
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, Normalize, ToTensor
    from methods.detection.transforms import Compose as DetCompose, RandomHorizontalFlip as DetRandomHorizontalFlip, RandomVerticalFlip as DetRandomVerticalFlip, ToTensor as DetToTensor
    from methods.detection.utils import collate_fn as det_collate_fn, build_coco_compliant_batch
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    from torchvision.transforms.functional import to_pil_image
    import matplotlib

    phase = "train"     # train, validation
    target = "detection"    # detection, density, segmentation
    debug_num_sample = 8
    patch_size = 480
    mask = False

    # Setting transforms
    if target == "detection":
        if phase == "train":
            transforms = DetCompose([
                DetRandomHorizontalFlip(mask=mask),
                DetRandomVerticalFlip(mask=mask),
                DetToTensor(),
            ])
        elif phase == "validation":
            transforms = DetCompose([
                DetToTensor(),
            ])
    elif target == "density" or target == "segmentation":
        if phase == "train":
            transforms = Compose([
                ToTensor(),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
            ])
        elif phase == "validation":
            transforms = Compose([
                ToTensor(),
            ])


    if target == "detection":
        collate = det_collate_fn

    train_dataset_det_params = {
        'root': "data/sticky-trap-insects",
        'split': 'train-half1',
        'patch_size': patch_size,
        'transforms': transforms,
        'target': target,
        'target_params': {'side': 28, 'mask': mask},
        'in_memory': False,
    }

    validation_dataset_det_params = {
        'root': "data/sticky-trap-insects",
        'split': 'train-half2',
        'patch_size': patch_size,
        'transforms': transforms,
        'target': target,
        'target_params': {'side': 28, 'mask': mask},
        'in_memory': False,
        'random_offset': 0,
        'overlap': 120,
    }

    train_dataset_density_params = {
        'root': "data/sticky-trap-insects",
        'split': 'train-half1',
        'patch_size': patch_size,
        'transforms': transforms,
        'target': target,
        'target_params': {'k_size': 75, 'sigma': 11, 'target_normalize_scale_factor': 1.0},
        'in_memory': False,        
    }

    validation_dataset_density_params = {
        'root': "data/sticky-trap-insects",
        'split': 'train-half2',
        'patch_size': patch_size,
        'transforms': transforms,
        'target': target,
        'target_params': {'k_size': 75, 'sigma': 11, 'target_normalize_scale_factor': 1.0},
        'in_memory': False,
        'random_offset': 0,
        'overlap': 120,
    }

    train_dataset_segmentation_params = {
        'root': "data/sticky-trap-insects",
        'split': 'train-half1',
        'patch_size': patch_size,
        'transforms': transforms,
        'target': target,
        'target_params': {'radius': 18, 'radius_ignore': 22, 'v_bal': 0.1, 'sigma_bal': 8, 'sep_width': 1, 'sigma_sep': 4, 'lambda_sep': 50},
        'in_memory': False,   
    }

    validation_dataset_segmentation_params = {
        'root': "data/sticky-trap-insects",
        'split': 'train-half2',
        'patch_size': patch_size,
        'transforms': transforms,
        'target': target,
        'target_params': {'radius': 18, 'radius_ignore': 22, 'v_bal': 0.1, 'sigma_bal': 8, 'sep_width': 1, 'sigma_sep': 4, 'lambda_sep': 50},
        'in_memory': False,
        'random_offset': 0,
        'overlap': 100,       
    }

    batch_size = 1
    num_workers = 0

    debug_dir = Path('datasets/trash')
    debug_dir.mkdir(exist_ok=True)

    if target == "detection":
        if phase == "train":
            dataset = DotAnnotatedDataset(**train_dataset_det_params)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
        elif phase == "validation":
            dataset = DotAnnotatedDataset(**validation_dataset_det_params)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    elif target == "density":
        if phase == "train":
            dataset = DotAnnotatedDataset(**train_dataset_density_params)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        elif phase == "validation":
            dataset = DotAnnotatedDataset(**validation_dataset_density_params)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif target == "segmentation":
        if phase == "train":
            dataset = DotAnnotatedDataset(**train_dataset_segmentation_params)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        elif phase == "validation":
            dataset = DotAnnotatedDataset(**validation_dataset_segmentation_params)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    text_pos = (5, 5)

    for index, sample in enumerate(data_loader):
        if index == debug_num_sample:
            break

        image_ids = sample[4]

        if target == "detection":
            images, targets = build_coco_compliant_batch(sample[0], mask=mask)      # a fake bb is added if 0 insects
            for img, target, image_id in zip(images, targets, image_ids):
                image_bb, image_label = target['boxes'], target['labels']
                class_labels = np.unique(image_label)

                img = to_pil_image(img)
                
                j = 0
                for i in class_labels:
                    class_img = img.copy()
                    draw = ImageDraw.Draw(class_img)

                    for bb in image_bb[image_label == i].tolist():
                        draw.rectangle(bb, outline='red', width=2)

                    # Add text to image
                    text = f"GT N. Insects (Cls#{i}): {len(image_bb)}"
                    draw.text(text_pos, text=text, fill=(0, 191, 255))

                    class_img.save(debug_dir / f'cls{i}_patch{j}_{image_id}')
                    j += 1

        elif target == "density":
            def normalize_map(density_map):
                dmin, dmax = density_map.min(), density_map.max()
                if dmin == dmax:
                    return density_map
                return (density_map - dmin) / (dmax - dmin)

            input_and_target = sample[0]
            # split channels to get input and target maps
            n_channels = input_and_target.shape[1]
            x_channels = 3
            y_channels = n_channels - x_channels
            images, gt_dmaps = input_and_target.split((x_channels, y_channels), dim=1)
            for img, gt_dmap, image_id in zip(images, gt_dmaps, image_ids):
                class_label = gt_dmap.shape[0]

                img = (255 * img.cpu().numpy()).astype(np.uint8)
                img = np.moveaxis(img, 0, -1)
                img = Image.fromarray(img).convert("RGB")
                img.save(debug_dir / f'img_{image_id}')

                for i in range(class_label):
                    count = gt_dmap.sum()
                    gt_dmap = normalize_map(gt_dmap[i, :, :])
                    gt_dmap = (255 * gt_dmap.cpu().squeeze().numpy()).astype(np.uint8)
                    #gt_dmap = np.moveaxis(gt_dmap, 0, -1)
                    gt_dmap = Image.fromarray(gt_dmap)
                    draw = ImageDraw.Draw(gt_dmap)
                    text = f"Num of Insects: {count}"
                    draw.text((text_pos), text=text, fill=191)
                    gt_dmap.save(debug_dir / f'dmap_cls{i}_{image_id}')

        elif target == "segmentation":
            input_and_target = sample[0]
            n_channels = input_and_target.shape[1]
            x_channels = 3
            y_channels = (n_channels - x_channels) // 2
            images, targets, weights = input_and_target.split((x_channels, y_channels, y_channels), dim=1)
            for img, seg_map, w_map, image_id in zip(images, targets, weights, image_ids):
                class_label = seg_map.shape[0]

                img = (255 * img.cpu().numpy()).astype(np.uint8)
                img = np.moveaxis(img, 0, -1)
                img = Image.fromarray(img).convert("RGB")
                img.save(debug_dir / f'img_{image_id}')

                for i in range(class_label):
                    seg_map = seg_map[i, :, :]
                    seg_map = (255 * seg_map.cpu().squeeze().numpy()).astype(np.uint8)
                    seg_map = Image.fromarray(seg_map)
                    seg_map.save(debug_dir / f'segmap_cls{i}_{image_id}')
                    w_map = w_map[i, :, :]
                    w_map = (255 * w_map.cpu().squeeze().numpy()).astype(np.uint8)
                    w_map = Image.fromarray(w_map)
                    w_map.save(debug_dir / f'wmap_cls{i}_{image_id}')

if __name__ == "__main__":
    main()
