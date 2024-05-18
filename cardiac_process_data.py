import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.io import ImageReadMode, read_image, write_png
from torchvision.transforms.functional import InterpolationMode, resize
from tqdm import tqdm


class JobData:
    def __init__(
        self,
        image,
        mask_lung_left,
        mask_lung_right,
        mask_heart,
        height,
        width,
        output_directory,
    ):
        self.image = image
        self.rle_lung_left = mask_lung_left
        self.rle_lung_right = mask_lung_right
        self.rle_heart = mask_heart
        self.height = height
        self.width = width
        self.output_directory: Path = output_directory


class DatasetProcessor:
    def __init__(self, path, output_directory="data/cardiac", is_train=True):
        directory = Path(path)
        self.output_directory = Path(output_directory)
        self.is_train = is_train

        if not self.output_directory.exists():
            self.output_directory.mkdir(parents=True)

        self.csv = pd.read_csv(
            str(directory.joinpath("ChestX-Ray8.csv")),
            engine="pyarrow",
            index_col=0,
        )
        self.images = [x for x in directory.glob("chestxray/images_*/**/*.png")]

    @staticmethod
    def resize_image(image, shape=[512, 512]):
        return resize(
            image,
            size=shape,
            interpolation=InterpolationMode.NEAREST,
            antialias=True,
        )

    @staticmethod
    def rle_to_mask(rle: str, height: int, width: int):
        runs = torch.tensor([int(x) for x in rle.split()], dtype=torch.int32)
        starts = runs[::2]
        lengths = runs[1::2]
        mask = torch.zeros([height * width], dtype=torch.uint8)
        for start, lengths in zip(starts, lengths):
            start -= 1
            end = start + lengths
            mask[start:end] = 255
        return mask.reshape((height, width))

    @staticmethod
    def generate_mask(rle_lung_left, rle_lung_right, rle_heart, height, width):
        output_image = torch.zeros([height, width, 3], dtype=torch.uint8)
        mask_lung_left = DatasetProcessor.rle_to_mask(
            rle_lung_left, height=height, width=width
        )
        mask_lung_right = DatasetProcessor.rle_to_mask(
            rle_lung_right, height=height, width=width
        )
        mask_heart = DatasetProcessor.rle_to_mask(rle_heart, height=height, width=width)
        output_image[((mask_lung_left + mask_lung_right) == 255)] = torch.tensor(
            [128, 0, 0],
            dtype=torch.uint8,
        )
        output_image[mask_heart == 255] = torch.tensor(
            [128, 64, 128],
            dtype=torch.uint8,
        )
        return output_image.permute([2, 0, 1])

    @staticmethod
    def generate_mask_hdf5(rle_lung_left, rle_lung_right, rle_heart, height, width):
        mask_lung_left = DatasetProcessor.rle_to_mask(
            rle_lung_left, height=height, width=width
        )
        mask_lung_right = DatasetProcessor.rle_to_mask(
            rle_lung_right, height=height, width=width
        )
        mask_heart = DatasetProcessor.rle_to_mask(rle_heart, height=height, width=width)
        mask_lung = mask_lung_left + mask_lung_right
        foreground = (mask_lung + mask_heart) - (mask_lung * mask_heart)
        background = torch.abs(255 - foreground)
        return torch.stack([background, mask_lung, mask_heart])

    @staticmethod
    def generate_mask_v2(rle_lung_left, rle_lung_right, rle_heart, height, width):
        mask_lung_left = DatasetProcessor.rle_to_mask(
            rle_lung_left, height=height, width=width
        )
        mask_lung_right = DatasetProcessor.rle_to_mask(
            rle_lung_right, height=height, width=width
        )
        mask_heart = DatasetProcessor.rle_to_mask(rle_heart, height=height, width=width)
        return torch.stack([mask_heart, mask_lung_left, mask_lung_right])

    @staticmethod
    def transform_to_hovernet_format(masks: torch.Tensor):
        # Generate mask for background
        mask_lung = masks[1] + masks[2]
        mask_heart = masks[0]
        class_map = torch.zeros_like(masks[0])

        # Assign heart mask as 1 and lung mask as 2
        class_map += (mask_heart == 255) * 1
        class_map += (mask_lung == 255) * 2

        # Generate instance map for each object
        instance_map = torch.zeros_like(masks[0])
        for i in range(3):
            instance_map += (masks[i] == 255) * (i + 1)
        return torch.stack([instance_map, class_map], dim=-1)

    @staticmethod
    def generate_new_name(root: Path, path):
        filename = path.split("/")[-1]
        return str(root.joinpath(filename).absolute())

    @staticmethod
    def _process(job: JobData):
        (
            image_path,
            rle_lung_left,
            rle_lung_right,
            rle_heart,
            height,
            width,
            output_directory,
        ) = (
            job.image,
            job.rle_lung_left,
            job.rle_lung_right,
            job.rle_heart,
            job.height,
            job.width,
            job.output_directory,
        )

        image = read_image(image_path, ImageReadMode.RGB)
        mask = DatasetProcessor.generate_mask(
            rle_lung_left,
            rle_lung_right,
            rle_heart,
            height,
            width,
        )

        image = DatasetProcessor.resize_image(image)
        mask = DatasetProcessor.resize_image(mask)
        new_image_path = DatasetProcessor.generate_new_name(
            output_directory.joinpath("image"),
            image_path,
        )
        new_label_path = DatasetProcessor.generate_new_name(
            output_directory.joinpath("label"),
            image_path,
        )
        write_png(image.cpu(), new_image_path)
        write_png(mask.cpu(), new_label_path)

    def process(self):
        output_image_path = (
            self.output_directory.joinpath("train")
            if self.is_train
            else self.output_directory.joinpath("test")
        )
        if not output_image_path.exists():
            output_image_path.mkdir()

        new_image_path = output_image_path.joinpath("image")
        new_label_path = output_image_path.joinpath("label")
        if not new_image_path.exists():
            new_image_path.mkdir()
        if not new_label_path.exists():
            new_label_path.mkdir()

        images = [x for x in self.images]
        jobs_data = []
        for image_path in images:
            image_name = image_path.name
            selected_row = self.csv.loc[image_name]
            left_lung_rle = selected_row["Left Lung"]
            right_lung_rle = selected_row["Right Lung"]
            heart_rle = selected_row["Heart"]
            height = selected_row["Height"]
            width = selected_row["Width"]
            jobs_data.append(
                JobData(
                    image=str(image_path),
                    mask_lung_left=left_lung_rle,
                    mask_lung_right=right_lung_rle,
                    mask_heart=heart_rle,
                    height=height,
                    width=width,
                    output_directory=output_image_path,
                )
            )

        print("Start Generating")
        with ProcessPoolExecutor(max_workers=1) as executor:
            for _ in tqdm(executor.map(self._process, jobs_data), total=len(jobs_data)):
                pass

    def process_image(self, output_directory: str, length=None):
        output_image_path = Path(output_directory)
        if not output_image_path.exists():
            output_image_path.mkdir(parents=True)
        images = self.images
        length = len(images) if length is None else length

        print("Start Generating")

        dataset_images = np.zeros([length, 256, 256, 3], dtype=np.uint8)
        for idx, image_path in enumerate(tqdm(images[:length], total=len(images))):
            image = read_image(str(image_path), ImageReadMode.RGB)
            image = DatasetProcessor.resize_image(image, shape=[256, 256])
            dataset_images[idx] = image.permute([1, 2, 0])
        np.save(str(output_image_path.joinpath("images")), dataset_images)

    def process_label(self, output_directory: str, length=None):
        output_image_path = Path(output_directory)
        if not output_image_path.exists():
            output_image_path.mkdir(parents=True)
        images = self.images
        length = len(images) if length is None else length

        print("Start Generating")

        labels = np.zeros([length, 256, 256, 2], dtype=np.uint8)
        for idx, image_path in enumerate(tqdm(images[:length], total=length)):
            image_name = image_path.name
            selected_row = self.csv.loc[image_name]
            left_lung_rle = selected_row["Left Lung"]
            right_lung_rle = selected_row["Right Lung"]
            heart_rle = selected_row["Heart"]
            height = selected_row["Height"]
            width = selected_row["Width"]

            masks = DatasetProcessor.generate_mask_v2(
                left_lung_rle,
                right_lung_rle,
                heart_rle,
                height,
                width,
            )
            masks = DatasetProcessor.resize_image(masks, shape=[256, 256])
            labels[idx] = DatasetProcessor.transform_to_hovernet_format(masks)
            # write_png(((torch.tensor(labels[0, ..., 0]).unsqueeze(0) == 4) * 255).to(torch.uint8), 'instance.png')
            # write_png(((torch.tensor(labels[0, ..., 1]).unsqueeze(0) == 0) * 255).to(torch.uint8), 'map.png')
        np.save(str(output_image_path.joinpath("labels")), labels)


def process_images(path, output_path):
    train_images_processor = DatasetProcessor(path=path, is_train=True)
    train_images_processor.process_image(output_directory=output_path)
    train_images_processor.process_label(output_directory=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="directory path")
    parser.add_argument("-o", "--output", required=True, help="output directory")
    parsed: argparse.Namespace = parser.parse_args()
    process_images(path=parsed.path, output_path=parsed.output)
