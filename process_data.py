import argparse
import random
import shutil
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm


class JobData:
    def __init__(
        self,
        output_directory,
        image,
        label,
    ):
        self.image = image
        self.label = label
        self.output_directory: Path = output_directory


class DatasetProcessor:
    def __init__(self, path, output_directory="data/breastcancer"):
        root_directory = Path(path)
        self.output_directory = Path(output_directory)

        if not self.output_directory.exists():
            self.output_directory.mkdir(parents=True)

        self.imagefile_path = root_directory.joinpath("images", "fold1", "images.npy")
        self.maskfile_path = root_directory.joinpath("masks", "fold1", "masks.npy")

        self.mask = np.load(str(self.maskfile_path))

    def process_images(self):
        print("Generating images")
        shutil.copyfile(
            str(self.imagefile_path),
            self.output_directory.joinpath("images.npy"),
        )

    def process_labels(self):
        print("Generating labels")
        length = self.mask.shape[0]
        labels = np.zeros([length, 256, 256, 2])
        for idx in tqdm(range(length), total=length):
            mask = self.mask[idx]
            labels[idx] = DatasetProcessor.transform_to_hovernet_format(mask)
        np.save(self.output_directory.joinpath("labels"), labels)

    def generate_split(self):
        length = self.mask.shape[0]
        indices = [x for x in range(length)]
        random.shuffle(indices)
        train_indices = indices[: int(length * 0.8)]
        valid_indices = indices[int(length * 0.8) :]
        splits = [
            {
                "train": train_indices,
                "valid": valid_indices,
            }
        ]
        joblib.dump(splits, "splits.dat")

    @staticmethod
    def transform_to_hovernet_format(mask: np.ndarray):
        instance_map = mask[..., :-1].sum(axis=-1)
        class_map = (mask != np.zeros([256, 256, 6])) * np.asanyarray(
            [[[1, 2, 3, 4, 5, 0]]]
        ).repeat(256, axis=0).repeat(256, axis=1)
        class_map = class_map[..., :-1].sum(axis=-1)
        return np.stack([instance_map, class_map], axis=-1)

        # # Generate mask for background
        # mask_lung = masks[1] + masks[2]
        # mask_heart = masks[0]
        # class_map = torch.zeros_like(masks[0])

        # # Assign heart mask as 1 and lung mask as 2
        # class_map += (mask_heart == 255) * 1
        # class_map += (mask_lung == 255) * 2

        # # Generate instance map for each object
        # instance_map = torch.zeros_like(masks[0])
        # for i in range(3):
        #     instance_map += (masks[i] == 255) * (i + 1)
        # return torch.stack([instance_map, class_map], dim=-1)

    # @staticmethod
    # def resize_image(image, shape=[512, 512]):
    #     return resize(
    #         image,
    #         size=shape,
    #         interpolation=InterpolationMode.NEAREST,
    #         antialias=True,
    #     )

    # @staticmethod
    # def rle_to_mask(rle: str, height: int, width: int):
    #     runs = torch.tensor([int(x) for x in rle.split()], dtype=torch.int32)
    #     starts = runs[::2]
    #     lengths = runs[1::2]
    #     mask = torch.zeros([height * width], dtype=torch.uint8)
    #     for start, lengths in zip(starts, lengths):
    #         start -= 1
    #         end = start + lengths
    #         mask[start:end] = 255
    #     return mask.reshape((height, width))

    # @staticmethod
    # def generate_mask(rle_lung_left, rle_lung_right, rle_heart, height, width):
    #     output_image = torch.zeros([height, width, 3], dtype=torch.uint8)
    #     mask_lung_left = DatasetProcessor.rle_to_mask(
    #         rle_lung_left, height=height, width=width
    #     )
    #     mask_lung_right = DatasetProcessor.rle_to_mask(
    #         rle_lung_right, height=height, width=width
    #     )
    #     mask_heart = DatasetProcessor.rle_to_mask(rle_heart, height=height, width=width)
    #     output_image[((mask_lung_left + mask_lung_right) == 255)] = torch.tensor(
    #         [128, 0, 0],
    #         dtype=torch.uint8,
    #     )
    #     output_image[mask_heart == 255] = torch.tensor(
    #         [128, 64, 128],
    #         dtype=torch.uint8,
    #     )
    #     return output_image.permute([2, 0, 1])

    # @staticmethod
    # def generate_mask_hdf5(rle_lung_left, rle_lung_right, rle_heart, height, width):
    #     mask_lung_left = DatasetProcessor.rle_to_mask(
    #         rle_lung_left, height=height, width=width
    #     )
    #     mask_lung_right = DatasetProcessor.rle_to_mask(
    #         rle_lung_right, height=height, width=width
    #     )
    #     mask_heart = DatasetProcessor.rle_to_mask(rle_heart, height=height, width=width)
    #     mask_lung = mask_lung_left + mask_lung_right
    #     foreground = (mask_lung + mask_heart) - (mask_lung * mask_heart)
    #     background = torch.abs(255 - foreground)
    #     return torch.stack([background, mask_lung, mask_heart])

    # @staticmethod
    # def transform_breast_to_hovernet_format(mask: torch.Tensor):
    #     # Generate mask for background
    #     class_map = torch.zeros_like(mask)

    #     # Assign tumor as 1 while background is 0
    #     class_map += (mask == 1) * 1
    #     instance_map = class_map

    #     return torch.stack([instance_map, class_map], dim=-1)

    # @staticmethod
    # def transform_to_hovernet_format(masks: torch.Tensor):
    #     # Generate mask for background
    #     mask_lung = masks[1] + masks[2]
    #     mask_heart = masks[0]
    #     class_map = torch.zeros_like(masks[0])

    #     # Assign heart mask as 1 and lung mask as 2
    #     class_map += (mask_heart == 255) * 1
    #     class_map += (mask_lung == 255) * 2

    #     # Generate instance map for each object
    #     instance_map = torch.zeros_like(masks[0])
    #     for i in range(3):
    #         instance_map += (masks[i] == 255) * (i + 1)
    #     return torch.stack([instance_map, class_map], dim=-1)

    # @staticmethod
    # def generate_new_name(root: Path, path, idx):
    #     filename = path.split("/")[-1]
    #     headername, middle, extension = (
    #         filename.split(".")[0],
    #         filename.split(".")[1],
    #         filename.split(".")[-1],
    #     )
    #     return str(root.joinpath(f"{headername}.{middle}_{idx}.{extension}").absolute())

    # @staticmethod
    # def _process(job: JobData):
    #     image_path, label_path, output_directory = (
    #         job.image,
    #         job.label,
    #         job.output_directory,
    #     )

    # image = read_image(image_path, ImageReadMode.RGB)
    # mask = read_image(label_path, ImageReadMode.GRAY)
    # for idx, (image, mask) in enumerate(
    #     zip(ten_crop(image, [512, 512]), ten_crop(mask, [512, 512]))
    # ):
    #     new_image_path = DatasetProcessor.generate_new_name(
    #         output_directory.joinpath("image"),
    #         image_path,
    #         idx=idx,
    #     )
    #     new_label_path = DatasetProcessor.generate_new_name(
    #         output_directory.joinpath("label"),
    #         image_path,
    #         idx=idx,
    #     )
    #     write_png(image.cpu(), new_image_path)
    #     write_png(mask.cpu(), new_label_path)

    # def process(self):
    #     output_image_path = self.output_directory.joinpath("train")

    #     if not output_image_path.exists():
    #         output_image_path.mkdir()

    #     new_image_path = output_image_path.joinpath("image")
    #     new_label_path = output_image_path.joinpath("label")

    #     if not new_image_path.exists():
    #         new_image_path.mkdir()
    #     if not new_label_path.exists():
    #         new_label_path.mkdir()

    #     images = [x for x in self.image_path.glob("*.png")]
    #     masks = [x for x in self.masks_path.glob("*.png")]

    #     if len(images) != len(masks):
    #         raise Exception("Mismatch number of masks and images")

    #     jobs_data = []
    #     for image_path in images:
    #         jobs_data.append(
    #             JobData(
    #                 image=str(image_path),
    #                 label=str(self.masks_path.joinpath(str(image_path.name))),
    #                 output_directory=output_image_path,
    #             )
    #         )

    #     print("Start Generating")
    #     with ProcessPoolExecutor() as executor:
    #         for _ in tqdm(executor.map(self._process, jobs_data), total=len(jobs_data)):
    #             pass


def process_images(path, output_path):
    train_images_processor = DatasetProcessor(path=path, output_directory=output_path)
    train_images_processor.process_images()
    train_images_processor.process_labels()
    train_images_processor.generate_split()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="directory path")
    parser.add_argument("-o", "--output", required=True, help="output directory")
    parsed: argparse.Namespace = parser.parse_args()
    process_images(path=parsed.path, output_path=parsed.output)
