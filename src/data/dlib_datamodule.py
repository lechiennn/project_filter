from typing import Any, Dict, Optional, Tuple
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import albumentations as A
from albumentations import Compose
import os
from xml.etree import ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw

class DlibDataset(Dataset):
    
    def __init__(
        self,
        data_dir: str = 'data/ibug_300W_large_face_landmark_dataset',
        xml_file: str = 'labels_ibug_300W_train.xml'
    ):
        super().__init__()
        self.data_dir = data_dir
        self.imageList = self._load(data_dir, xml_file)

        
    def _load(self, data_dir, xml_file):
        tree = ET.parse(os.path.join(data_dir,xml_file))
        root = tree.getroot()
        images = root.find('images')

        imageList = []

        for image in images:
            filename = image.get('file')
            width =  int(image.get('width'))
            height = int(image.get('height'))

            box = image.find('box')
            box_top = int(box.get('top'))
            box_left = int(box.get('left'))
            box_width = int(box.get('width'))
            box_height = int(box.get('height'))

            landmarks = np.array([
                [float(part.get('x')), float(part.get('y'))] for part in box
            ])
            
            landmarks -= np.array([box_left, box_top])  # crop

            imageList.append(dict(
                filename=filename, width=width, height=height,
                box_top=box_top, box_left=box_left, box_width=box_width, box_height=box_height,
                landmarks=landmarks,)
            )
            
        return imageList


    def __len__(self):
        return len(self.imageList)
    
    def __getitem__(self, idx):
        sample = self.imageList[idx]
        filename = sample['filename']
        box_top: int = sample['box_top']
        box_left: int = sample['box_left']
        box_width: int = sample['box_width']
        box_height: int = sample['box_height']
        landmarks: np.ndarray = sample['landmarks']
        original_image: Image = Image.open(
            os.path.join(self.data_dir, filename)).convert('RGB')
        cropped_image: Image = original_image.crop(
            (box_left, box_top, box_left+box_width, box_top+box_height))

        return cropped_image, landmarks # unnormalized
    
    @staticmethod
    def annotate_image(image: Image, landmarks: np.ndarray):
        draw = ImageDraw.Draw(image)
        for landmark in landmarks:
            topleft = landmark - 2
            botright = landmark + 2
            draw.ellipse([topleft[0], topleft[1], botright[0], botright[1]], fill='red')
        
        return image


class TransformDataset(Dataset):
    def __init__(self, dataset: DlibDataset, transform = None):
        self.dataset = dataset
        if transform is not None:
            self.transform = transform
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToTensorV2(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # image in PIL format, landmarks in image pixel coordinates
        image, landmarks = self.dataset[idx]
        image = np.array(image)
        transformed = self.transform(
            image=image, keypoints=landmarks)
        image, landmarks = transformed["image"], transformed["keypoints"]
        _, height, width = image.shape
        landmarks = landmarks / np.array([width, height]) - 0.5
        return image, landmarks.astype(np.float32) # center and normalize

    # @staticmethod
    # def collate_fn(batch):
    #     images, landmarks = zip(*batch)
    #     return torch.stack(images), np.stack(landmarks)

    ## assume image batch tensor, normalized by imagenet
    @staticmethod
    def annotate_tensor(image: torch.Tensor, landmarks: np.ndarray) -> Image:

        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]

        def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
            # 3, H, W, B
            ten = x.clone().permute(1, 2, 3, 0)
            for t, m, s in zip(ten, mean, std):
                t.mul_(s).add_(m)
            # B, 3, H, W
            return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

        images = denormalize(image)
        images_to_save = []
        for lm, img in zip(landmarks, images):
            img = img.permute(1, 2, 0).numpy()*255
            h, w, _ = img.shape
            lm = (lm + 0.5) * np.array([w, h]) # convert to image pixel coordinates
            img = DlibDataset.annotate_image(Image.fromarray(img.astype(np.uint8)), lm)
            images_to_save.append( torchvision.transforms.ToTensor()(img) )

        return torch.stack(images_to_save)


class DlibDataModule(LightningDataModule):
    '''
    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test
    '''

    def __init__(
        self,
        data_dir: str = "data/ibug_300W_large_face_landmark_dataset",
        train_val_split: Tuple[int, int, int] = (6000, 666),
        data_train: DlibDataset = None,
        data_test: DlibDataset = None,
        train_transform: Optional[Compose] = None,
        test_transform: Optional[Compose] = None,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""

        if not self.data_train and not self.data_val and not self.data_test:
            data_train = self.hparams.data_train(data_dir=self.hparams.data_dir)
            data_test = self.hparams.data_test(data_dir=self.hparams.data_dir)
            
            data_train, data_val = random_split(
                dataset=data_train,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42),
            )

            self.data_train = TransformDataset(data_train, transform=self.hparams.train_transform)
            self.data_val = TransformDataset(data_val, transform=self.hparams.test_transform)
            self.data_test = TransformDataset(data_test, transform=self.hparams.test_transform)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass



if __name__ == "__main__":
    import pyrootutils
    import torchvision
    from omegaconf import DictConfig
    import hydra
    import numpy as np
    from PIL import Image, ImageDraw
    from tqdm import tqdm

    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "data")
    output_path = path / "outputs"
    print("root", path, config_path)
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    def test_dataset(cfg: DictConfig):
        dataset: DlibDataset = hydra.utils.instantiate(cfg.data_train)
        dataset = dataset(data_dir=cfg.data_dir)
        print("dataset", len(dataset))
        image, landmarks = dataset[100]
        print("image", image.size, "landmarks", landmarks.shape)
        annotated_image = DlibDataset.annotate_image(image, landmarks)
        annotated_image.save(output_path / "test_dataset_result.png")

    def test_datamodule(cfg: DictConfig):
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
        datamodule.prepare_data()
        datamodule.setup()
        loader = datamodule.train_dataloader()
        bx, by = next(iter(loader))
        print("n_batch", len(loader), bx.shape, by.shape, type(by))
        annotated_batch = TransformDataset.annotate_tensor(bx, by)
        print("annotated_batch", annotated_batch.shape)
        torchvision.utils.save_image(annotated_batch, output_path / "test_datamodule_result.png")
        
        for bx, by in tqdm(datamodule.train_dataloader()):
            pass
        print("training data passed")

        for bx, by in tqdm(datamodule.val_dataloader()):
            pass
        print("validation data passed")

        for bx, by in tqdm(datamodule.test_dataloader()):
            pass
        print("test data passed")

    @hydra.main(version_base="1.3", config_path=config_path, config_name="dlib.yaml")
    def main(cfg: DictConfig):
        # print(cfg)
        test_dataset(cfg)
        test_datamodule(cfg)
    main()