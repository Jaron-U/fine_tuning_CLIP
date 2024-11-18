from .build import DATASET_REGISTRY
from .base_dataset import Datum, DatasetBase
import os

@DATASET_REGISTRY.register()
class MyDataset(DatasetBase):
    dataset_dir = "train_dataset"

    def __init__(self, cfg):
        root = cfg.DATASET.ROOT
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")

        train = self._read_data(train_dir)
        test = self._read_data(test_dir)

        super().__init__(train_x=train, test=test)
    
    def _read_data(self, dir_path):
        items = []
        class_dirs = os.listdir(dir_path)

        for class_dir in class_dirs:
            class_path = os.path.join(dir_path, class_dir)
            if not os.path.isdir(class_path):
                continue
            
            label = int(class_dir.split("_")[0])
            class_name = class_dir.split("_")[1]

            for image_name in os.listdir(class_path):
                img_path = os.path.join(class_path, image_name)
                item = Datum(impath=img_path, label=label, classname=class_name)
                items.append(item)
        return items