import os.path as osp

from .builder import DATASETS
from .multimodal import MultimodalDataset


@DATASETS.register_module()
class CarlaDataset(MultimodalDataset):
    """Carla pothole+road dataset.

    Args:

    """

    CLASSES = ('otherobject', 'pothole', 'road')

    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0]]

    def __init__(self, **kwargs):
        super(CarlaDataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', **kwargs)
        assert osp.exists(self.data_root), f"dataset path is not exist"
