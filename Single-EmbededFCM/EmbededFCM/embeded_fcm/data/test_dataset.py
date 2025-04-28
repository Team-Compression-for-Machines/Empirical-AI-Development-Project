import logging
from pathlib import Path

from embeded_fcm.model_wrappers.mmdet.datasets.coco import CocoDataset
from embeded_fcm.model_wrappers.mmdet.datasets.api_wrappers import COCO

class SFUHW(CocoDataset):
    def __init__(
        self,
        root,
        imgs_folder="images",
        annotation_file=None,
        seqinfo="seqinfo.ini",
        dataset_name="sfu-hw-object-v2",
        ext="png"    
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Validate image folder
        _imgs_folder = Path(root) / imgs_folder
        if not _imgs_folder.is_dir():
            raise RuntimeError(f'Invalid image sample directory "{_imgs_folder}"')

        # Validate annotation file
        self._annotation_file = None
        if annotation_file and annotation_file.lower() != "none":
            ann_path = Path(root) / annotation_file
            if not ann_path.is_file():
                raise RuntimeError(f'Invalid annotation file "{ann_path}"')
            self._annotation_file = str(ann_path)
        else:
            self.logger.warning(
                "No annotation found; evaluation may not produce ground-truth metrics."
            )

        # Validate sequence info (optional)
        self._sequence_info_file = None
        if seqinfo and seqinfo.lower() != "none":
            seq_path = Path(root) / seqinfo
            if not seq_path.is_file():
                self.logger.warning(f"Sequence information missing at: {seq_path}")
            else:
                self._sequence_info_file = str(seq_path)
        else:
            self.logger.warning("No sequence information provided.")

        # Store basic attributes
        self._dataset_name = dataset_name
        self._imgs_folder = str(_imgs_folder)
        self._img_ext = ext

        # Define a minimal test pipeline: load image, format bundle, collect
        test_pipeline = [
            dict(type='LoadImageFromFile'),
            #dict(type='DefaultFormatBundle'),
            #dict(type='Collect', keys=['img'])
        ]

        # Initialize base CocoDataset with test pipeline
        super().__init__(
            ann_file=self._annotation_file,
            pipeline=test_pipeline,
            classes=self.CLASSES,
            data_root=str(root),
            img_prefix=self._imgs_folder,
            test_mode=True,
            filter_empty_gt=False
        )

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def dataset(self):
        return self._dataset

    @property
    def annotation_path(self):
        return self._annotation_file

    @property
    def seqinfo_path(self):
        return self._sequence_info_file

    @property
    def imgs_folder_path(self):
        return self._imgs_folder

    def __len__(self):
        return len(self._dataset)
    
    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = f"{self.imgs_folder_path}/{info['file_name']}"
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos