from pathlib import Path
import json

import pandas as pd

from terra_ai_datasets.creation.dataset import TerraDataset
from terra_ai_datasets.creation.utils import DatasetPathsData
from terra_ai_datasets.creation.validators import creation_data
from terra_ai_datasets.creation.validators.structure import DatasetData


class LoadDataset(TerraDataset):
    def __init__(self, path_to_dataset: str):
        path_to_dataset = Path(path_to_dataset)
        assert path_to_dataset.is_dir(), "Датасет по указанному пути не найден"

        self.dataset_paths_data = DatasetPathsData(Path(path_to_dataset))

        for split in self.dataset_paths_data.instructions.dataframe.iterdir():
            self.dataframe[split.stem] = pd.read_csv(split, index_col=0)

        for instr_path in self.dataset_paths_data.instructions.parameters.iterdir():
            put, put_id, put_type = instr_path.stem.split('_')
            with open(instr_path, 'r') as conf:
                put_data = getattr(creation_data, f"{put.capitalize()}Data")(**json.loads(json.load(conf)))
            self.input.append(put_data) if put == 'input' else self.output.append(put_data)

        with open(self.dataset_paths_data.config, 'r') as conf:
            self.dataset_data = DatasetData(**json.loads(json.load(conf)))
