from pathlib import Path
import json

import pandas as pd
import numpy as np

from terra_ai_datasets.creation.dataset import TerraDataset
from terra_ai_datasets.creation.utils import DatasetPathsData
from terra_ai_datasets.creation.validators import creation_data
from terra_ai_datasets.creation.validators.structure import DatasetData


class LoadDataset(TerraDataset):
    def __init__(self, path_to_dataset: str):
        path_to_dataset = Path(path_to_dataset)
        assert path_to_dataset.is_dir(), "Датасет по указанному пути не найден"

        dataset_paths_data = DatasetPathsData(Path(path_to_dataset))

        for split in dataset_paths_data.instructions.dataframe.iterdir():
            self.dataframe[split.stem] = pd.read_csv(split, index_col=0)

        for instr_path in dataset_paths_data.instructions.parameters.iterdir():
            put, put_id, put_type = instr_path.stem.split('_')
            with open(instr_path, 'r') as conf:
                put_data = getattr(creation_data, f"{put.capitalize()}Data")(**json.loads(json.load(conf)))
            self.input.append(put_data) if put == 'input' else self.output.append(put_data)

        with open(dataset_paths_data.config, 'r') as conf:
            self.dataset_data = DatasetData(**json.loads(json.load(conf)))

        if self.dataset_data.is_created:
            for split, dataframe in self.dataframe.items():
                if self.dataset_data.use_generator:
                    self._dataset[split] = self.create_dataset_object_from_instructions(
                        self.input, self.output, dataframe
                    )
                else:
                    for array_path in dataset_paths_data.arrays.inputs.iterdir():
                        put_id, put = array_path.stem.split('_')
                        self.X[put][put_id] = np.load(array_path)
                    for array_path in dataset_paths_data.arrays.outputs.iterdir():
                        put_id, put = array_path.stem.split('_')
                        self.Y[put][put_id] = np.load(array_path)
                    self._dataset[split] = self.create_dataset_object_from_arrays(self.X[split], self.Y[split])
