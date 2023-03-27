from pathlib import Path
import json

import pandas as pd
import joblib

from terra_ai_datasets.creation.dataset import TerraDataset
from terra_ai_datasets.creation.utils import DatasetPathsData
from terra_ai_datasets.creation.validators import creation_data, inputs, outputs
from terra_ai_datasets.creation.validators.structure import DatasetData
from terra_ai_datasets.creation.validators.tasks import LayerInputTypeChoice


class LoadDataset(TerraDataset):

    def __init__(self, path_to_dataset: str):
        super().__init__()
        path_to_dataset = Path(path_to_dataset)
        assert path_to_dataset.is_dir(), "Датасет по указанному пути не найден"

        dataset_paths_data = DatasetPathsData(Path(path_to_dataset))

        for split in dataset_paths_data.instructions.dataframe.iterdir():
            self.dataframe[split.stem] = pd.read_csv(split, index_col=0)

        for instr_path in dataset_paths_data.instructions.parameters.iterdir():
            put, put_id, put_type = instr_path.stem.split('_')
            if not self.put_instructions.get(int(put_id)):
                self.put_instructions[int(put_id)] = {}
            with open(instr_path, 'r') as conf:
                dict_data = json.loads(json.load(conf))
                self.put_instructions[int(put_id)][f"{put_id}_{put_type}"] = \
                    getattr(creation_data, f"{put.capitalize()}InstructionsData")(**dict_data)
                put = self.put_instructions[int(put_id)][f'{put_id}_{put_type}']
                self.put_instructions[int(put_id)][f"{put_id}_{put_type}"].parameters = \
                    getattr(inputs, f"{put.type}Validator")(**dict_data['parameters']) \
                    if put.type in LayerInputTypeChoice\
                    else getattr(outputs, f"{put.type}Validator")(**dict_data['parameters'])

        with open(dataset_paths_data.config, 'r') as conf:
            self.dataset_data = DatasetData(**json.loads(json.load(conf)))

        for prep_path in dataset_paths_data.preprocessing.iterdir():
            self.preprocessing[prep_path.stem] = joblib.load(prep_path)

        if self.dataset_data.is_created:
            for split, dataframe in self.dataframe.items():
                if self.dataset_data.use_generator:
                    self._dataset[split] = self.create_dataset_object_from_instructions(
                        self.put_instructions, dataframe
                    )
                else:
                    for array_path in dataset_paths_data.arrays.inputs.iterdir():
                        put_id, put = array_path.stem.split('_')
                        self.X[put][int(put_id)] = joblib.load(array_path)
                    for array_path in dataset_paths_data.arrays.outputs.iterdir():
                        put_id, put = array_path.stem.split('_')
                        self.Y[put][int(put_id)] = joblib.load(array_path)
                    self._dataset[split] = self.create_dataset_object_from_arrays(self.X[split], self.Y[split])
