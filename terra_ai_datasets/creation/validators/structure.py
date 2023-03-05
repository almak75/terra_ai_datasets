from datetime import datetime

from pydantic import BaseModel

from terra_ai_datasets.creation.validators.tasks import TasksChoice


class DatasetData(BaseModel):
    task: TasksChoice
    use_generator: bool
    is_created: bool
    date: datetime = datetime.now().isoformat()

    class Config:
        use_enum_values = True
