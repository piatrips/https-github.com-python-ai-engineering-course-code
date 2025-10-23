from __future__ import annotations
from pydantic import BaseModel, Field, ValidationError
from typing import Optional

class PreprocessingSchema(BaseModel):
    normalize: bool = True
    missing_strategy: str = Field('mean', regex='^(mean|median|drop)$')
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

class TrainingSchema(BaseModel):
    learning_rate: float = Field(0.01, gt=0.0)
    epochs: int = Field(10, ge=1)
    batch_size: int = Field(32, ge=1)
    model_name: str = 'simple'


if __name__ == '__main__':
    try:
        p = PreprocessingSchema(normalize=False, missing_strategy='mean')
        t = TrainingSchema(learning_rate=0.1, epochs=5, batch_size=16)
        print('preproc:', p.dict())
        print('train:', t.dict())
    except ValidationError as e:
        print('Validation failed:', e.json())