from pydantic import BaseModel
from typing import Optional, Dict


class Parameters(BaseModel):
    name: str
    module_name: str
    module_type: str
    sample_rate: Optional[int] = None
    values: Optional[Dict[str, float]] = None