from pydantic import BaseModel
from typing import Optional, Dict, Literal


class Parameters(BaseModel):
    name: str
    module_name: str
    module_type: Literal["lti", "non-linear"]
    sample_rate: Optional[int] = None
    values: Optional[Dict[str, float]] = None