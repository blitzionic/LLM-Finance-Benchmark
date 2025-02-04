from pydantic import BaseModel, Field
from enum import Enum

class AnswerEnum(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

class AnswerSchema(BaseModel):
    answer: AnswerEnum = Field(
        ...,
        description="The answer to the question, one of A, B, C, or D."
    )
