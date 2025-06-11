from enum import Enum
from pydantic import BaseModel, Field

# pydantic ensures output format 
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
    reasoning: str = Field(
        ...,
        description="Additional context or reasoning that explains how the answer was determined."
    )
