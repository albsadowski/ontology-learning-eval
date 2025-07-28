from typing import Final

from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from .config import chat_model
from .tasks import Task
from .utils import debuglog


class MAUDResult(BaseModel):
    option: str = Field(
        ..., description="label of selected option, e.g. A, B, C, D, ..."
    )


PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=[],
    template="""
# Instruction

Analyse the ontology extracted from a merger agreement and answer the multiple-choice question
by choosing the option that best characterizes the agreement.

# Ontology
{ontology}

# Question

{question}
{options}
""",
)


@debuglog
def get_answer(model: str, task: Task, ontology: str, **kwargs):
    try:
        res = (
            chat_model(model)
            .with_structured_output(MAUDResult)
            .invoke(
                PROMPT.format(
                    ontology=ontology,
                    question=task.question,
                    options="\n".join(
                        f"Option {label}: {answer}" for label, answer in task.answers
                    ),
                )
            )
        )
    except Exception as err:
        print(f"err: failed to parse MAUDResult\n{err}")
        return None
    if isinstance(res, MAUDResult):
        return res.option.replace("Option ", "")
    else:
        print("err: failed to parse MAUDResult")
        return None
