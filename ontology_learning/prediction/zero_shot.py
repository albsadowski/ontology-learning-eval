from typing import Final

from langchain.prompts import PromptTemplate

from ..config import chat_model
from ..get_answer import MAUDResult
from ..tasks import Task


PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["input", "instruction", "question", "options"],
    template="""
# Instruction

{instruction}

# Merger Agreement

{input}

# Question

{question}
{options}

# Output Format

Return a single, valid JSON object that strictly adheres to the following structure. Do not
include any markdown formatting (e.g., ```json) around the JSON object itself.

```json
{{
    "option": "C"
}}
```
""",
)


def predict(model: str, task: Task, input: str, **kwargs) -> str | None:
    prompt = PROMPT.format(
        input=input,
        instruction=task.instruction,
        question=task.question,
        options="\n".join(
            f"Option {label}: {answer}" for label, answer in task.answers
        ),
    )
    res = (
        chat_model(model)
        .with_structured_output(MAUDResult, method="json_schema")
        .invoke(prompt)
    )
    if isinstance(res, MAUDResult):
        return res.option
    else:
        print("err: failed to parse MAUD result")
        return None
