"""
LLMs4OL-inspired ontology learning pipeline. Assumptions:
- generic in nature, no assumption on the domain,
- the pipeline does not get any hints about the type
  of task it'll be evaluated on.

Original LLMs4OL does not cover integration step, so this approach
is an extension, to suit better our problem.

 Input Text
     |
     +----- Task A: Term Typing -----> Types
     |                                   |
     +----- Task B: Taxonomy --------> Hierarchy
     |                                   |
     +----- Task C: Relations -------> Relations
                                         |
                                         v
                                Integration Step (LLM)
                                         |
                                         v
                                   Populated OWL Ontology
"""

from functools import partial
from pathlib import Path
from typing import Final, TypedDict

from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import StrOutputParser

from ..config import chat_model
from ..fix_pipeline import fix_ontology
from ..strip_pipeline import strip_ontology
from ..get_answer import get_answer
from ..tasks import Task
from ..utils import debuglog


TASK_A_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["input"],
    template="""
Given the following text, identify and classify all significant terms:

Text: {input}

For each important term found in the text, classify it using this format:
"[TERM] is a [TYPE]"

Example output:
- "Purchase Price is a financial_term"
- "Target Company is a legal_entity"
""",
)


TASK_B_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["input"],
    template="""
Given the following text, identify hierarchical relationships between any types mentioned:

Text: {input}

For each potential hierarchy, evaluate if true or false:
"[TYPE_A] is a subclass of [TYPE_B]. This statement is [TRUE/FALSE]"

Example output:
- "Subsidiary is a subclass of Legal_Entity. This statement is TRUE"
- "Contract is a subclass of Person. This statement is FALSE"
""",
)


TASK_C_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["input"],
    template="""
Given the following text, identify semantic relationships between entities:

Text: {input}

For each potential relationship, evaluate if true or false:
"[ENTITY_A] [RELATION] [ENTITY_B]. This statement is [TRUE/FALSE]"

Example output:  
- "Acquirer purchases Target_Company. This statement is TRUE"
- "Purchase_Price guarantees Closing_Date. This statement is FALSE"
""",
)


INTEGRATION_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["task_a", "task_b", "task_c"],
    template="""
You are an ontology expert. Create a complete OWL ontology from the following ontology learning results:

**Term Classifications:**
{task_a}

**Hierarchical Relationships:**
{task_b}

**Semantic Relationships:**
{task_c}

Generate a complete OWL ontology in Turtle format that includes:

1. **Classes**: Create owl:Class for each unique type from term classifications and hierarchical relationships
2. **Class Hierarchy**: Use rdfs:subClassOf for TRUE hierarchical relationships
3. **Object Properties**: Create owl:ObjectProperty for each relation type from semantic relationships
4. **Individuals**: Create instances for classified terms
5. **Property Assertions**: Add property statements for TRUE semantic relationships

Requirements:
- Use proper OWL syntax and namespaces
- Handle conflicts by choosing the most specific classification
- Only include TRUE statements from relationships
- Use CamelCase for class names and camelCase for properties
- Include rdfs:label for human-readable names

Output format: Valid Turtle (.ttl) syntax only, no explanations. Do not add comments.
""",
)


class State(TypedDict):
    model: str
    input: str
    task_a_result: str | None
    task_b_result: str | None
    task_c_result: str | None
    populated_ontology: str | None


@debuglog
def task_a(state: State, **kwargs):
    llm = chat_model(state["model"]) | StrOutputParser()
    res = llm.invoke(TASK_A_PROMPT.format(input=state["input"]))
    return {"task_a_result": res}


@debuglog
def task_b(state: State, **kwargs):
    llm = chat_model(state["model"]) | StrOutputParser()
    res = llm.invoke(
        TASK_B_PROMPT.format(input=state["input"]),
    )
    return {"task_b_result": res}


@debuglog
def task_c(state: State, **kwargs):
    llm = chat_model(state["model"]) | StrOutputParser()
    res = llm.invoke(
        TASK_C_PROMPT.format(input=state["input"]),
    )
    return {"task_c_result": res}


@debuglog
def integration(state: State, **kwargs):
    llm = chat_model(state["model"]) | StrOutputParser()
    res = llm.invoke(
        INTEGRATION_PROMPT.format(
            task_a=state["task_a_result"],
            task_b=state["task_b_result"],
            task_c=state["task_c_result"],
        ),
    )
    return {"populated_ontology": res}


def graph(debug: bool):
    builder = StateGraph(State)
    ctx = {"debug": debug}

    builder.add_node("task_a", partial(task_a, **ctx))
    builder.add_node("task_b", partial(task_b, **ctx))
    builder.add_node("task_c", partial(task_c, **ctx))
    builder.add_node("integration", partial(integration, **ctx))

    builder.add_edge(START, "task_a")
    builder.add_edge(START, "task_b")
    builder.add_edge(START, "task_c")
    builder.add_edge("task_a", "integration")
    builder.add_edge("task_b", "integration")
    builder.add_edge("task_c", "integration")
    builder.add_edge("integration", END)

    return builder.compile()


def predict(
    model: str,
    task: Task,
    input: str,
    debug: bool,
    fix: bool,
    strip: bool,
    task_dir: Path | None,
) -> str | None:
    populated_ontology = graph(debug).invoke(
        {
            "model": model,
            "input": input,
        }
    )["populated_ontology"]
    if not populated_ontology:
        print(f"err: failed to construct ontology for {input}")
        return None
    if fix:
        populated_ontology = fix_ontology(model, input, populated_ontology, debug)
    if strip:
        populated_ontology = strip_ontology(model, populated_ontology, debug=debug)
    if task_dir:
        with open(task_dir / "llms4ol", "w") as f:
            f.write(populated_ontology)
    return get_answer(model, task, populated_ontology, debug=debug)
