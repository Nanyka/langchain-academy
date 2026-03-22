import uuid
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError

from typing import Any, Literal, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs
from langchain_core.messages import SystemMessage, HumanMessage

import os

from langchain_ollama import ChatOllama

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

import configuration

## Utilities 

def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    return str(content)


def _last_human_message_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return _normalize_text(_message_text(message))
    return ""


def _is_read_only_todo_request(messages: list[Any]) -> bool:
    message_text = _last_human_message_text(messages)
    if not message_text:
        return False

    todo_terms = ("todo", "to do", "to-do", "task", "tasks")
    read_terms = ("summary", "summarize", "list", "show", "what are", "what's", "whats", "remind me", "tell me")
    write_terms = ("add ", "create ", "update ", "change ", "modify ", "mark ", "delete ", "remove ", "archive ")

    mentions_todos = any(term in message_text for term in todo_terms)
    asks_to_read = any(term in message_text for term in read_terms)
    asks_to_write = any(term in message_text for term in write_terms)
    return mentions_todos and asks_to_read and not asks_to_write


def _coerce_tool_calls(messages: list, schema_name: str) -> list[dict[str, Any]]:
    """Run a focused tool-calling extraction prompt and return matching tool calls."""
    response = model.invoke(messages)
    return [call for call in getattr(response, "tool_calls", []) if call["name"] == schema_name]


def _merge_unique(existing: list[str], updates: list[str]) -> list[str]:
    merged = list(existing)
    seen = {_normalize_text(item) for item in existing}
    for item in updates:
        if item and _normalize_text(item) not in seen:
            merged.append(item)
            seen.add(_normalize_text(item))
    return merged


def _merge_profile(existing_profile: Optional[dict[str, Any]], new_profile: "Profile") -> dict[str, Any]:
    merged = dict(existing_profile or {})
    profile_data = new_profile.model_dump(mode="json")

    for field in ("name", "location", "job"):
        if profile_data.get(field):
            merged[field] = profile_data[field]

    merged["connections"] = _merge_unique(
        list(existing_profile.get("connections", [])) if existing_profile else [],
        profile_data.get("connections", []),
    )
    merged["interests"] = _merge_unique(
        list(existing_profile.get("interests", [])) if existing_profile else [],
        profile_data.get("interests", []),
    )
    return merged


def _sanitize_todo_args(args: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(args)

    deadline = sanitized.get("deadline")
    if isinstance(deadline, dict):
        sanitized["deadline"] = None
    elif isinstance(deadline, str):
        try:
            sanitized["deadline"] = datetime.fromisoformat(deadline)
        except ValueError:
            sanitized["deadline"] = None

    solutions = sanitized.get("solutions")
    if not isinstance(solutions, list) or not solutions:
        sanitized["solutions"] = ["No specific solution provided."]

    return sanitized

## Schema definitions

# User profile schema
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list
    )
    interests: list[str] = Field(
        description="Interests that the user has", 
        default_factory=list
    )

# ToDo schema
class ToDo(BaseModel):
    task: str = Field(description="The task to be completed.")
    time_to_complete: Optional[int] = Field(description="Estimated time to complete the task (minutes).")
    deadline: Optional[datetime] = Field(
        description="When the task needs to be completed by (if applicable)",
        default=None
    )
    solutions: list[str] = Field(
        description="List of specific, actionable solutions (e.g., specific ideas, service providers, or concrete options relevant to completing the task)",
        min_items=1,
        default_factory=list
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task",
        default="not started"
    )

## Initialize the model and tools

# Update memory tool
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['user', 'todo', 'instructions']

# Initialize the model
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
model = ChatOllama(model=OLLAMA_MODEL, temperature=0)

## Prompts 

# Chatbot instruction for choosing what to update and what tools to call 
MODEL_SYSTEM_MESSAGE = """{task_maistro_role} 

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them) 
2. The user's ToDo list
3. General instructions for updating the ToDo list

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the current ToDo List (may be empty if no tasks have been added yet):
<todo>
{todo}
</todo>

Here are the current user-specified preferences for updating the ToDo list (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below. 

2. Decide whether any of the your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
- If tasks are mentioned, update the ToDo list by calling UpdateMemory tool with type `todo`
- If the user has specified preferences for how to update the ToDo list, update the instructions by calling UpdateMemory tool with type `instructions`

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Tell the user them when you update the todo list
- Do not tell the user that you have updated instructions

4. Err on the side of updating the todo list. No need to ask for explicit permission.

5. Do not update the todo list when the user is only asking to view, summarize, or discuss existing tasks.

6. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made."""

# Focused extraction prompts for updating stored memories
PROFILE_UPDATE_INSTRUCTION = """Extract profile updates from the conversation.

Use the Profile tool to capture only information explicitly stated or strongly implied.
Leave fields empty if the conversation does not update them."""

TODO_UPDATE_INSTRUCTION = """Extract ToDo items from the conversation.

Use the ToDo tool once for each task the user wants tracked.
Prefer concrete tasks with helpful solutions when the conversation provides them.
If no new or updated task is mentioned, do not call any tool."""

# Instructions for updating the ToDo list
CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo list items. Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""

## Node definitions

def task_mAIstro(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load memories from the store and use them to personalize the chatbot's response."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category
    task_maistro_role = configurable.task_maistro_role

   # Retrieve profile memory from the store
    namespace = ("profile", todo_category, user_id)
    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None

    # Retrieve people memory from the store
    namespace = ("todo", todo_category, user_id)
    memories = store.search(namespace)
    todo = "\n".join(f"{mem.value}" for mem in memories)

    # Retrieve custom instructions
    namespace = ("instructions", todo_category, user_id)
    memories = store.search(namespace)
    if memories:
        instructions = memories[0].value
    else:
        instructions = ""
    
    system_msg = MODEL_SYSTEM_MESSAGE.format(task_maistro_role=task_maistro_role, user_profile=user_profile, todo=todo, instructions=instructions)

    # Respond using memory as well as the chat history
    model_input = [SystemMessage(content=system_msg)] + state["messages"]
    if _is_read_only_todo_request(state["messages"]):
        response = model.invoke(model_input)
    else:
        response = model.bind_tools([UpdateMemory]).invoke(model_input)

    return {"messages": [response]}

def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category

    # Define the namespace for the memories
    namespace = ("profile", todo_category, user_id)

    existing_items = store.search(namespace)
    existing_profile = existing_items[0].value if existing_items else None

    extraction_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=PROFILE_UPDATE_INSTRUCTION)] + state["messages"][:-1]
        )
    )
    extraction_response = model.bind_tools([Profile], tool_choice="Profile").invoke(extraction_messages)

    if extraction_response.tool_calls:
        profile_update = Profile.model_validate(extraction_response.tool_calls[0]["args"])
        merged_profile = _merge_profile(existing_profile, profile_update)
        profile_key = existing_items[0].key if existing_items else str(uuid.uuid4())
        store.put(namespace, profile_key, merged_profile)

    tool_calls = state['messages'][-1].tool_calls
    # Return tool message with update verification
    return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id":tool_calls[0]['id']}]}

def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category

    # Define the namespace for the memories
    namespace = ("todo", todo_category, user_id)

    existing_items = store.search(namespace)
    existing_by_task = {}
    for item in existing_items:
        task = item.value.get("task")
        if task:
            existing_by_task[_normalize_text(task)] = item

    extraction_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=TODO_UPDATE_INSTRUCTION)] + state["messages"][:-1]
        )
    )
    extraction_response = model.bind_tools([ToDo]).invoke(extraction_messages)

    added = 0
    updated = 0
    for call in extraction_response.tool_calls:
        if call["name"] != "ToDo":
            continue
        try:
            todo_item = ToDo.model_validate(_sanitize_todo_args(call["args"]))
        except ValidationError:
            continue
        todo_data = todo_item.model_dump(mode="json")
        normalized_task = _normalize_text(todo_data["task"])
        existing_item = existing_by_task.get(normalized_task)
        if existing_item:
            store.put(namespace, existing_item.key, todo_data)
            updated += 1
        else:
            store.put(namespace, str(uuid.uuid4()), todo_data)
            added += 1

    tool_calls = state['messages'][-1].tool_calls
    if added == 0 and updated == 0:
        todo_update_msg = "No ToDo updates were needed."
    else:
        parts = []
        if added:
            parts.append(f"Added {added} ToDo item{'s' if added != 1 else ''}.")
        if updated:
            parts.append(f"Updated {updated} ToDo item{'s' if updated != 1 else ''}.")
        todo_update_msg = " ".join(parts)
    return {"messages": [{"role": "tool", "content": todo_update_msg, "tool_call_id":tool_calls[0]['id']}]}

def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category
    
    namespace = ("instructions", todo_category, user_id)

    existing_memory = store.get(namespace, "user_instructions")
        
    # Format the memory in the system prompt
    system_msg = CREATE_INSTRUCTIONS.format(current_instructions=existing_memory.value if existing_memory else None)
    new_memory = model.invoke([SystemMessage(content=system_msg)]+state['messages'][:-1] + [HumanMessage(content="Please update the instructions based on the conversation")])

    # Overwrite the existing memory in the store 
    key = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})
    tool_calls = state['messages'][-1].tool_calls
    # Return tool message with update verification
    return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id":tool_calls[0]['id']}]}

# Conditional edge
#def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_todos", "update_instructions", "update_profile"]:
def route_message(state: MessagesState, config: RunnableConfig) -> Literal[END, "update_todos", "update_instructions", "update_profile"]:

    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state['messages'][-1]
    if len(message.tool_calls) ==0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call['args']['update_type'] == "user":
            return "update_profile"
        elif tool_call['args']['update_type'] == "todo":
            return "update_todos"
        elif tool_call['args']['update_type'] == "instructions":
            return "update_instructions"
        else:
            raise ValueError

# Create the graph + all nodes
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)

# Define the flow of the memory extraction process
builder.add_node(task_mAIstro)
builder.add_node(update_todos)
builder.add_node(update_profile)
builder.add_node(update_instructions)

# Define the flow 
builder.add_edge(START, "task_mAIstro")
builder.add_conditional_edges("task_mAIstro", route_message)
builder.add_edge("update_todos", "task_mAIstro")
builder.add_edge("update_profile", "task_mAIstro")
builder.add_edge("update_instructions", "task_mAIstro")

# Compile the graph
graph = builder.compile()
