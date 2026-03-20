from pydantic import BaseModel, Field

from trustcall import create_extractor

from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig
import os

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
import configuration

# Initialize the LLM
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
model = ChatOllama(model=OLLAMA_MODEL, temperature=0) 

# Schema 
class UserProfile(BaseModel):
    """ Profile of a user """
    user_name: str | None = Field(default=None, description="The user's preferred name")
    user_location: str | None = Field(default=None, description="The user's location")
    interests: list[str] = Field(default_factory=list, description="A list of the user's interests")

# Create the extractor
trustcall_extractor = create_extractor(
    model,
    tools=[UserProfile],
    tool_choice="UserProfile", # Enforces use of the UserProfile tool
)

# Chatbot instruction
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user. 
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# Extraction instruction
TRUSTCALL_INSTRUCTION = """Create or update the memory (JSON doc) to incorporate information from the following conversation:"""

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load memory from the store and use it to personalize the chatbot's response."""
    
    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)

    # Get the user ID from the config
    user_id = configurable.user_id

    # Retrieve memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    # Format the memories for the system prompt
    if existing_memory and existing_memory.value:
        memory_dict = existing_memory.value
        formatted_memory = (
            f"Name: {memory_dict.get('user_name') or 'Unknown'}\n"
            f"Location: {memory_dict.get('user_location') or 'Unknown'}\n"
            f"Interests: {', '.join(memory_dict.get('interests', []))}"      
        )
    else:
        formatted_memory = None

    # Format the memory in the system prompt
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted_memory)

    # Respond using memory as well as the chat history
    response = model.invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": response}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and save a memory to the store."""
    
    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)

    # Get the user ID from the config
    user_id = configurable.user_id

    # Retrieve existing memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")
    existing_value = existing_memory.value if existing_memory else {}

    # Get the profile as the value from the list, and convert it to a JSON doc
    existing_profile = {"UserProfile": existing_value} if existing_value else None
    extractor_messages = [SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"]
    updated_profile = {}

    try:
        result = trustcall_extractor.invoke(
            {"messages": extractor_messages, "existing": existing_profile}
        )
        responses = result.get("responses", [])
        if responses:
            updated_profile = responses[0].model_dump(
                exclude_none=True, exclude_defaults=True
            )
    except Exception:
        updated_profile = {}

    if not updated_profile:
        fallback_model = model.with_structured_output(UserProfile)
        fallback_result = fallback_model.invoke(extractor_messages)
        updated_profile = fallback_result.model_dump(
            exclude_none=True, exclude_defaults=True
        )

    if not updated_profile:
        return

    merged_profile = dict(existing_value)
    merged_profile.update(updated_profile)

    # Save the updated profile
    key = "user_memory"
    store.put(namespace, key, merged_profile)

# Define the graph
builder = StateGraph(MessagesState,config_schema=configuration.Configuration)
builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)
graph = builder.compile()
