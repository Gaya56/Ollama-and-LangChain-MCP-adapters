 Understanding the MCP Workflow: Building a Local MCP client using Ollama and LangChain MCP adapters
Plaban Nayak
Plaban Nayak

Follow
20 min read
·
May 24, 2025
171







🌟 What is MCP?
Model Context Protocol (MCP) is an innovative protocol designed to seamlessly connect AI models with applications.

MCP is an open source protocol that standardizes how our LLM applications connect to and work with the desired tools and data sources.

Think of it as a universal translator between your AI models and the applications that want to use them. Just like how HTTP helps web browsers talk to web servers, MCP helps applications talk to AI models!


Model Context Protocol Architecture
🚀Why MCP ?
Models are as good as the context given to them. We can have an incredible model at our vicinity but if it does not has the ability to connect to outside world and retrieve data and context necessary then it will not be as useful as it can possibly be.

Everything that we are going to do with MCP can be achieved without MCP.But think of every tool we need to connect to the LLM to make it more reasonable.Suppose we have to use different APIs to cater to different services. But every service provider construct their APIs in different probably in a different language.

Here we want to make sure that we are communicating to all the required APIs in the same language. This is whre MCP comes to the rescue. It standardizes how our AI applications interact with external systems. Instead of building same integration for different data sources over and over again, depending on the data source it helps to build the required facility once and reuse it.

💪 Key Advantages of MCP
🔌 Plug-and-Play Integration

Connect any AI model to any application
No need to rewrite code when switching models
Standardized communication protocol
🛠️ Tool-First Approach

Models expose their capabilities as tools
Applications can discover and use tools dynamically
Perfect for agents and autonomous systems
🌐 Language Agnostic

Write clients in any programming language
Models can be implemented in any framework
True interoperability across the AI ecosystem
⚡ Real-time Communication

Support for Server-Sent Events (SSE)
Stream results as they become available
Perfect for chat applications and interactive systems
🏗️ Architecture Deep Dive
🖥️ Server Side (server.py)
@mcp.tool()
def add_data(query: str) -> bool:
    """Add new data to the people table"""
    # Tool implementation
The server:

📝 Defines tools as simple Python functions
🎯 Exposes tools through MCP decorators
🔄 Handles requests and returns responses
📚 Maintains tool documentation and schemas
📡 Server Side (server.py) Components
FastMCP Server
MCP Tools
Database Management
Error Handling
👥 Client Side (langchain_client.py)
class LangchainMCPClient:
    async def initialize_agent(self):
        """Initialize the agent with tools"""
        # Client implementation
The client:

🤝 Connects to the MCP server
🔍 Discovers available tools
🤖 Creates an AI agent to use the tools
💬 Handles user interactions
👥 Client Side Components (langchain_client.py)
LangChain Integration
Agent System
Tool Management
Chat History
🔗 MCP Layer
Tool Registration
Request Handling
Response Processing
Event Streaming
🔄 The Workflow
🚀 Server Startup
Server initializes with tools
Tools register their capabilities
Server starts listening for connections
2. 🤝 Client Connection

Client discovers server
Retrieves available tools
Creates agent with tool access
3. 💬 User Interaction

User sends request
Agent processes request
Tools execute on server
Results return to user
4. 📊 Data Flow

For adding a record:

User: "add John 30 Engineer"
↓
Agent: Formats SQL query
↓
MCP Tool: INSERT INTO people VALUES ('John', 30, 'Engineer')
↓
Server: Executes SQL
↓
Database: Stores data
↓
Response: "Successfully added John (age: 30, profession: Engineer)"
For reading records:

User: "show all records"
↓
Agent: Formats SELECT query
↓
MCP Tool: SELECT * FROM people
↓
Server: Fetches data
↓
Client: Formats table
↓
Response: Displays formatted table
🎯 Real-World Example
We often use Cursor IDE or Clause Desktop as MCP hosts, where the client relies on an external LLM (Claude Sonnet, GPT-4, etc.). While these tools are excellent, there are cases — especially when handling sensitive data — where fully secure and private MCP clients are essential. In our implementation, we created a MCP client powered by Local LLM which adds rows into a SQLite DB and selects rows from the SQLite (database management system) where:

📥 Adding Data
User requests to add a person
Agent formats SQL query
MCP tool executes query
Confirmation returns to user
2. 📤 Reading Data

User requests to view records
Agent creates SELECT query
MCP tool fetches data
Results format in nice table
🛠️ Technology Stack for MCP Implementation
1. 🐍 Python Framework & Libraries
Python 3.x — Core programming language
FastMCP — MCP server implementation
LangChain — AI/LLM framework integration
SQLite3 — Database management
asyncio — Asynchronous I/O operations
nest_asyncio — Nested event loop support
2. 🤖 AI/LLM Integration
Ollama — Local LLM model hosting(“llama3.2”)
3. 🗃️ Database Layer
SQLite — Lightweight database
  def init_db():
      conn = sqlite3.connect('demo.db')
      cursor = conn.cursor()
      # Schema creation...
4. 🔌 Communication Protocols
SSE (Server-Sent Events) — Real-time updates
MCP Protocol — Tool communication
  server_config = {
      "default": {
          "url": f"{mcp_server_url}/sse",
          "transport": "sse",
          "options": {...}
      }
  }
Code Implementation
Install required libraries
# 🔄 Core MCP and LangChain Packages
pip install langchain            # LangChain framework
pip install langchain-core      # Core LangChain functionalities
pip install langchain-community # Community tools and integrations
pip install langchain-mcp-adapters # MCP adapters for LangChain
pip install fastmcp            # FastMCP server implementation

# 🤖 LLM Integration
pip install langchain-ollama   # Ollama integration for LangChain

# 🔌 Networking and Async
pip install httpx             # Async HTTP client
pip install nest-asyncio      # Nested async support
2. server.py

import sqlite3
import argparse
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('sqlite-demo')

def init_db():
    conn = sqlite3.connect('demo.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            profession TEXT NOT NULL
        )
    ''')
    conn.commit()
    return conn, cursor

@mcp.tool()
def add_data(query: str) -> bool:
    """Add new data to the people table using a SQL INSERT query.

    Args:
        query (str): SQL INSERT query following this format:
            INSERT INTO people (name, age, profession)
            VALUES ('John Doe', 30, 'Engineer')
        
    Schema:
        - name: Text field (required)
        - age: Integer field (required)
        - profession: Text field (required)
        Note: 'id' field is auto-generated
    
    Returns:
        bool: True if data was added successfully, False otherwise
    
    Example:
        >>> query = '''
        ... INSERT INTO people (name, age, profession)
        ... VALUES ('Alice Smith', 25, 'Developer')
        ... '''
        >>> add_data(query)
        True
    """
    conn, cursor = init_db()
    try:
        print(f"\n\nExecuting add_data with query: {query}")
        cursor.execute(query)
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Error adding data: {e}")
        return False
    finally:
        conn.close()

@mcp.tool()
def read_data(query: str = "SELECT * FROM people") -> list:
    """Read data from the people table using a SQL SELECT query.

    Args:
        query (str, optional): SQL SELECT query. Defaults to "SELECT * FROM people".
            Examples:
            - "SELECT * FROM people"
            - "SELECT name, age FROM people WHERE age > 25"
            - "SELECT * FROM people ORDER BY age DESC"
    
    Returns:
        list: List of tuples containing the query results.
              For default query, tuple format is (id, name, age, profession)
    
    Example:
        >>> # Read all records
        >>> read_data()
        [(1, 'John Doe', 30, 'Engineer'), (2, 'Alice Smith', 25, 'Developer')]
        
        >>> # Read with custom query
        >>> read_data("SELECT name, profession FROM people WHERE age < 30")
        [('Alice Smith', 'Developer')]
    """
    conn, cursor = init_db()
    try:
        print(f"\n\nExecuting read_data with query: {query}")
        cursor.execute(query)
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Error reading data: {e}")
        return []
    finally:
        conn.close()



if __name__ == "__main__":
    # Start the server
    print("🚀Starting server... ")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_type", type=str, default="sse", choices=["sse", "stdio"],
    )

    args = parser.parse_args()
    # Only pass server_type to run()
    mcp.run(args.server_type)



3. langchain_client.py

import asyncio
import nest_asyncio
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import MCPTool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser, ReActJsonSingleInputOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import httpx
from langchain.tools import Tool
from typing import Optional, Any, Callable, Awaitable

# Enable nested asyncio for Jupyter-like environments
nest_asyncio.apply()

REACT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: {tool_names}
Action Input: the SQL query to execute
Observation: the result of the action
Thought: I now know the final answer
Final Answer: [The formatted table for read_data or success message for add_data]

For example:
Question: add John Doe 30 year old Engineer
Thought: I need to add a new person to the database
Action: add_data
Action Input: INSERT INTO people (name, age, profession) VALUES ('John Doe', 30, 'Engineer')
Observation: Data added successfully
Thought: I have successfully added the person
Final Answer: Successfully added John Doe (age: 30, profession: Engineer) to the database

Question: show all records
Thought: I need to retrieve all records from the database
Action: read_data
Action Input: SELECT * FROM people
Observation: [Formatted table with records]
Thought: I have retrieved all records
Final Answer: [The formatted table showing all records]

Begin!

Question: {input}
{agent_scratchpad}"""

class LangchainMCPClient:
    def __init__(self, mcp_server_url="http://127.0.0.1:8000"):
        print("Initializing LangchainMCPClient...")
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0.6,
            streaming=False  # Disable streaming for better compatibility
        )
        # Updated server configuration with shorter timeouts
        server_config = {
            "default": {
                "url": f"{mcp_server_url}/sse",
                "transport": "sse",
                "options": {
                    "timeout": 10.0,
                    "retry_connect": True,
                    "max_retries": 2,
                    "read_timeout": 5.0,
                    "write_timeout": 5.0
                }
            }
        }
        print(f"Connecting to MCP server at {mcp_server_url}...")
        self.mcp_client = MultiServerMCPClient(server_config)
        self.chat_history = []
        
        # System prompt for the agent
        self.SYSTEM_PROMPT = """You are an AI assistant that helps users interact with a database.
        You can add and read data from the database using the available tools.
        When adding data:
        1. Format the SQL query correctly: INSERT INTO people (name, age, profession) VALUES ('Name', Age, 'Profession')
        2. Make sure to use single quotes around text values
        3. Don't use quotes around numeric values
        
        When reading data:
        1. Use SELECT * FROM people for all records
        2. Use WHERE clause for filtering: SELECT * FROM people WHERE condition
        3. Present results in a clear, formatted way
        
        Always:
        1. Think through each step carefully
        2. Verify actions were successful
        3. Provide clear summaries of what was done"""

    async def check_server_connection(self):
        """Check if the MCP server is accessible"""
        base_url = self.mcp_client.connections["default"]["url"].replace("/sse", "")
        try:
            print(f"Testing connection to {base_url}...")
            async with httpx.AsyncClient(timeout=5.0) as client:  # Shorter timeout
                # Try the SSE endpoint directly
                sse_url = f"{base_url}/sse"
                print(f"Checking SSE endpoint at {sse_url}...")
                response = await client.get(sse_url, timeout=5.0)
                print(f"Got response: {response.status_code}")
                if response.status_code == 200:
                    print("SSE endpoint is accessible!")
                    return True
                
                print(f"Server responded with status code: {response.status_code}")
                return False
                
        except httpx.ConnectError:
            print(f"Could not connect to server at {base_url}")
            print("Please ensure the server is running and the port is correct")
            return False
        except httpx.ReadTimeout:
            print("Connection established but timed out while reading")
            print("This is normal for SSE connections - proceeding...")
            return True
        except Exception as e:
            print(f"Error connecting to MCP server: {type(e).__name__} - {str(e)}")
            return False

    async def initialize_agent(self):
        """Initialize the agent with tools and prompt template"""
        print("\nInitializing agent...")
        if not await self.check_server_connection():
            raise ConnectionError("Cannot connect to MCP server. Please ensure the server is running.")
            
        try:
            print("Getting available tools...")
            mcp_tools = await self.mcp_client.get_tools()
            
            # Verify tools are properly initialized
            print("Verifying tools...")
            for i, tool in enumerate(mcp_tools):
                print(f"\nTool {i}:")
                print(f"  Name: {tool.name if hasattr(tool, 'name') else 'No name'}")
                print(f"  Description: {tool.description if hasattr(tool, 'description') else 'No description'}")
                print(f"  Type: {type(tool)}")
                print(f"  Callable: {callable(tool)}")
                print(f"  Methods: {[method for method in dir(tool) if not method.startswith('_')]}")
                print(f"  Full tool: {tool.__dict__}")
                
                # Test call
                try:
                    print("  Testing tool call...")
                    if i == 0:
                        test_query = "INSERT INTO people (name, age, profession) VALUES ('Test', 30, 'Test')"
                    else:
                        test_query = "SELECT * FROM people"
                    result = await tool.ainvoke({"query": test_query})
                    print(f"  Test result: {result}")
                except Exception as e:
                    print(f"  Test error: {type(e).__name__} - {str(e)}")
            
            if len(mcp_tools) < 2:
                raise ValueError(f"Expected 2 tools, got {len(mcp_tools)}")
            
            # Create async wrapper functions with better error handling
            async def add_data_wrapper(query: str):
                try:
                    tool = mcp_tools[0]  # add_data tool
                    if not tool:
                        print("Tool 0 (add_data) not properly initialized")
                        return "Error: Add data tool not properly initialized"
                    print(f"Executing add_data with query: {query}")
                    # Clean up the query
                    query = query.strip().replace('\n', ' ').replace('  ', ' ')
                    # Fix common formatting issues
                    if "VALUES" in query:
                        parts = query.split("VALUES")
                        if len(parts) == 2:
                            values = parts[1].strip()
                            if values.startswith("(") and values.endswith(")"):
                                values = values[1:-1].split(",")
                                if len(values) == 3:
                                    name = values[0].strip().strip("'")
                                    age = values[1].strip()
                                    profession = values[2].strip().strip("'")
                                    query = f"INSERT INTO people (name, age, profession) VALUES ('{name}', {age}, '{profession}')"
                    # Call the tool using the async method
                    result = await tool.ainvoke({"query": query})
                    print(f"Add data result: {result}")
                    if result:
                        return "Data added successfully"  # Clear success message
                    return "Failed to add data"  # Clear failure message
                except Exception as e:
                    print(f"Error in add_data_wrapper: {type(e).__name__} - {str(e)}")
                    return f"Error adding data: {str(e)}"
                
            async def read_data_wrapper(query: str = "SELECT * FROM people"):
                try:
                    tool = mcp_tools[1]  # read_data tool
                    if not tool:
                        print("Tool 1 (read_data) not properly initialized")
                        return "Error: Read data tool not properly initialized"
                    print(f"Executing read_data with query: {query}")
                    # Clean up the query
                    query = query.strip().replace('\n', ' ').replace('  ', ' ')
                    # Call the tool using the async method
                    result = await tool.ainvoke({"query": query})
                    print(f"Read data result: {result}")
                    if not result:
                        return "No records found"
                    
                    # Format results in a table
                    records = []
                    for i in range(0, len(result), 4):
                        records.append({
                            'name': result[i+1],
                            'age': result[i+2],
                            'profession': result[i+3]
                        })
                    
                    # Create table header
                    output = [
                        f"Showing {len(records)} records:",
                        "",
                        "| Name          | Age | Profession       |",
                        "|---------------|-----|------------------|"
                    ]
                    
                    # Add each record
                    for record in records:
                        name = record['name'].ljust(13)
                        age = str(record['age']).ljust(5)
                        profession = record['profession'].ljust(16)
                        output.append(f"| {name} | {age} | {profession} |")
                    
                    return "\n".join(output)
                except Exception as e:
                    print(f"Error in read_data_wrapper: {type(e).__name__} - {str(e)}")
                    return f"Error reading data: {str(e)}"

            # Create Langchain tools with async functions
            self.tools = [
                Tool(
                    name="add_data",
                    description="Add a person to the database. Example: INSERT INTO people (name, age, profession) VALUES ('John Doe', 30, 'Engineer')",
                    func=lambda x: "Use async version",
                    coroutine=add_data_wrapper
                ),
                Tool(
                    name="read_data",
                    description="Read from the database. Example: SELECT * FROM people",
                    func=lambda x: "Use async version",
                    coroutine=read_data_wrapper
                )
            ]
            
            print(f"Found {len(self.tools)} tools")
            
            # Create the prompt template with system message
            system_message = SystemMessage(content=self.SYSTEM_PROMPT)
            human_message = HumanMessagePromptTemplate.from_template(REACT_TEMPLATE)
            prompt = ChatPromptTemplate.from_messages([
                system_message,
                human_message
            ]).partial(tool_names="add_data or read_data")
            
            # Create the agent with simpler configuration
            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            # Create the executor with better configuration
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=1,  # Only try once
                early_stopping_method="force",  # Stop after max_iterations
                return_intermediate_steps=True  # Ensure we get the steps
            )
            
            print("\nAvailable tools:")
            for tool in self.tools:
                print(f"- {tool.name}: {tool.description}")
                
        except Exception as e:
            print(f"\nError initializing agent: {e}")
            raise

    async def process_message(self, user_input: str) -> str:
        """Process a single user message and return the agent's response"""
        try:
            print("\nProcessing message:", user_input)
            # Execute the agent
            response = await self.agent_executor.ainvoke({
                "input": user_input,
                "chat_history": self.chat_history
            })
            
            print("\nRaw response:", response)
            final_result = None
            
            # Get the result from intermediate steps
            if isinstance(response, dict) and "intermediate_steps" in response:
                steps = response["intermediate_steps"]
                if steps and isinstance(steps[-1], tuple):
                    action, observation = steps[-1]
                    
                    # Handle add_data response
                    if "add_data" in str(action):
                        query = str(action.tool_input)
                        if "VALUES" in query:
                            values = query[query.find("VALUES")+7:].strip("() ")
                            name, age, profession = [v.strip().strip("'") for v in values.split(",")]
                            final_result = f"Successfully added {name} (age: {age}, profession: {profession}) to the database"
                    
                    # Handle read_data response
                    elif "read_data" in str(action):
                        if isinstance(observation, str) and "Showing" in observation:
                            final_result = observation  # Use the formatted table
                        else:
                            final_result = str(observation)  # Use any other read response
                    
                    # Use raw observation if no specific handling
                    if final_result is None:
                        final_result = str(observation)
                    
                    # Update response output and chat history
                    response["output"] = final_result
                    self.chat_history.extend([
                        HumanMessage(content=user_input),
                        AIMessage(content=final_result)
                    ])
                    
                    print("\nFinal result:", final_result)
                    return final_result
                
            return "Could not process the request. Please try again."
            
        except Exception as e:
            error_msg = f"Error processing message: {type(e).__name__} - {str(e)}\nPlease try rephrasing your request."
            print(f"\nError processing message: {type(e).__name__} - {str(e)}")
            print(f"Full error: {e.__dict__}")
            return error_msg

    async def interactive_chat(self):
        """Start an interactive chat session"""
        print("Chat session started. Type 'exit' to quit.")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                print("Ending chat session...")
                break
            
            response = await self.process_message(user_input)
            print("\nAgent:", response)

async def main():
    try:
        print("Starting Langchain MCP Client...")
        client = LangchainMCPClient()
        
        print("\nInitializing agent...")
        await client.initialize_agent()
        
        print("\nStarting interactive chat...")
        await client.interactive_chat()
        
    except ConnectionError as e:
        print(f"\nConnection Error: {e}")
        print("Please check that:")
        print("1. The MCP server is running (python server.py --server_type=sse)")
        print("2. The server URL is correct (http://127.0.0.1:8000)")
        print("3. The server is accessible from your machine")
    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__} - {str(e)}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 
⚠️ Important Notes:
System Requirements:
Python 3.8 or higher
SQLite3 (included with Python)
Sufficient disk space for LLM models
2. Ollama Setup:

Install Ollama separately from: Ollama
Pull required models:
ollama run llama3.2
🚀 Getting Started
# Start the MCP server
python server.py --server_type=sse

# Run the client
python langchain_client.py
1. 🚀 Initialization Phase
Server starts and registers tools
Client connects to server
Client discovers available tools
Agent initializes with tools
2. 💬 User Interaction Phase
When user types: “add Panama 55 year old as the SME”

Input goes to agent
Agent formats SQL query
Query passes to MCP tool
Tool executes on server
Result returns to user
Response Log
server.py log



langchain_client.py response log

(.venv) C:\Users\PLNAYAK\Documents\local_mcp_server>python langchain_client.py
Starting Langchain MCP Client...
Initializing LangchainMCPClient...
Connecting to MCP server at http://127.0.0.1:8000...

Initializing agent...

Initializing agent...
Testing connection to http://127.0.0.1:8000...
Checking SSE endpoint at http://127.0.0.1:8000/sse...
Connection established but timed out while reading
This is normal for SSE connections - proceeding...
Getting available tools...
Verifying tools...

Tool 0:
  Name: add_data
  Description: Add new data to the people table using a SQL INSERT query.

    Args:
        query (str): SQL INSERT query following this format:
            INSERT INTO people (name, age, profession)
            VALUES ('John Doe', 30, 'Engineer')

    Schema:
        - name: Text field (required)
        - age: Integer field (required)
        - profession: Text field (required)
        Note: 'id' field is auto-generated

    Returns:
        bool: True if data was added successfully, False otherwise

    Example:
        >>> query = '''
        ... INSERT INTO people (name, age, profession)
        ... VALUES ('Alice Smith', 25, 'Developer')
        ... '''
        >>> add_data(query)
        True

  Type: <class 'langchain_core.tools.structured.StructuredTool'>
  Callable: True
  Methods: ['InputType', 'OutputType', 'abatch', 'abatch_as_completed', 'ainvoke', 'args', 'args_schema', 'arun', 'as_tool', 'assign', 'astream', 'astream_events', 'astream_log', 'atransform', 'batch', 'batch_as_completed', 'bind', 'callback_manager', 'callbacks', 'config_schema', 'config_specs', 'configurable_alternatives', 'configurable_fields', 'construct', 'copy', 'coroutine', 'description', 'dict', 'from_function', 'from_orm', 'func', 'get_config_jsonschema', 'get_graph', 'get_input_jsonschema', 'get_input_schema', 'get_lc_namespace', 'get_name', 'get_output_jsonschema', 'get_output_schema', 'get_prompts', 'handle_tool_error', 'handle_validation_error', 'input_schema', 'invoke', 'is_lc_serializable', 'is_single_input', 'json', 'lc_attributes', 'lc_id', 'lc_secrets', 'map', 'metadata', 'model_computed_fields', 'model_config', 'model_construct', 'model_copy', 'model_dump', 'model_dump_json', 'model_extra', 'model_fields', 'model_fields_set', 'model_json_schema', 'model_parametrized_name', 'model_post_init', 'model_rebuild', 'model_validate', 'model_validate_json', 'model_validate_strings', 'name', 'output_schema', 'parse_file', 'parse_obj', 'parse_raw', 'pick', 'pipe', 'raise_deprecation', 'response_format', 'return_direct', 'run', 'schema', 'schema_json', 'stream', 'tags', 'to_json', 'to_json_not_implemented', 'tool_call_schema', 'transform', 'update_forward_refs', 'validate', 'verbose', 'with_alisteners', 'with_config', 'with_fallbacks', 'with_listeners', 'with_retry', 'with_types']
  Full tool: {'name': 'add_data', 'description': "Add new data to the people table using a SQL INSERT query.\n\n    Args:\n        query (str): SQL INSERT query following this format:\n            INSERT INTO people (name, age, profession)\n            VALUES ('John Doe', 30, 'Engineer')\n        \n    Schema:\n        - name: Text field (required)\n        - age: Integer field (required)\n        - profession: Text field (required)\n        Note: 'id' field is auto-generated\n    \n    Returns:\n        bool: True if data was added successfully, False otherwise\n    \n    Example:\n        >>> query = '''\n        ... INSERT INTO people (name, age, profession)\n        ... VALUES ('Alice Smith', 25, 'Developer')\n        ... '''\n        >>> add_data(query)\n        True\n    ", 'args_schema': {'properties': {'query': {'title': 'Query', 'type': 'string'}}, 'required': ['query'], 'title': 'add_dataArguments', 'type': 'object'}, 'return_direct': False, 'verbose': False, 'callbacks': None, 'callback_manager': None, 'tags': None, 'metadata': None, 'handle_tool_error': False, 'handle_validation_error': False, 'response_format': 'content_and_artifact', 'func': None, 'coroutine': <function convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x000002E4EF853600>}
  Testing tool call...
  Test result: true

Tool 1:
  Name: read_data
  Description: Read data from the people table using a SQL SELECT query.

    Args:
        query (str, optional): SQL SELECT query. Defaults to "SELECT * FROM people".
            Examples:
            - "SELECT * FROM people"
            - "SELECT name, age FROM people WHERE age > 25"
            - "SELECT * FROM people ORDER BY age DESC"

    Returns:
        list: List of tuples containing the query results.
              For default query, tuple format is (id, name, age, profession)

    Example:
        >>> # Read all records
        >>> read_data()
        [(1, 'John Doe', 30, 'Engineer'), (2, 'Alice Smith', 25, 'Developer')]

        >>> # Read with custom query
        >>> read_data("SELECT name, profession FROM people WHERE age < 30")
        [('Alice Smith', 'Developer')]

  Type: <class 'langchain_core.tools.structured.StructuredTool'>
  Callable: True
  Methods: ['InputType', 'OutputType', 'abatch', 'abatch_as_completed', 'ainvoke', 'args', 'args_schema', 'arun', 'as_tool', 'assign', 'astream', 'astream_events', 'astream_log', 'atransform', 'batch', 'batch_as_completed', 'bind', 'callback_manager', 'callbacks', 'config_schema', 'config_specs', 'configurable_alternatives', 'configurable_fields', 'construct', 'copy', 'coroutine', 'description', 'dict', 'from_function', 'from_orm', 'func', 'get_config_jsonschema', 'get_graph', 'get_input_jsonschema', 'get_input_schema', 'get_lc_namespace', 'get_name', 'get_output_jsonschema', 'get_output_schema', 'get_prompts', 'handle_tool_error', 'handle_validation_error', 'input_schema', 'invoke', 'is_lc_serializable', 'is_single_input', 'json', 'lc_attributes', 'lc_id', 'lc_secrets', 'map', 'metadata', 'model_computed_fields', 'model_config', 'model_construct', 'model_copy', 'model_dump', 'model_dump_json', 'model_extra', 'model_fields', 'model_fields_set', 'model_json_schema', 'model_parametrized_name', 'model_post_init', 'model_rebuild', 'model_validate', 'model_validate_json', 'model_validate_strings', 'name', 'output_schema', 'parse_file', 'parse_obj', 'parse_raw', 'pick', 'pipe', 'raise_deprecation', 'response_format', 'return_direct', 'run', 'schema', 'schema_json', 'stream', 'tags', 'to_json', 'to_json_not_implemented', 'tool_call_schema', 'transform', 'update_forward_refs', 'validate', 'verbose', 'with_alisteners', 'with_config', 'with_fallbacks', 'with_listeners', 'with_retry', 'with_types']
  Full tool: {'name': 'read_data', 'description': 'Read data from the people table using a SQL SELECT query.\n\n    Args:\n        query (str, optional): SQL SELECT query. Defaults to "SELECT * FROM people".\n            Examples:\n            - "SELECT * FROM people"\n            - "SELECT name, age FROM people WHERE age > 25"\n            - "SELECT * FROM people ORDER BY age DESC"\n    \n    Returns:\n        list: List of tuples containing the query results.\n              For default query, tuple format is (id, name, age, profession)\n    \n    Example:\n        >>> # Read all records\n        >>> read_data()\n        [(1, \'John Doe\', 30, \'Engineer\'), (2, \'Alice Smith\', 25, \'Developer\')]\n        \n        >>> # Read with custom query\n        >>> read_data("SELECT name, profession FROM people WHERE age < 30")\n        [(\'Alice Smith\', \'Developer\')]\n    ', 'args_schema': {'properties': {'query': {'default': 'SELECT * FROM people', 'title': 'Query', 'type': 'string'}}, 'title': 'read_dataArguments', 'type': 'object'}, 'return_direct': False, 'verbose': False, 'callbacks': None, 'callback_manager': None, 'tags': None, 'metadata': None, 'handle_tool_error': False, 'handle_validation_error': False, 'response_format': 'content_and_artifact', 'func': None, 'coroutine': <function convert_mcp_tool_to_langchain_tool.<locals>.call_tool at 0x000002E4EF8BE8E0>}
  Testing tool call...
  Test result: ['1', 'Test', '30', 'Test', '2', 'plaban nayak', '45', 'manager', '3', 'plaban nayak', '45', 'manager', '4', 'plaban nayak', '45', 'manager', '5', 'Test', '30', 'Test', '6', 'soma', '34', 'HR', '7', 'Test', '30', 'Test', '8', 'salmon', '35', 'accountant', '9', 'Test', '30', 'Test', '10', 'Kamilla', '24', 'Receptionist', '11', 'Test', '30', 'Test', '12', 'kishore', '27', 'facility manager', '13', 'Test', '30', 'Test', '14', 'Test', '30', 'Test', '15', 'Test', '30', 'Test', '16', 'Test', '30', 'Test', '17', 'Panama', '55', 'SME', '18', 'Test', '30', 'Test']
Found 2 tools

Available tools:
- add_data: Add a person to the database. Example: INSERT INTO people (name, age, profession) VALUES ('John Doe', 30, 'Engineer')
- read_data: Read from the database. Example: SELECT * FROM people

Starting interactive chat...
Chat session started. Type 'exit' to quit.

You: add Samiksha 30 years old Data Scientist

Processing message: add Samiksha 30 years old Data Scientist


> Entering new AgentExecutor chain...
Question: add Samiksha 30 years old Data Scientist
Thought: I need to add a new person to the database
Action: add_data
Action Input: INSERT INTO people (name, age, profession) VALUES ('Samiksha', 30, 'Data Scientist')Executing add_data with query: INSERT INTO people (name, age, profession) VALUES ('Samiksha', 30, 'Data Scientist')
Add data result: true
Data added successfully

> Finished chain.

Raw response: {'input': 'add Samiksha 30 years old Data Scientist', 'chat_history': [], 'output': 'Agent stopped due to iteration limit or time limit.', 'intermediate_steps': [(AgentAction(tool='add_data', tool_input="INSERT INTO people (name, age, profession) VALUES ('Samiksha', 30, 'Data Scientist')", log="Question: add Samiksha 30 years old Data Scientist\nThought: I need to add a new person to the database\nAction: add_data\nAction Input: INSERT INTO people (name, age, profession) VALUES ('Samiksha', 30, 'Data Scientist')"), 'Data added successfully')]}

Final result: Successfully added Samiksha (age: 30, profession: Data Scientist) to the database

Agent: Successfully added Samiksha (age: 30, profession: Data Scientist) to the database

Processing message: Show all records


> Entering new AgentExecutor chain...
Question: show all records
Thought: I need to retrieve all records from the database
Action: read_data
Action Input: SELECT * FROM peopleExecuting read_data with query: SELECT * FROM people
Read data result: ['1', 'Test', '30', 'Test', '2', 'plaban nayak', '45', 'manager', '3', 'plaban nayak', '45', 'manager', '4', 'plaban nayak', '45', 'manager', '5', 'Test', '30', 'Test', '6', 'soma', '34', 'HR', '7', 'Test', '30', 'Test', '8', 'salmon', '35', 'accountant', '9', 'Test', '30', 'Test', '10', 'Kamilla', '24', 'Receptionist', '11', 'Test', '30', 'Test', '12', 'kishore', '27', 'facility manager', '13', 'Test', '30', 'Test', '14', 'Test', '30', 'Test', '15', 'Test', '30', 'Test', '16', 'Test', '30', 'Test', '17', 'Panama', '55', 'SME', '18', 'Test', '30', 'Test', '19', 'Samiksha', '30', 'Data Scientist']
Showing 19 records:

| Name          | Age | Profession       |
|---------------|-----|------------------|
| Test          | 30    | Test             |
| plaban nayak  | 45    | manager          |
| plaban nayak  | 45    | manager          |
| plaban nayak  | 45    | manager          |
| Test          | 30    | Test             |
| soma          | 34    | HR               |
| Test          | 30    | Test             |
| salmon        | 35    | accountant       |
| Test          | 30    | Test             |
| Kamilla       | 24    | Receptionist     |
| Test          | 30    | Test             |
| kishore       | 27    | facility manager |
| Test          | 30    | Test             |
| Test          | 30    | Test             |
| Test          | 30    | Test             |
| Test          | 30    | Test             |
| Panama        | 55    | SME              |
| Test          | 30    | Test             |
| Samiksha      | 30    | Data Scientist   |

> Finished chain.

Raw response: {'input': 'Show all records', 'chat_history': [HumanMessage(content='add Samiksha 30 years old Data Scientist', additional_kwargs={}, response_metadata={}), AIMessage(content='Successfully added Samiksha (age: 30, profession: Data Scientist) to the database', additional_kwargs={}, response_metadata={})], 'output': 'Agent stopped due to iteration limit or time limit.', 'intermediate_steps': [(AgentAction(tool='read_data', tool_input='SELECT * FROM people', log='Question: show all records\nThought: I need to retrieve all records from the database\nAction: read_data\nAction Input: SELECT * FROM people'), 'Showing 19 records:\n\n| Name          | Age | Profession       |\n|---------------|-----|------------------|\n| Test          | 30    | Test             |\n| plaban nayak  | 45    | manager          |\n| plaban nayak  | 45    | manager          |\n| plaban nayak  | 45    | manager          |\n| Test          | 30    | Test             |\n| soma          | 34    | HR               |\n| Test          | 30    | Test             |\n| salmon        | 35    | accountant       |\n| Test          | 30    | Test             |\n| Kamilla       | 24    | Receptionist     |\n| Test          | 30    | Test             |\n| kishore       | 27    | facility manager |\n| Test          | 30    | Test             |\n| Test          | 30    | Test             |\n| Test          | 30    | Test             |\n| Test          | 30    | Test             |\n| Panama        | 55    | SME              |\n| Test          | 30    | Test             |\n| Samiksha      | 30    | Data Scientist   |')]}

Final result: Showing 19 records:

| Name          | Age | Profession       |
|---------------|-----|------------------|
| Test          | 30    | Test             |
| plaban nayak  | 45    | manager          |
| plaban nayak  | 45    | manager          |
| plaban nayak  | 45    | manager          |
| Test          | 30    | Test             |
| soma          | 34    | HR               |
| Test          | 30    | Test             |
| salmon        | 35    | accountant       |
| Test          | 30    | Test             |
| Kamilla       | 24    | Receptionist     |
| Test          | 30    | Test             |
| kishore       | 27    | facility manager |
| Test          | 30    | Test             |
| Test          | 30    | Test             |
| Test          | 30    | Test             |
| Test          | 30    | Test             |
| Panama        | 55    | SME              |
| Test          | 30    | Test             |
| Samiksha      | 30    | Data Scientist   |

Agent: Showing 19 records:

| Name          | Age | Profession       |
|---------------|-----|------------------|
| Test          | 30    | Test             |
| plaban nayak  | 45    | manager          |
| plaban nayak  | 45    | manager          |
| plaban nayak  | 45    | manager          |
| Test          | 30    | Test             |
| soma          | 34    | HR               |
| Test          | 30    | Test             |
| salmon        | 35    | accountant       |
| Test          | 30    | Test             |
| Kamilla       | 24    | Receptionist     |
| Test          | 30    | Test             |
| kishore       | 27    | facility manager |
| Test          | 30    | Test             |
| Test          | 30    | Test             |
| Test          | 30    | Test             |
| Test          | 30    | Test             |
| Panama        | 55    | SME              |
| Test          | 30    | Test             |
| Samiksha      | 30    | Data Scientist   |

You:

🎯 Benefits of This Workflow
🔌 Modularity
Easy to add new tools
Simple to modify existing tools
Clean separation of concerns
2. 🚀 Scalability

Async operations
Connection pooling
Resource management
3. 👥 User Experience

Natural language input
Formatted output
Error handling
4. 🛠️ Maintainability

Clear code structure
Separated components
Easy debugging
This workflow creates a robust, scalable, and user-friendly system for database operations through natural language commands! 🎉

🔮 Future Possibilities
🎨 Creative Applications
AI art generation
Natural language processing
Autonomous agents
2. 🏢 Enterprise Use Cases

Database management
Customer service
Process automation
3. 🔬 Research Applications

Model comparison
Tool composition
Agent development
🎉 Conclusion
MCP represents a significant step forward in AI integration, making it easier than ever to connect models with applications. Whether we are building a chat interface, a database manager, or an autonomous agent, MCP provides the foundation you need to succeed!