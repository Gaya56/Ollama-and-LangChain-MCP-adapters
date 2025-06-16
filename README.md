# Ollama and LangChain MCP Adapters

A local Model Context Protocol (MCP) implementation that demonstrates how to build a secure, private MCP client using Ollama for local LLM hosting and LangChain for AI framework integration. This project showcases database management through MCP tools with a SQLite backend.

## Introduction

This project implements a complete MCP (Model Context Protocol) ecosystem that enables AI models to interact with databases through standardized tools. Unlike cloud-based solutions like Cursor IDE or Claude Desktop, this implementation runs entirely locally, making it perfect for handling sensitive data that requires privacy and security.

The system consists of:
- **MCP Server**: Exposes database operations as standardized tools
- **MCP Client**: Connects to the server using LangChain and Ollama
- **Local LLM**: Uses Ollama to run models locally (llama3.2)
- **Database**: SQLite database for storing and retrieving people records

## How It Works

### Architecture Overview

```
User Input → LangChain Agent → MCP Tools → SQLite Database
     ↑                                           ↓
Local LLM ← Formatted Response ← Query Results ←
```

### Key Components

1. **MCP Server (`server.py`)**
   - Implements FastMCP server with SQLite integration
   - Exposes two main tools:
     - `add_data`: Adds new records to the people table
     - `read_data`: Retrieves records from the database
   - Handles SQL query execution and error management

2. **MCP Client (`langchain_client.py`)**
   - Creates a LangChain agent with access to MCP tools
   - Uses local Ollama LLM for natural language processing
   - Provides interactive chat interface for database operations
   - Implements ReAct (Reasoning and Acting) pattern for tool usage

3. **Database Schema**
   ```sql
   CREATE TABLE people (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       name TEXT NOT NULL,
       age INTEGER NOT NULL,
       profession TEXT NOT NULL
   );
   ```

### Workflow Example

When you ask: *"Add John Doe, 30 years old, Engineer"*

1. **User Input** → LangChain Agent processes the request
2. **Agent Reasoning** → Determines it needs to use `add_data` tool
3. **SQL Generation** → Creates: `INSERT INTO people (name, age, profession) VALUES ('John Doe', 30, 'Engineer')`
4. **MCP Tool Execution** → Server executes the SQL query
5. **Response** → Returns success confirmation to user

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Ollama installed and running
- Git (for cloning the repository)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Ollama-and-LangChain-MCP-adapters
   ```

2. **Install Ollama** (if not already installed)
   ```bash
   # Visit https://ollama.com for installation instructions
   # For Linux/macOS:
   curl -fsSL https://ollama.com/install.sh | sh
   ```

3. **Pull the required LLM model**
   ```bash
   ollama pull llama3.2
   ```

4. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the MCP Server** (in one terminal)
   ```bash
   python server.py --server_type=sse
   ```
   
   You should see: `🚀Starting server...`

2. **Start the MCP Client** (in another terminal)
   ```bash
   python langchain_client.py
   ```

3. **Interact with the system**
   ```
   Chat session started. Type 'exit' to quit.
   
   You: add Alice Smith, 28 years old, Developer
   Assistant: Successfully added Alice Smith (age: 28, profession: Developer) to the database
   
   You: show all records
   Assistant: [Displays formatted table with all records]
   
   You: exit
   ```

### Example Commands

- **Adding records**:
  - "add John Doe, 30 years old, Engineer"
  - "insert Bob Wilson, 45, Manager"
  - "add new person: Sarah Johnson, age 27, profession Data Scientist"

- **Reading records**:
  - "show all records"
  - "list all people"
  - "display everyone in the database"
  - "show people older than 30"

### Troubleshooting

**Connection Issues**:
- Ensure Ollama is running: `ollama serve`
- Check if the MCP server is accessible: `curl http://127.0.0.1:8000`
- Verify the correct Python environment is activated

**Missing Dependencies**:
```bash
pip install langchain langchain-core langchain-community langchain-mcp-adapters fastmcp langchain-ollama httpx nest-asyncio
```

## Project Structure

```
Ollama-and-LangChain-MCP-adapters/
├── server.py                 # MCP server implementation
├── langchain_client.py       # MCP client with LangChain integration
├── requirements.txt          # Python dependencies
├── demo.db                   # SQLite database (created automatically)
├── communication_protocol.py # Server configuration
├── DataFlow.MD              # Detailed workflow documentation
├── article_reference.md     # Background information and references
└── README.md                # This file
```

## Reference

For detailed background information, architectural insights, and the theoretical foundation of this implementation, please refer to [`article_reference.md`](article_reference.md). This file contains:

- In-depth explanation of MCP (Model Context Protocol)
- Advantages and use cases for local MCP implementations
- Detailed architecture breakdown
- Technology stack rationale
- Real-world examples and workflows

The article provides comprehensive context for understanding why and how this local MCP solution was developed, especially for scenarios requiring data privacy and security.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.