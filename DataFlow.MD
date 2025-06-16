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
