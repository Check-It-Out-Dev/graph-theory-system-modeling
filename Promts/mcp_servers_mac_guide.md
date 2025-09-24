# Complete MCP Server Installation Guide for Mac

This guide covers installing three popular MCP servers: **Sequential Thinking**, **Neo4j**, and **Filesystem** servers for Claude Code.

## Prerequisites

First, ensure you have Node.js installed:

```bash
# Check if Node.js is installed
node --version
npm --version

# If not installed, install via Homebrew
brew install node

# Or download from https://nodejs.org
```

## Step 1: Install Claude Code MCP Servers

### Method 1: Using `claude mcp add` Commands

```bash
# Install Sequential Thinking server
claude mcp add sequential-thinking --scope user -- npx -y @modelcontextprotocol/server-sequential-thinking

# Install Neo4j server
claude mcp add neo4j-cypher --scope user -- mcp-neo4j-cypher

# Install Filesystem server
claude mcp add filesystem --scope user -- npx -y @modelcontextprotocol/server-filesystem ~/Documents ~/Projects
```

### Method 2: Manual Installation (Recommended)

If the CLI commands give you issues, install the packages manually first:

```bash
# Pre-install the MCP server packages
npx -y @modelcontextprotocol/server-sequential-thinking
npx -y mcp-neo4j-cypher  
npx -y @modelcontextprotocol/server-filesystem
```

## Step 2: Configure the ~/.claude.json File

### Create or open the configuration file:

```bash
# Create the file if it doesn't exist
touch ~/.claude.json

# Open it in your default editor
open ~/.claude.json

# Or use VS Code
code ~/.claude.json

# Or use nano
nano ~/.claude.json
```

### Add the complete configuration:

```json
{
  "mcpServers": {
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
      "type": "stdio"
    },
    "neo4j-cypher": {
      "command": "mcp-neo4j-cypher",
      "args": ["--transport", "stdio"],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "your-password",
        "NEO4J_DATABASE": "neo4j"
      }
    }
  },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y", 
        "@modelcontextprotocol/server-filesystem",
        "/Users/YOUR_USERNAME/Documents",
        "/Users/YOUR_USERNAME/Projects"
      ],
      "type": "stdio"
    }
  }
}
```

**Important**: Replace `YOUR_USERNAME` with your actual Mac username, and update the Neo4j credentials.

### To find your username:
```bash
echo $USER
```

### Complete example with real paths:
```json
{
  "mcpServers": {
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
      "type": "stdio"
    },
    "neo4j-cypher": {
      "command": "npx",
      "args": ["-y", "mcp-neo4j-cypher"],
      "type": "stdio",
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "password123"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y", 
        "@modelcontextprotocol/server-filesystem",
        "/Users/john/Documents",
        "/Users/john/Projects",
        "/Users/john/Downloads"
      ],
      "type": "stdio"
    }
  }
}
```

## Step 3: Neo4j Setup (if needed)

If you don't have Neo4j installed locally:

```bash
# Install Neo4j via Homebrew
brew install neo4j

# Start Neo4j
brew services start neo4j

# Or run once
neo4j console

# Access Neo4j Browser at http://localhost:7474
# Default credentials: neo4j/neo4j (you'll be prompted to change)
```

## Step 4: Verify Installation

### Restart Claude Code:
```bash
# If Claude Code is running, restart it completely
```

### Check MCP server status:
```bash
# In Claude Code terminal, run:
/mcp

# Or list servers:
claude mcp list
```

### Test each server:
```bash
# Test individual servers
claude mcp get sequential-thinking
claude mcp get neo4j-cypher
claude mcp get filesystem
```

## Step 5: Test Functionality

Once Claude Code is restarted, you can test each server:

### Sequential Thinking:
Ask Claude: *"Use sequential thinking to plan a web application architecture"*

### Neo4j:
Ask Claude: *"Show me the schema of my Neo4j database"* or *"Query my graph database for..."*

### Filesystem:
Ask Claude: *"List the files in my Documents folder"* or *"Create a new project structure"*

## Server Capabilities

### Sequential Thinking Server
- Enables structured, reflective thinking processes
- Maintains context across extended reasoning chains
- Perfect for complex problem-solving, architectural decisions, and debugging analysis

### Neo4j Server
- Provides database schema extraction and Cypher query generation
- Graph database schema analysis
- Natural language to Cypher translation
- Knowledge graph management

### Filesystem Server
- Direct access to local file system
- Read, write, and organize files
- Perfect for project management and file operations

## Troubleshooting

### Common Issues:

1. **Permission errors**: Ensure your user has read/write access to the specified directories
2. **Neo4j connection issues**: Verify Neo4j is running and credentials are correct
3. **NPX not found**: Ensure Node.js and npm are properly installed and in your PATH

### Debug individual servers:
```bash
# Test servers manually
npx -y @modelcontextprotocol/server-sequential-thinking
npx -y mcp-neo4j-cypher
npx -y @modelcontextprotocol/server-filesystem ~/Documents
```

### Check logs:
Look for error messages in the Claude Code terminal when servers start up.

## Management Commands

```bash
# List installed servers
claude mcp list

# Remove a server
claude mcp remove <server-name>

# Test server connection
claude mcp get <server-name>

# Check status within Claude Code
/mcp
```

## Best Practices

1. **Start Simple**: Don't add too many MCP servers at once as it affects performance
2. **Use Scopes**: Specify `--scope user` for personal configurations
3. **Direct Config**: Edit configuration files directly for complex setups with many parameters
4. **Regular Updates**: Restart Claude Code after configuration changes
5. **Backup**: Keep backups of your configuration files

## Configuration File Locations

- **Claude Code**: `~/.claude.json` (recommended)
- **Claude Desktop**: `~/Library/Application Support/Claude/claude_desktop_config.json`

After completing these steps, all three MCP servers should be active and Claude Code will have enhanced capabilities for structured thinking, graph database access, and filesystem operations.