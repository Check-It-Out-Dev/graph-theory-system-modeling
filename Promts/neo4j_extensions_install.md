# Neo4j Community Edition Free Extensions Installation Script

## Important Note for Homebrew Users

**Homebrew installations of Neo4j typically do NOT include plugins in the `labs` directory.** All extensions must be downloaded manually from their respective GitHub repositories. This guide provides the correct download links and installation steps.

## Prerequisites Check

```bash
# Verify Neo4j is installed via Homebrew
brew list | grep neo4j

# Get Neo4j version
neo4j version

# Set Neo4j Home
export NEO4J_HOME=$(brew --prefix neo4j)
echo "Neo4j Home: $NEO4J_HOME"

# Check directory structure
echo "Checking directory structure..."
ls -la "$NEO4J_HOME/"

# Check if labs directory exists and has content
if [ -d "$NEO4J_HOME/labs" ]; then
    echo "Labs directory contents:"
    ls -la "$NEO4J_HOME/labs/"
else
    echo "❌ No labs directory found (typical for Homebrew installations)"
fi

# Check if products directory exists
if [ -d "$NEO4J_HOME/products" ]; then
    echo "Products directory contents:"
    ls -la "$NEO4J_HOME/products/"
else
    echo "❌ No products directory found (typical for Homebrew installations)"
fi

echo "✅ All extensions must be downloaded manually"
```

## Step 1: Stop Neo4j (if running)

```bash
# Stop Neo4j service
neo4j stop

# Or if running as service
brew services stop neo4j
```

## Step 2: Install APOC Core (Download Required)

```bash
# Navigate to Neo4j installation
cd "$NEO4J_HOME"

# Create plugins directory if it doesn't exist
mkdir -p plugins
cd plugins

# Check Neo4j version first
NEO4J_VERSION=$(neo4j version | grep -o '[0-9]\{4\}\.[0-9]\{2\}' | head -1)
echo "Neo4j version detected: $NEO4J_VERSION"

# Download APOC Core matching your Neo4j version
case $NEO4J_VERSION in
    "2025.08")
        wget https://github.com/neo4j/apoc/releases/download/2025.08.0/apoc-2025.08.0-core.jar
        ;;
    "5.26")
        wget https://github.com/neo4j/apoc/releases/download/5.26.0/apoc-5.26.0-core.jar
        ;;
    "5.25")
        wget https://github.com/neo4j/apoc/releases/download/5.25.1/apoc-5.25.1-core.jar
        ;;
    *)
        echo "Please download APOC manually from: https://github.com/neo4j/apoc/releases"
        echo "Match the first two version numbers (e.g., Neo4j $NEO4J_VERSION needs APOC $NEO4J_VERSION.x)"
        ;;
esac

cd ..
echo "✅ APOC Core downloaded"
```

## Step 3: Install Graph Data Science Community Edition

```bash
cd "$NEO4J_HOME/plugins"

# Download GDS Community Edition based on Neo4j version
case $NEO4J_VERSION in
    "2025.08")
        echo "Downloading GDS for Neo4j 2025.08..."
        wget https://github.com/neo4j/graph-data-science/releases/download/2.12.0/neo4j-graph-data-science-2.12.0.jar
        ;;
    "5.26")
        echo "Downloading GDS for Neo4j 5.26..."
        wget https://github.com/neo4j/graph-data-science/releases/download/2.11.1/neo4j-graph-data-science-2.11.1.jar
        ;;
    "5.25")
        echo "Downloading GDS for Neo4j 5.25..."
        wget https://github.com/neo4j/graph-data-science/releases/download/2.10.1/neo4j-graph-data-science-2.10.1.jar
        ;;
    *)
        echo "Please download GDS manually from: https://github.com/neo4j/graph-data-science/releases"
        echo "Check compatibility matrix: https://neo4j.com/docs/graph-data-science/current/installation/"
        ;;
esac

echo "✅ Graph Data Science Community Edition downloaded"
```

## Step 4: Install GenAI Plugin (if available)

```bash
cd "$NEO4J_HOME"

# Check if GenAI plugin exists in products directory
if [ -f "products/neo4j-genai-"*.jar ]; then
    echo "Copying GenAI plugin from products directory..."
    cp products/neo4j-genai-*.jar plugins/
    echo "✅ GenAI Plugin installed"
elif [ -d "products" ]; then
    echo "⚠️  GenAI plugin not found in products directory"
    echo "Available in products:"
    ls -la products/ 2>/dev/null || echo "No products directory found"
else
    echo "ℹ️  No products directory found in Homebrew installation"
    echo "GenAI plugin may not be included in Homebrew version"
fi

# Note: GenAI plugin is often only available in Enterprise Edition or newer versions
echo "Note: GenAI plugin is typically available in Enterprise Edition or Neo4j 5.13+"
```

## Step 5: Configure Neo4j Settings

```bash
# Add necessary configuration to neo4j.conf
NEO4J_CONF="$NEO4J_HOME/conf/neo4j.conf"

echo "Configuring Neo4j settings..."

# Enable procedures (add to neo4j.conf if not present)
if ! grep -q "dbms.security.procedures.unrestricted" "$NEO4J_CONF"; then
    echo "dbms.security.procedures.unrestricted=apoc.*,gds.*" >> "$NEO4J_CONF"
fi

# Enable APOC export/import
if ! grep -q "apoc.export.file.enabled" "$NEO4J_CONF"; then
    echo "apoc.export.file.enabled=true" >> "$NEO4J_CONF"
    echo "apoc.import.file.enabled=true" >> "$NEO4J_CONF"
fi

echo "✅ Configuration updated"
```

## Step 6: Verify Installation

```bash
# List installed plugins
echo "Installed plugins:"
ls -la "$NEO4J_HOME/plugins/"

# Start Neo4j
echo "Starting Neo4j..."
neo4j start

# Wait for Neo4j to start
sleep 10

# Verify APOC installation
echo "Verifying APOC installation..."
echo "RETURN apoc.version()" | cypher-shell -u neo4j -p your-password

# Verify GDS installation
echo "Verifying GDS installation..."
echo "RETURN gds.version()" | cypher-shell -u neo4j -p your-password

echo "🎉 All free extensions installed successfully!"
```

## Alternative: One-Line Installation (Homebrew Version)

```bash
# For Neo4j 2025.08 (adjust versions as needed)
NEO4J_HOME=$(brew --prefix neo4j) && \
cd "$NEO4J_HOME" && \
mkdir -p plugins && \
cd plugins && \
wget -q https://github.com/neo4j/apoc/releases/download/2025.08.0/apoc-2025.08.0-core.jar && \
wget -q https://github.com/neo4j/graph-data-science/releases/download/2.12.0/neo4j-graph-data-science-2.12.0.jar && \
cd .. && \
echo "dbms.security.procedures.unrestricted=apoc.*,gds.*" >> conf/neo4j.conf && \
echo "apoc.export.file.enabled=true" >> conf/neo4j.conf && \
echo "apoc.import.file.enabled=true" >> conf/neo4j.conf && \
neo4j restart && \
echo "🎉 Extensions installed and Neo4j restarted!"
```

## Troubleshooting

### Issue: Permission denied
```bash
# Fix permissions
sudo chown -R $(whoami) $(brew --prefix neo4j)
```

### Issue: Version mismatch
```bash
# Check Neo4j version
neo4j version

# Download matching versions:
# For Neo4j 2025.08.x → APOC 2025.08.x, GDS 2.12.x
# For Neo4j 5.26.x → APOC 5.26.x, GDS 2.11.x
```

### Issue: Plugin not loading
```bash
# Check Neo4j logs
tail -f $(brew --prefix neo4j)/logs/neo4j.log

# Verify plugin files exist
ls -la $(brew --prefix neo4j)/plugins/

# Check configuration
grep -i "procedures.unrestricted" $(brew --prefix neo4j)/conf/neo4j.conf
```

## Available Free Extensions Summary

| Extension | Description | Installation | Status |
|-----------|-------------|--------------|---------|
| **APOC Core** | 400+ procedures for data integration, algorithms, utilities | Copy from labs/ or download | ✅ FREE |
| **Graph Data Science CE** | 65+ graph algorithms, ML capabilities (4 CPU limit) | Download from GitHub | ✅ FREE |
| **GenAI Plugin** | AI/ML integration, vector operations | Copy from products/ | ✅ FREE |

## Paid Extensions (Not Free)

These require Neo4j Enterprise Edition and/or separate licenses:
- ❌ **Neo4j Bloom** (Graph visualization)
- ❌ **APOC Extended** (Additional procedures)
- ❌ **Graph Data Science Enterprise** (Unlimited CPU cores)

## Testing Your Installation

After installation, test each extension:

```cypher
// Test APOC
RETURN apoc.version();

// Test GDS  
RETURN gds.version();

// List all available procedures
CALL dbms.procedures() YIELD name, description 
WHERE name STARTS WITH 'apoc' OR name STARTS WITH 'gds' 
RETURN name, description LIMIT 10;

// Test APOC functionality
CALL apoc.help('apoc');

// Test GDS functionality  
CALL gds.list();
```

This setup gives you the most comprehensive free Neo4j experience with powerful graph algorithms, data processing capabilities, and AI integration tools.