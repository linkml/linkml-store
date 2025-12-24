# OpenMetadata Integration with LinkML-Store

## Vision: Organized Discovery + Powerful Operations

This document outlines how **OpenMetadata** and **LinkML-Store** can work together to provide a comprehensive data platform that combines enterprise-scale data organization with flexible, AI-ready data operations.

### The Core Concept

- **OpenMetadata**: Organize and catalog large, heterogeneous datasets across your entire organization
- **LinkML-Store**: Hook into cataloged datasets to perform complex queries, AI-powered search, and data operations

## Why This Combination?

### The Challenge

Modern organizations face a common problem:
- Data is scattered across multiple systems (databases, data lakes, APIs, files)
- Different teams use different technologies (SQL databases, document stores, graph databases, search engines)
- Finding relevant data requires knowing where to look and how to access it
- Performing complex operations requires understanding multiple query languages and APIs
- AI/ML workflows need unified access to heterogeneous data sources

### The Solution: Two-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenMetadata Layer                        │
│         (Discovery, Cataloging, Governance)                  │
│                                                              │
│  "What data exists? Where is it? Who owns it?               │
│   What does it mean? Can I access it?"                      │
└───────────────────────┬─────────────────────────────────────┘
                        │ catalogs & organizes
                        │
┌───────────────────────┴─────────────────────────────────────┐
│                  LinkML-Store Layer                          │
│         (Query, Operations, AI Integration)                  │
│                                                              │
│  "Give me all records matching X. Find semantically         │
│   similar items. Infer missing values. Export results."     │
└───────────────────────┬─────────────────────────────────────┘
                        │ abstracts & operates on
                        │
┌───────────────────────┴─────────────────────────────────────┐
│              Heterogeneous Data Backends                     │
│  DuckDB │ MongoDB │ PostgreSQL │ Neo4j │ Solr │ Files │...  │
└─────────────────────────────────────────────────────────────┘
```

## OpenMetadata: The Organizational Layer

### What OpenMetadata Provides

**OpenMetadata** is an open-source unified metadata platform that serves as your organization's data catalog:

1. **Discovery**: Find data assets across 100+ different systems
2. **Cataloging**: Centralized metadata for databases, tables, dashboards, pipelines, ML models
3. **Governance**: Data ownership, access control, quality rules, compliance tracking
4. **Lineage**: Track data flow from source to consumption with column-level lineage
5. **Collaboration**: Teams can document, tag, and discuss data assets

### OpenMetadata's Strengths

- **Broad connector ecosystem**: Ingests metadata from diverse sources automatically
- **Unified metadata model**: Common representation across different systems
- **Enterprise features**: SSO, RBAC, audit logs, data quality monitoring
- **Team collaboration**: Glossaries, tags, descriptions, conversations
- **Standards-based**: Uses JSON Schema for entity definitions

### Example: Finding Data with OpenMetadata

A data scientist needs customer behavior data:

1. **Search** in OpenMetadata: "customer purchase behavior"
2. **Discover** relevant datasets across multiple systems:
   - `analytics_db.customers` table (PostgreSQL)
   - `user_events` collection (MongoDB)
   - `purchase_graph` (Neo4j)
   - `customer_features.parquet` (Data Lake)
3. **Understand** through metadata:
   - Schema definitions and descriptions
   - Data quality scores
   - Lineage showing data sources
   - Ownership and access policies
4. **Access** information through OpenMetadata's catalog

But then what? How do you actually query and work with this data?

## LinkML-Store: The Operations Layer

### What LinkML-Store Provides

**LinkML-Store** provides a unified interface for data operations across heterogeneous backends:

1. **Unified Query Interface**: Single API for querying different database types
2. **Schema Validation**: LinkML schemas ensure data quality and consistency
3. **AI-Powered Search**: Semantic search using LLM embeddings (without vector databases)
4. **Advanced Operations**: Complex queries, aggregations, inference, data enrichment
5. **Multi-Backend Support**: Work with DuckDB, MongoDB, Neo4j, Solr, and more through one interface

### LinkML-Store's Strengths

- **Backend abstraction**: Write queries once, run on multiple database types
- **AI integration**: Built-in LLM embeddings, RAG, and inference engines
- **Schema-driven**: LinkML schemas provide structure and validation
- **Flexible**: Switch backends without changing application code
- **Developer-friendly**: Python API, CLI tools, REST API

### Example: Operating on Data with LinkML-Store

Continuing the example above, the data scientist now wants to:

1. **Query** across multiple backends with a unified interface
2. **Search** semantically: "find customers likely to churn"
3. **Infer** missing values using ML models
4. **Aggregate** results from different sources
5. **Export** in various formats for analysis

```python
from linkml_store import Client

# Connect to data sources cataloged in OpenMetadata
client = Client()

# Query PostgreSQL customers
pg_db = client.attach_database("postgresql://analytics_db")
customers = pg_db.get_collection("customers")

# Query MongoDB events
mongo_db = client.attach_database("mongodb://events_db")
events = mongo_db.get_collection("user_events")

# Semantic search across customers (using LLM embeddings)
at_risk_customers = customers.search(
    "customers with declining engagement and purchase frequency"
)

# Complex query with aggregation
high_value = customers.query(
    where={"lifetime_value": {"$gt": 1000}},
    facet_on=["region", "segment"]
)

# Inference: predict missing attributes
predictor = customers.attach_predictor("sklearn")
completed = predictor.derive(partial_customer_data)
```

## How They Work Together

### Workflow: From Discovery to Analysis

#### 1. **Organize with OpenMetadata**

First, catalog your data landscape:

```
# In OpenMetadata
- Ingest metadata from all data sources
- Document datasets with descriptions and owners
- Define data quality rules
- Set up access policies
- Create glossaries and tag data assets
```

#### 2. **Discover with OpenMetadata**

Users find what they need:

```
User searches: "customer transaction data"
↓
OpenMetadata returns:
  - transactions table in postgres (analytical)
  - transactions collection in mongodb (operational)
  - transaction_graph in neo4j (fraud detection)

User views:
  - Schema: {customer_id, amount, timestamp, merchant, ...}
  - Quality: 99.2% complete, updated daily
  - Lineage: Source → ETL → These tables → Dashboards
  - Access: Request access from data-team@org.com
```

#### 3. **Connect with LinkML-Store**

Configure linkml-store to access discovered resources:

```yaml
# linkml-store-config.yaml
databases:
  analytics:
    handle: postgresql://analytics_db
    schema_location: schemas/transaction.yaml
    collections:
      transactions:
        alias: transactions
        type: Transaction

  operational:
    handle: mongodb://operational_db
    collections:
      transactions:
        alias: live_transactions
        type: Transaction
```

#### 4. **Query with LinkML-Store**

Perform operations across sources:

```python
# Unified interface to heterogeneous data
from linkml_store import Client

client = Client.from_config("linkml-store-config.yaml")

# Query analytical database (PostgreSQL)
analytical_txns = client.databases["analytics"].collections["transactions"]
results = analytical_txns.query(where={"amount": {"$gt": 1000}})

# Query operational database (MongoDB)
live_txns = client.databases["operational"].collections["live_transactions"]
recent = live_txns.find({"timestamp": {"$gte": "2024-01-01"}})

# Semantic search across both
suspicious = analytical_txns.search("unusual transaction patterns")

# Complex aggregation
summary = analytical_txns.facet_query(
    facet_on=["merchant", "category"],
    where={"flagged": True}
)
```

#### 5. **Enrich Back to OpenMetadata (Optional)**

Feed insights back to the catalog:

```python
# Publish data quality metrics from linkml-store operations
quality_metrics = analytical_txns.validate()
publish_to_openmetadata(quality_metrics)

# Track query lineage
track_lineage("transactions", query, results)
```

## Key Integration Points

### 1. OpenMetadata Connector for LinkML-Store

**Purpose**: Catalog linkml-store databases in OpenMetadata

```
OpenMetadata Ingestion Framework
├── LinkML-Store Connector (new)
│   ├── Discover databases and collections
│   ├── Extract LinkML schemas → OpenMetadata entities
│   ├── Harvest metadata: row counts, data types, samples
│   └── Track linkml-store endpoints and configurations
```

**Implementation**:
- Use linkml-store REST API for metadata harvesting
- Convert LinkML schemas to OpenMetadata entity definitions
- Schedule periodic ingestion for updates

### 2. LinkML-Store Configuration from OpenMetadata

**Purpose**: Auto-configure linkml-store from OpenMetadata catalog

```python
from linkml_store.integrations.openmetadata import OpenMetadataAdapter

# Discover and configure from OpenMetadata
adapter = OpenMetadataAdapter(openmetadata_url="http://metadata.org")

# Get configuration for specific data assets
config = adapter.generate_linkml_config(
    database_fqn="postgres.analytics_db.public",
    collections=["transactions", "customers"]
)

# Create linkml-store client
client = Client.from_config(config)
```

### 3. Unified Schema Management

**Purpose**: Keep LinkML schemas in sync with OpenMetadata

```
LinkML Schema (YAML)
        ↕
    Converter
        ↕
OpenMetadata Entity (JSON Schema)
```

**Benefits**:
- Single source of truth for data structure
- Validation rules shared between systems
- Schema evolution tracked in both systems

### 4. Query Lineage Tracking

**Purpose**: Track linkml-store operations in OpenMetadata lineage

```python
# linkml-store with lineage tracking
client = Client(track_lineage=True)

# Query creates lineage record
results = collection.query(...)

# Lineage published to OpenMetadata:
# Source: transactions collection
# Operation: filter + aggregation
# Output: analysis results
# User: data-scientist@org.com
# Timestamp: 2024-11-24T10:30:00Z
```

## Use Cases

### Use Case 1: Scientific Data Federation

**Scenario**: Research institution with diverse datasets

**OpenMetadata Setup**:
- Catalog genomics data (Parquet in S3)
- Catalog experimental results (MongoDB)
- Catalog analysis results (PostgreSQL)
- Catalog instrument data (HDF5 files)
- Document protocols and methodologies
- Track data provenance and publication lineage

**LinkML-Store Usage**:
- Define LinkML schemas for scientific entities
- Query across all data sources with unified API
- Semantic search: "experiments with CRISPR on mice"
- Validate data against community standards
- Run inference to complete partial records
- Export for Jupyter notebook analysis

**Value**: Scientists discover data through OpenMetadata, analyze it through linkml-store

### Use Case 2: Enterprise Data Platform

**Scenario**: Large company with many data sources

**OpenMetadata Setup**:
- Catalog all databases (50+ systems)
- Catalog data warehouses and lakes
- Catalog BI dashboards and reports
- Establish data governance policies
- Track sensitive data for compliance
- Enable cross-team data discovery

**LinkML-Store Usage**:
- Provide unified query layer for developers
- Enable semantic search across enterprise data
- Support multi-backend analytics
- Validate against company data standards
- Support feature engineering for ML
- Enable data API services

**Value**: Business users find data via OpenMetadata; developers and data scientists operate on it via linkml-store

### Use Case 3: Multi-Tenant SaaS Platform

**Scenario**: SaaS company with per-tenant data isolation

**OpenMetadata Setup**:
- Catalog tenant databases and schemas
- Document multi-tenant architecture
- Track data ownership per tenant
- Manage access policies
- Monitor data quality per tenant

**LinkML-Store Usage**:
- Abstract tenant-specific backends
- Provide tenant-isolated queries
- Enable cross-tenant analytics (where permitted)
- Support per-tenant customization
- Unified API regardless of tenant storage

**Value**: Platform operators manage metadata; services access data uniformly

## Implementation Roadmap

### Phase 1: Basic Integration (Weeks 1-4)

**Goal**: Catalog linkml-store databases in OpenMetadata

- [ ] Build OpenMetadata ingestion connector for linkml-store
- [ ] Extract metadata from linkml-store REST API
- [ ] Convert LinkML schemas to OpenMetadata entities
- [ ] Test with sample linkml-store databases
- [ ] Document connector usage

**Deliverable**: OpenMetadata can discover and catalog linkml-store databases

### Phase 2: Configuration Bridge (Weeks 5-8)

**Goal**: Auto-configure linkml-store from OpenMetadata

- [ ] Implement OpenMetadata adapter for linkml-store
- [ ] Generate linkml-store config from OpenMetadata catalog
- [ ] Support automatic connection string resolution
- [ ] Handle authentication and credentials
- [ ] Create CLI tools for configuration generation

**Deliverable**: Users can discover data in OpenMetadata and instantly query it via linkml-store

### Phase 3: Schema Synchronization (Weeks 9-12)

**Goal**: Unified schema management

- [ ] Build LinkML ↔ OpenMetadata schema converters
- [ ] Implement bidirectional sync
- [ ] Handle schema evolution
- [ ] Validate consistency
- [ ] Document schema management workflow

**Deliverable**: Schemas maintained consistently across both systems

### Phase 4: Advanced Features (Weeks 13+)

**Goal**: Deep integration features

- [ ] Query lineage tracking
- [ ] Data quality metrics integration
- [ ] Unified access control
- [ ] Real-time metadata updates
- [ ] Usage analytics
- [ ] Collaborative features

**Deliverable**: Fully integrated data platform

## Technical Architecture

### Recommended Architecture Pattern

```
┌─────────────────────────────────────────────────────────┐
│                  Applications & Users                    │
│   (Data Scientists, Engineers, Analysts, Services)      │
└───────┬────────────────────────────┬────────────────────┘
        │                            │
        │ Discovery                  │ Operations
        │                            │
        v                            v
┌──────────────────┐        ┌──────────────────┐
│   OpenMetadata   │◄──────►│  LinkML-Store    │
│   Web UI / API   │  sync  │   Client API     │
└────────┬─────────┘        └────────┬─────────┘
         │                           │
         │ catalogs                  │ queries
         │                           │
         v                           v
┌─────────────────────────────────────────────────────────┐
│              Heterogeneous Data Landscape                │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│PostgreSQL│ MongoDB  │  Neo4j   │  Solr    │ Data Lake   │
│  Tables  │Documents │  Graphs  │  Indices │ Parquet     │
└──────────┴──────────┴──────────┴──────────┴─────────────┘
```

### Component Responsibilities

**OpenMetadata Components**:
- **Catalog UI**: User-facing discovery and documentation
- **Ingestion Framework**: Harvest metadata from all sources
- **Metadata Store**: Central repository for metadata
- **API Server**: REST API for metadata operations
- **Quality Framework**: Data quality rules and monitoring

**LinkML-Store Components**:
- **Client API**: Python/CLI/REST interface for operations
- **Database Abstraction**: Unified interface across backends
- **Index Layer**: LLM-powered semantic search
- **Inference Layer**: ML-based data enrichment
- **Schema Validator**: LinkML-based validation

**Integration Components** (to be built):
- **OpenMetadata → LinkML-Store Connector**: Metadata ingestion
- **LinkML-Store → OpenMetadata Publisher**: Lineage and quality metrics
- **Schema Bridge**: Format conversion and synchronization
- **Configuration Generator**: Auto-config from catalog

## Getting Started

### Prerequisites

1. **OpenMetadata** installed and running (see [docs.open-metadata.org](https://docs.open-metadata.org))
2. **LinkML-Store** installed: `pip install linkml-store`
3. Access to data sources you want to catalog and query

### Quick Start: Manual Integration

#### Step 1: Catalog Data in OpenMetadata

```bash
# Install OpenMetadata CLI
pip install openmetadata-ingestion

# Run ingestion for your databases
metadata ingest -c postgres_config.yaml
metadata ingest -c mongodb_config.yaml
```

#### Step 2: Document in OpenMetadata UI

- Add descriptions to tables and columns
- Set data ownership
- Add tags and glossary terms
- Define quality rules

#### Step 3: Configure LinkML-Store

Create `linkml-store-config.yaml`:

```yaml
databases:
  postgres_analytics:
    handle: postgresql://user:pass@host/analytics_db
    schema_location: schemas/analytics.yaml
    collections:
      transactions:
        alias: transactions
        type: Transaction
      customers:
        alias: customers
        type: Customer
```

#### Step 4: Query with LinkML-Store

```python
from linkml_store import Client

# Load configuration
client = Client.from_config("linkml-store-config.yaml")

# Access collections
transactions = client.databases["postgres_analytics"].collections["transactions"]

# Perform operations
results = transactions.query(where={"amount": {"$gt": 1000}})
summary = transactions.facet_query(facet_on=["category"])
similar = transactions.search("large international wire transfers")
```

### Next Steps

1. **Explore OpenMetadata**: Catalog your data landscape
2. **Explore LinkML-Store**: Define schemas and test queries
3. **Monitor Integration**: Track usage and identify patterns
4. **Provide Feedback**: Help shape the integration roadmap

## Resources

### OpenMetadata

- **Website**: [https://open-metadata.org/](https://open-metadata.org/)
- **Documentation**: [https://docs.open-metadata.org/](https://docs.open-metadata.org/)
- **GitHub**: [https://github.com/open-metadata/OpenMetadata](https://github.com/open-metadata/OpenMetadata)
- **Community**: Slack workspace at [https://slack.open-metadata.org/](https://slack.open-metadata.org/)

### LinkML-Store

- **Documentation**: [https://linkml.io/linkml-store/](https://linkml.io/linkml-store/)
- **GitHub**: [https://github.com/linkml/linkml-store](https://github.com/linkml/linkml-store)
- **LinkML**: [https://linkml.io/](https://linkml.io/)

### LinkML

- **Website**: [https://linkml.io/](https://linkml.io/)
- **Documentation**: [https://linkml.io/linkml/](https://linkml.io/linkml/)
- **Schema Guide**: [https://linkml.io/linkml/schemas/](https://linkml.io/linkml/schemas/)

## Contributing

This integration is in early conceptual stages. Contributions are welcome:

1. **Feedback**: Share your use cases and requirements
2. **Design**: Help refine the integration architecture
3. **Implementation**: Build connector components
4. **Documentation**: Improve and expand this guide
5. **Testing**: Try the integration and report issues

## Conclusion

**OpenMetadata** and **LinkML-Store** together provide a powerful combination:

- **OpenMetadata organizes** your data landscape, making it discoverable and governable
- **LinkML-Store operates** on that data, providing flexible querying and AI-ready capabilities

This two-layer architecture enables organizations to:
- **Find** data assets across heterogeneous systems
- **Understand** data through comprehensive metadata
- **Access** data through unified interfaces
- **Query** across different backends with one API
- **Analyze** with AI-powered semantic search
- **Govern** with centralized policies and quality rules

The future of data platforms is not about consolidating everything into one system, but about organizing diverse systems and providing unified ways to work with them. OpenMetadata + LinkML-Store delivers exactly that.
