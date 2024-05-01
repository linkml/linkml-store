.. about

About
=========================================================

LinkML-Store is an early effort to provide a unifying storage layer
over multiple different backends, unified via LinkML schemas.

The default backend is DuckDB, but partial implementations are provided for:

- MongoDB
- Solr

This frameworks also allows *composable indexes*. Currently two are supported:

- Simple native trigram method
- LLM text embedding
