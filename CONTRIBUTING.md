# Contribution Guidelines

When contributing to this repository, please first discuss the changes you wish to make via an issue, email, or any other method, with the owners of this repository before issuing a pull request.

## How to contribute

### Reporting bugs or making feature requests

To report a bug or suggest a new feature, please go to the [linkml/linkml-store issue tracker](https://github.com/linkml/linkml-store/issues), as we are
consolidating issues there.

Please supply enough details to the developers to enable them to verify and troubleshoot your issue:

* Provide a clear and descriptive title as well as a concise summary of the issue to identify the problem.
* Describe the exact steps which reproduce the problem in as many details as possible.
* Describe the behavior you observed after following the steps and point out what exactly is the problem with that behavior.
* Explain which behavior you expected to see instead and why.
* Provide screenshots of the expected or actual behaviour where applicable.


### The development lifecycle

1. Create a bug fix or feature development branch, based off the `main` branch of the upstream repo, and not your fork. Name the branch appropriately, briefly summarizing the bug fix or feature request. If none come to mind, you can include the issue number in the branch name. Some examples of branch names are, `bugfix/breaking-pipfile-error` or `feature/add-click-cli-layer`, or `bugfix/issue-414`
2. Make sure your development branch has all the latest commits from the `main` branch.
3. After completing work and testing locally, push the code to the appropriate branch on your fork.
4. Create a pull request from the bug/feature branch of your fork to the `main` branch of the upstream repository.

Note: All the development must be done on a branch on your fork.

ALSO NOTE: github.com lets you create a pull request from the main branch, automating the steps above.

> A code review (which happens with both the contributor and the reviewer present) is required for contributing.

### How to write a great issue

Please review GitHub's overview article,
["Tracking Your Work with Issues"][about-issues].

<a id="great-pulls"></a>

### How to create a great pull/merge request

Please review GitHub's article, ["About Pull Requests"][about-pulls],
and make your changes on a [new branch][about-branches].

[about-branches]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches
[about-issues]: https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues
[about-pulls]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests
[issues]: https://github.com/linkml/linkml-store/issues/
[pulls]: https://github.com/linkml/linkml-store/pulls/

We recommend also reading [GitHub Pull Requests: 10 Tips to Know](https://blog.mergify.com/github-pull-requests-10-tips-to-know/)

## Adding a New Backend Store

LinkML-Store supports multiple backend stores (databases) through a common abstraction layer. To add support for a new backend, follow these steps:

### 1. Create the Backend Module Structure

Create a new directory under `src/linkml_store/api/stores/` for your backend:

```
src/linkml_store/api/stores/mybackend/
    __init__.py
    mybackend_database.py
    mybackend_collection.py
```

### 2. Implement the Database Class

Create a database class that inherits from `linkml_store.api.Database`:

```python
from linkml_store.api import Database
from .mybackend_collection import MyBackendCollection

class MyBackendDatabase(Database):
    collection_class = MyBackendCollection
    
    def __init__(self, handle: Optional[str] = None, **kwargs):
        super().__init__(handle=handle, **kwargs)
        # Initialize your backend connection here
    
    def commit(self, **kwargs):
        # Implement transaction commit
        pass
    
    def close(self, **kwargs):
        # Close backend connection
        pass
    
    def drop(self, missing_ok=True, **kwargs):
        # Drop the database
        pass
    
    def list_collection_names(self) -> List[str]:
        # Return list of collection names
        pass
```

### 3. Implement the Collection Class

Create a collection class that inherits from `linkml_store.api.Collection`:

```python
from linkml_store.api import Collection
from linkml_store.api.queries import Query, QueryResult

class MyBackendCollection(Collection):
    def insert(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        # Insert objects into the collection
        pass
    
    def delete(self, objs: Union[OBJECT, List[OBJECT]], **kwargs) -> Optional[int]:
        # Delete objects from the collection
        pass
    
    def query(self, query: Query, **kwargs) -> QueryResult:
        # Execute a query and return results
        pass
    
    def find(self, where: Optional[Dict[str, Any]] = None, **kwargs) -> QueryResult:
        # Find objects matching criteria
        pass
    
    def update(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        # Update existing objects
        pass
```

### 4. Register the Backend

Add your backend to the `HANDLE_MAP` in `src/linkml_store/api/client.py`:

```python
HANDLE_MAP = {
    # ... existing backends ...
    "mybackend": "linkml_store.api.stores.mybackend.mybackend_database.MyBackendDatabase",
}
```

If your backend uses file extensions, add them to `SUFFIX_MAP`:

```python
SUFFIX_MAP = {
    # ... existing suffixes ...
    "mdb": "mybackend:///{path}",
}
```

### 5. Add Dependencies

If your backend requires additional dependencies, add them to `pyproject.toml`:

```toml
[tool.poetry.extras]
mybackend = ["mybackend-python-client>=1.0.0"]
all = [
    # ... existing dependencies ...
    "mybackend-python-client>=1.0.0",
]
```

### 6. Write Tests

Create test files in the `tests/` directory:

```python
# tests/test_stores/test_mybackend.py
import pytest
from linkml_store import Client

def test_mybackend_basic_operations():
    client = Client()
    db = client.attach_database("mybackend:///:memory:", alias="test")
    collection = db.create_collection("TestCollection")
    
    # Test insert
    collection.insert([{"id": "1", "name": "Test"}])
    
    # Test query
    result = collection.find()
    assert len(result.rows) == 1
    assert result.rows[0]["name"] == "Test"
```

### 7. Document Your Backend

Add documentation explaining:
- Connection string format (e.g., `mybackend://host:port/database`)
- Any special configuration options
- Limitations or backend-specific features
- Example usage in notebooks

### Best Practices

1. **Handle Errors Gracefully**: Provide clear error messages for connection failures and unsupported operations
2. **Type Mapping**: Implement proper type mapping between LinkML types and your backend's native types
3. **Performance**: Consider implementing batch operations for better performance
4. **Schema Support**: If your backend supports schemas, integrate with LinkML's schema validation
5. **Indexing**: Implement index creation/management if supported by your backend
6. **Query Translation**: Convert LinkML queries to your backend's query language

### Example Backends to Study

Look at these existing implementations for reference:
- `duckdb`: SQL-based backend with array support
- `mongodb`: Document-based NoSQL backend
- `filesystem`: File-based storage using JSON/YAML/CSV files
- `neo4j`: Graph database backend
