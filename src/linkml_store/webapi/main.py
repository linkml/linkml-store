import os
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import httpx
import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from linkml_renderer.renderers.html_renderer import HTMLRenderer
from linkml_renderer.style.style_engine import StyleEngine
from linkml_runtime.dumpers import json_dumper
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from linkml_store import Client
from linkml_store.api import Collection, Database
from linkml_store.api.queries import Query as StoreQuery
from linkml_store.utils.format_utils import load_objects
from linkml_store.webapi.html import HTML_TEMPLATES_DIR

html_renderer = HTMLRenderer()

# Load environment variables from .env file
load_dotenv()

# Parse command-line arguments
# parser = argparse.ArgumentParser(description="LinkML Store FastAPI server")
# parser.add_argument("--config", type=str, help="Path to the configuration file")
# args = parser.parse_args()


# Load configuration
config = None
if os.environ.get("LINKML_STORE_CONFIG"):
    with open(os.environ["LINKML_STORE_CONFIG"], "r") as f:
        config = yaml.safe_load(f)

# Initialize client
client = Client().from_config(config) if config else Client()

app = FastAPI(title="LinkML Store API")

templates = Jinja2Templates(directory=HTML_TEMPLATES_DIR)


# Pydantic models for requests and responses


class Link(BaseModel):
    rel: str
    href: str


class Meta(BaseModel):
    path: Optional[str] = None
    path_template: Optional[str] = None
    params: Dict[str, Any] = {}
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "1.0"
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class Error(BaseModel):
    code: str
    message: str
    details: Optional[str] = None


class APIResponse(BaseModel):
    data: Optional[Any] = None
    meta: Meta = Field(default_factory=Meta)
    links: Optional[List[Link]] = None
    errors: Optional[List[Error]] = None


class DatabaseCreate(BaseModel):
    name: str
    handle: str


class CollectionCreate(BaseModel):
    name: str
    alias: Optional[str] = None


class ObjectInsert(BaseModel):
    objects: List[Dict[str, Any]]


# Helper functions


def get_client():
    return client


def get_database(client: Client = Depends(get_client)):
    def _get_database(database_name: str) -> Database:
        try:
            return client.get_database(database_name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Database '{database_name}' not found")

    return _get_database


def get_collection(database: Database = Depends(get_database())):
    def _get_collection(collection_name: str) -> Collection:
        try:
            return database.get_collection(collection_name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

    return _get_collection


# API routes


@app.get("/", response_model=APIResponse, description="Top level.")
async def top(request: Request, client: Client = Depends(get_client)):
    links = [
        Link(rel="self", href="/"),
        Link(rel="docs", href="/docs"),
        Link(rel="databases", href="/databases"),
        Link(rel="config", href="/config"),
    ]
    return APIResponse(data={}, links=links)


@app.get("/config", response_model=APIResponse, description="Configuration metadata")
async def config(request: Request, client: Client = Depends(get_client)):
    client = get_client()
    data = client.metadata
    links = [
        Link(rel="self", href="/"),
        Link(rel="docs", href="/docs"),
        Link(rel="databases", href="/databases"),
        Link(rel="config", href="/config"),
    ]
    return APIResponse(data=data, links=links)


@app.get(
    "/databases", response_model=APIResponse, description="List all databases with clickable links to their details."
)
async def list_databases(request: Request, client: Client = Depends(get_client)):
    databases = list(client.databases.keys())

    database_links = [Link(rel="database", href=f"/databases/{db_name}") for db_name in databases]

    additional_links = [
        Link(rel="self", href="/databases"),
        Link(rel="create_database", href="/database/create"),
    ]

    response_data = {
        "objects": [
            {
                "name": db_name,
                "type": "Database",
                "links": [
                    {"rel": "self", "href": f"/databases/{db_name}"},
                    {"rel": "collections", "href": f"/databases/{db_name}/collections"},
                    {"rel": "schema", "href": f"/databases/{db_name}/schema"},
                ],
            }
            for db_name in databases
        ]
    }

    api_response = APIResponse(
        meta=Meta(path=request.url.path, path_template="databases", params={}),
        data=response_data,
        links=additional_links + database_links,
    )
    if request.headers.get("Accept") == "text/html":
        return templates.TemplateResponse("databases.html", {"request": request, "response": api_response})
    else:
        return JSONResponse(content=api_response.dict())


@app.post("/database/create", response_model=APIResponse)
async def create_database(database: DatabaseCreate, client: Client = Depends(get_client)):
    db = client.attach_database(database.handle, alias=database.name)
    return APIResponse(
        data={"name": db.metadata.alias, "handle": db.metadata.handle},
        links=[
            Link(rel="self", href=f"/databases/{db.metadata.alias}"),
            Link(rel="collections", href=f"/databases/{db.metadata.alias}/collections"),
        ],
    )


@app.get("/databases/{database_name}", response_model=APIResponse)
async def get_database_details(
    request: Request, database_name: str, get_db: Callable[[str], Database] = Depends(get_database)
):
    database = get_db(database_name)
    collections = database.list_collections()
    return APIResponse(
        meta=Meta(path=request.url.path, path_template="database_details", params={"database_name": database_name}),
        data={
            "name": database.metadata.alias,
            "handle": database.metadata.handle,
            "num_collections": len(collections),
            "collections": [
                {
                    "name": c.name,
                    "links": [
                        {"rel": "self", "href": f"/databases/{database_name}/collections/{c.name}"},
                        {"rel": "objects", "href": f"/databases/{database_name}/collections/{c.name}/objects"},
                    ],
                }
                for c in collections
            ],
        },
        links=[
            Link(rel="self", href=f"/databases/{database_name}"),
            Link(rel="collections", href=f"/databases/{database_name}/collections"),
            Link(rel="schema", href=f"/databases/{database_name}/schema"),
            Link(rel="all_databases", href="/databases"),
        ],
    )


@app.get("/databases/{database_name}/collections", response_model=APIResponse)
async def list_collections(
    request: Request, database_name: str, get_db: Callable[[str], Database] = Depends(get_database)
):
    database = get_db(database_name)
    collections = database.list_collections()
    return APIResponse(
        meta=Meta(path=request.url.path, path_template="collections", params={}),
        data={
            "items": [
                {
                    "name": c.name,
                    "type": "Collection",
                    "links": [
                        {"rel": "self", "href": f"/databases/{database_name}/collections/{c.name}"},
                        {"rel": "objects", "href": f"/databases/{database_name}/collections/{c.name}/objects"},
                    ],
                }
                for c in collections
            ]
        },
        links=[
            Link(rel="self", href=f"/databases/{database_name}/collections"),
            Link(rel="database", href=f"/databases/{database_name}"),
            Link(rel="create_collection", href=f"/databases/{database_name}/collections"),
        ],
    )


@app.get("/databases/{database_name}/collections/{collection_name}", response_model=APIResponse)
async def get_collection_details(
    request: Request,
    database_name: str,
    collection_name: str,
    get_db: Callable[[str], Database] = Depends(get_database),
):
    database = get_db(database_name)
    collection = database.get_collection(collection_name)
    return APIResponse(
        meta=Meta(
            path=request.url.path,
            path_template="collection_details",
            params={
                "database_name": database_name,
                "collection_name": collection_name,
            },
        ),
        data={"name": collection.name, "alias": collection.alias, "num_objects": collection.find({}).num_rows},
        links=[
            Link(rel="self", href=f"/databases/{database_name}/collections/{collection_name}"),
            Link(rel="objects", href=f"/databases/{database_name}/collections/{collection_name}/objects"),
            Link(rel="facets", href=f"/databases/{database_name}/collections/{collection_name}/facets"),
            Link(rel="search", href=f"/databases/{database_name}/collections/{collection_name}/search/{{term}}"),
            Link(rel="database", href=f"/databases/{database_name}"),
        ],
    )


@app.post("/databases/{database_name}/collections/{collection_name}/create", response_model=APIResponse)
async def create_collection(
    request: Request,
    database_name: str,
    collection: CollectionCreate,
    get_db: Callable[[str], Database] = Depends(get_database),
):
    database = get_db(database_name)
    new_collection = database.create_collection(collection.name, alias=collection.alias)
    return APIResponse(
        meta=Meta(path=request.url.path, path_template="create_collection", params={"database_name": database_name}),
        data={"name": new_collection.name, "alias": new_collection.alias},
        links=[
            Link(rel="self", href=f"/databases/{database_name}/collections/{new_collection.name}"),
            Link(rel="database", href=f"/databases/{database_name}"),
            Link(rel="objects", href=f"/databases/{database_name}/collections/{new_collection.name}/objects"),
            Link(rel="objects", href=f"/databases/{database_name}/collections/{new_collection.name}/facets"),
        ],
    )


@app.get("/databases/{database_name}/collections/{collection_name}/objects", response_model=APIResponse)
async def collection_list_objects(
    request: Request,
    database_name: str,
    collection_name: str,
    where: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    get_db: Callable[[str], Database] = Depends(get_database),
):
    database = get_db(database_name)
    collection = database.get_collection(collection_name)
    where_clause = load_objects(where) if where else None
    query = StoreQuery(from_table=collection.name, where_clause=where_clause, limit=limit, offset=offset)
    result = collection.query(query)

    total_count = collection.find({}).num_rows
    total_pages = (total_count + limit - 1) // limit
    current_page = offset // limit + 1

    base_url = f"/databases/{database_name}/collections/{collection_name}/objects"
    links = [Link(rel="self", href=f"{base_url}?limit={limit}&offset={offset}")]
    if current_page > 1:
        links.append(Link(rel="prev", href=f"{base_url}?limit={limit}&offset={offset - limit}"))
    if current_page < total_pages:
        links.append(Link(rel="next", href=f"{base_url}?limit={limit}&offset={offset + limit}"))

    links.extend(
        [
            Link(rel="first", href=f"{base_url}?limit={limit}&offset=0"),
            Link(rel="last", href=f"{base_url}?limit={limit}&offset={(total_pages - 1) * limit}"),
            Link(rel="collection", href=f"/databases/{database_name}/collections/{collection_name}"),
            Link(rel="database", href=f"/databases/{database_name}"),
        ]
    )

    return APIResponse(
        meta=Meta(
            path=request.url.path,
            path_template="objects",
            params={
                "database_name": database_name,
                "collection_name": collection_name,
            },
        ),
        data={
            "domain_objects": result.rows,
            "total_count": total_count,
            "page": current_page,
            "total_pages": total_pages,
            "page_size": limit,
        },
        links=links,
    )


@app.get("/databases/{database_name}/collections/{collection_name}/search/{term}", response_model=APIResponse)
async def search_objects(
    request: Request,
    database_name: str,
    collection_name: str,
    term: str,
    limit: int = Query(5, ge=1, le=100),
    offset: int = Query(0, ge=0),
    get_db: Callable[[str], Database] = Depends(get_database),
):
    database = get_db(database_name)
    collection = database.get_collection(collection_name)
    result = collection.search(term, limit=limit)

    total_count = result.num_rows
    total_pages = (total_count + limit - 1) // limit
    current_page = offset // limit + 1

    base_url = f"/databases/{database_name}/collections/{collection_name}/search/{term}"
    links = [Link(rel="self", href=f"{base_url}?limit={limit}&offset={offset}")]
    if current_page > 1:
        links.append(Link(rel="prev", href=f"{base_url}?limit={limit}&offset={offset - limit}"))
    if current_page < total_pages:
        links.append(Link(rel="next", href=f"{base_url}?limit={limit}&offset={offset + limit}"))

    links.extend(
        [
            Link(rel="first", href=f"{base_url}?limit={limit}&offset=0"),
            Link(rel="last", href=f"{base_url}?limit={limit}&offset={(total_pages - 1) * limit}"),
            Link(rel="collection", href=f"/databases/{database_name}/collections/{collection_name}"),
            Link(rel="database", href=f"/databases/{database_name}"),
        ]
    )

    return APIResponse(
        meta=Meta(
            path=request.url.path,
            path_template="collection_search",
            params={
                "database_name": database_name,
                "collection_name": collection_name,
                "term": term,
            },
        ),
        data={
            "objects": result.ranked_rows,
            "total_count": total_count,
            "page": current_page,
            "total_pages": total_pages,
            "page_size": limit,
        },
        links=links,
    )


@app.get("/databases/{database_name}/collections/{collection_name}/facets", response_model=APIResponse)
async def objects_facets(
    request: Request,
    database_name: str,
    collection_name: str,
    where: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    get_db: Callable[[str], Database] = Depends(get_database),
):
    database = get_db(database_name)
    collection = database.get_collection(collection_name)
    where_clause = load_objects(where) if where else None
    results = collection.query_facets(where_clause)

    total_count = collection.find({}).num_rows
    total_pages = (total_count + limit - 1) // limit
    current_page = offset // limit + 1

    base_url = f"/databases/{database_name}/collections/{collection_name}/facets"
    links = [Link(rel="self", href=f"{base_url}?limit={limit}&offset={offset}")]
    if current_page > 1:
        links.append(Link(rel="prev", href=f"{base_url}?limit={limit}&offset={offset - limit}"))
    if current_page < total_pages:
        links.append(Link(rel="next", href=f"{base_url}?limit={limit}&offset={offset + limit}"))

    links.extend(
        [
            Link(rel="first", href=f"{base_url}?limit={limit}&offset=0"),
            Link(rel="last", href=f"{base_url}?limit={limit}&offset={(total_pages - 1) * limit}"),
            Link(rel="collection", href=f"/databases/{database_name}/collections/{collection_name}"),
            Link(rel="database", href=f"/databases/{database_name}"),
        ]
    )

    return APIResponse(
        meta=Meta(
            path=request.url.path,
            path_template="facets",
            params={
                "database_name": database_name,
                "collection_name": collection_name,
            },
        ),
        data={
            "items": results,
            "total_count": total_count,
            "page": current_page,
            "total_pages": total_pages,
            "page_size": limit,
        },
        links=links,
    )


@app.post("/databases/{database_name}/collections/{collection_name}/objects", response_model=APIResponse)
async def insert_objects(
    database_name: str,
    collection_name: str,
    insert: ObjectInsert,
    get_db: Callable[[str], Database] = Depends(get_database),
):
    database = get_db(database_name)
    collection = database.get_collection(collection_name)
    collection.insert(insert.objects)
    return APIResponse(
        data={"inserted_count": len(insert.objects)},
        links=[
            Link(rel="self", href=f"/databases/{database_name}/collections/{collection_name}/objects"),
            Link(rel="collection", href=f"/databases/{database_name}/collections/{collection_name}"),
        ],
    )


@app.get("/databases/{database_name}/schema", response_model=APIResponse)
async def get_database_schema(
    request: Request, database_name: str, get_db: Callable[[str], Database] = Depends(get_database)
):
    database = get_db(database_name)
    schema = database.schema_view.schema
    schema_dict = json_dumper.to_dict(schema)
    return APIResponse(
        meta=Meta(path=request.url.path, path_template="schema", params={"database_name": database_name}),
        data={"schema": schema_dict},
        links=[
            Link(rel="self", href=f"/databases/{database_name}/schema"),
            Link(rel="database", href=f"/databases/{database_name}"),
        ],
    )


@app.get("/pages/{path:path}", response_class=HTMLResponse)
async def generic_page(request: Request, path: str, get_db: Callable[[str], Database] = Depends(get_database)):
    # Construct the API URL
    api_url = f"{request.base_url}{path}"
    query_params = dict(request.query_params)

    # Make a request to the API
    async with httpx.AsyncClient() as client:
        response = await client.get(api_url, params=query_params)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    # Parse the JSON response
    api_data = response.json()

    # Use the template specified in the API response
    meta = api_data["meta"]
    # template_name = meta['path_template'] + ".html.j2"
    template_name = "generic.html.j2"
    params = meta["params"]

    data = api_data["data"]
    data_html = None
    if "domain_objects" in data:
        objs = data["domain_objects"]
        db = get_db(params["database_name"])
        sv = db.schema_view
        collection = db.get_collection(params["collection_name"])
        if collection.class_definition():
            cn = collection.class_definition().name
            style_engine = StyleEngine(schemaview=sv)
            html_renderer.style_engine = style_engine
            data_html = [html_renderer.render(obj, schemaview=sv, source_element_name=cn) for obj in objs]

    # Render the appropriate template
    return templates.TemplateResponse(
        template_name,
        {
            "request": request,
            "response": api_data,
            "current_path": f"/pages/{path}",
            "data_html": data_html,
            "params": params,
        },
    )


def run_server():
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


# if __name__ == "__main__":
#    run_server()


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("linkml_store.webapi.main:app", host="127.0.0.1", port=8000, reload=True)
