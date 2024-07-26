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


class Item(BaseModel):
    name: str
    type: Optional[str] = None
    links: List[Link]
    data: Optional[Any] = None
    html: Optional[str] = None


class ItemType(BaseModel):
    name: str
    description: Optional[str] = None


class Meta(BaseModel):
    path: Optional[str] = None
    path_template: Optional[str] = None
    params: Dict[str, Any] = {}
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "1.0"
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    item_count: Optional[int] = None
    paged: bool = False
    page: int = 1
    page_size: Optional[int] = None


class Error(BaseModel):
    code: str
    message: str
    details: Optional[str] = None


class APIResponse(BaseModel):
    meta: Meta = Field(default_factory=Meta)
    items: Optional[List[Item]] = None
    item_type: Optional[ItemType] = None
    data: Optional[Any] = None
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
        Link(rel="pages", href="/pages"),
        Link(rel="databases", href="/databases"),
        Link(rel="config", href="/config"),
    ]
    return APIResponse(data={}, links=links)


@app.get("/config", response_model=APIResponse, description="Configuration metadata")
async def config(request: Request, client: Client = Depends(get_client)):
    client = get_client()
    data = client.metadata
    links = [
        Link(rel="self", href="/config"),
        Link(rel="parent", href="/"),
    ]
    return APIResponse(data=data, links=links)


@app.get(
    "/databases", response_model=APIResponse, description="List all databases with clickable links to their details."
)
async def list_databases(request: Request, client: Client = Depends(get_client)):
    databases = list(client.databases.keys())

    # database_links = [Link(rel="database", href=f"/databases/{db_name}") for db_name in databases]
    database_links = []

    additional_links = [
        Link(rel="self", href="/databases"),
        Link(rel="parent", href="/"),
        Link(rel="create_database", href="/database/create"),
    ]

    items = [
        Item(name=db_name, type="Database", links=[Link(rel="self", href=f"/databases/{db_name}")], data={})
        for db_name in databases
    ]

    api_response = APIResponse(
        meta=Meta(path=request.url.path, path_template="databases", params={}),
        data={},
        items=items,
        links=additional_links + database_links,
    )
    if request.headers.get("Accept") == "text/html":
        return templates.TemplateResponse("databases.html", {"request": request, "response": api_response})
    else:
        return JSONResponse(content=api_response.dict())


@app.post("/database/create", response_model=APIResponse)
async def create_database(database: DatabaseCreate, client: Client = Depends(get_client)):
    # TODO
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
    db_metadata = database.metadata.model_dump(exclude_none=True, exclude_defaults=True)
    if "collections" in db_metadata:
        # do not replicate information
        del db_metadata["collections"]
    return APIResponse(
        meta=Meta(path=request.url.path, path_template="database_details", params={"database_name": database_name}),
        data={
            "name": database.metadata.alias,
            "handle": database.metadata.handle,
            "config": db_metadata,
            "num_collections": len(collections),
        },
        links=[
            Link(rel="self", href=f"/databases/{database_name}"),
            Link(rel="collections", href=f"/databases/{database_name}/collections"),
            Link(rel="schema", href=f"/databases/{database_name}/schema"),
            Link(rel="parent", href="/databases"),
        ],
    )


@app.get("/databases/{database_name}/collections", response_model=APIResponse)
async def list_database_collections(
    request: Request, database_name: str, get_db: Callable[[str], Database] = Depends(get_database)
):
    database = get_db(database_name)
    collections = database.list_collections()
    items = [
        Item(
            name=c.alias,
            type="Collection",
            links=[
                Link(rel="self", href=f"/databases/{database_name}/collections/{c.alias}"),
                Link(rel="objects", href=f"/databases/{database_name}/collections/{c.alias}/objects"),
            ],
        )
        for c in collections
    ]
    return APIResponse(
        meta=Meta(path=request.url.path, path_template="collections", params={}),
        items=items,
        data={},
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
        data={
            "target_class_name": collection.target_class_name,
            "alias": collection.alias,
            "num_objects": collection.find({}).num_rows,
        },
        links=[
            Link(rel="self", href=f"/databases/{database_name}/collections/{collection_name}"),
            Link(rel="objects", href=f"/databases/{database_name}/collections/{collection_name}/objects"),
            Link(rel="attributes", href=f"/databases/{database_name}/collections/{collection_name}/attributes"),
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
    new_collection = database.create_collection(collection.alias, alias=collection.alias)
    return APIResponse(
        meta=Meta(path=request.url.path, path_template="create_collection", params={"database_name": database_name}),
        data={"name": new_collection.alias, "alias": new_collection.alias},
        links=[
            Link(rel="self", href=f"/databases/{database_name}/collections/{new_collection.alias}"),
            Link(rel="database", href=f"/databases/{database_name}"),
            Link(rel="objects", href=f"/databases/{database_name}/collections/{new_collection.alias}/objects"),
            Link(rel="objects", href=f"/databases/{database_name}/collections/{new_collection.alias}/facets"),
        ],
    )


@app.get("/databases/{database_name}/collections/{collection_name}/objects", response_model=APIResponse)
async def list_collection_objects(
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
    query = StoreQuery(from_table=collection.alias, where_clause=where_clause, limit=limit, offset=offset)
    result = collection.query(query)
    base_url = f"/databases/{database_name}/collections/{collection_name}/objects"

    items = []
    cd = collection.class_definition()
    id_att_name = collection.identifier_attribute_name
    for i, row in enumerate(result.rows):
        if id_att_name:
            name = row[id_att_name]
            link = Link(rel="self", href=f"{base_url}/{name}")
        else:
            ix = offset + i
            name = str(ix)
            link = Link(rel="self", href=f"{base_url}_index/{ix}")
        item = Item(name=name, data=row, links=[link])
        items.append(item)

    total_count = collection.find({}).num_rows
    total_pages = (total_count + limit - 1) // limit
    current_page = offset // limit + 1

    links = [Link(rel="self", href=f"{base_url}?limit={limit}&offset={offset}")]
    if current_page > 1:
        links.append(Link(rel="prev", href=f"{base_url}?limit={limit}&offset={offset - limit}"))
    if current_page < total_pages:
        links.append(Link(rel="next", href=f"{base_url}?limit={limit}&offset={offset + limit}"))

    links.extend(
        [
            Link(rel="first", href=f"{base_url}?limit={limit}&offset=0"),
            Link(rel="last", href=f"{base_url}?limit={limit}&offset={(total_pages - 1) * limit}"),
            Link(rel="parent", href=f"/databases/{database_name}/collections/{collection_name}"),
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
            paged=True,
            item_count=total_count,
            page=current_page,
            page_size=limit,
        ),
        item_type=ItemType(
            name=cd.name,
            description=cd.description,
        ),
        items=items,
        data={},
        links=links,
    )


@app.get("/databases/{database_name}/collections/{collection_name}/objects/{id}", response_model=APIResponse)
async def get_object_details(
    request: Request,
    database_name: str,
    collection_name: str,
    id: str,
    get_db: Callable[[str], Database] = Depends(get_database),
):
    database = get_db(database_name)
    collection = database.get_collection(collection_name)
    ids = id.split("+")
    result = collection.get(ids)

    base_url = f"/databases/{database_name}/collections/{collection_name}/objects/{id}"
    links = [Link(rel="self", href=base_url)]
    links.extend(
        [
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
async def list_collection_facets(
    request: Request,
    database_name: str,
    collection_name: str,
    where: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    get_db: Callable[[str], Database] = Depends(get_database),
):
    # DEPRECATED?
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


@app.get("/databases/{database_name}/collections/{collection_name}/attributes", response_model=APIResponse)
async def list_collection_attributes(
    request: Request,
    database_name: str,
    collection_name: str,
    where: Optional[str] = None,
    get_db: Callable[[str], Database] = Depends(get_database),
):
    database = get_db(database_name)
    collection = database.get_collection(collection_name)
    where_clause = load_objects(where) if where else None
    base_url = f"/databases/{database_name}/collections/{collection_name}/attributes"
    results = collection.query_facets(where_clause)
    items = [
        Item(
            name=facet_att,
            type="Attribute",
            links=[
                Link(rel="self", href=f"{base_url}/{facet_att}"),
            ],
            data=[{"value": v, "count": c} for v, c in data],
        )
        for facet_att, data in results.items()
    ]

    links = [
        Link(rel="self", href=base_url),
        Link(rel="parent", href=f"/databases/{database_name}/collections/{collection_name}"),
        Link(rel="grandparent", href=f"/databases/{database_name}"),
    ]

    return APIResponse(
        meta=Meta(
            path=request.url.path,
            path_template="facets",
            params={
                "database_name": database_name,
                "collection_name": collection_name,
            },
        ),
        data={},
        items=items,
        links=links,
    )


@app.get(
    "/databases/{database_name}/collections/{collection_name}/attributes/{attribute_name}", response_model=APIResponse
)
async def get_attribute_details(
    request: Request,
    database_name: str,
    collection_name: str,
    attribute_name: str,
    where: Optional[str] = None,
    get_db: Callable[[str], Database] = Depends(get_database),
):
    database = get_db(database_name)
    collection = database.get_collection(collection_name)
    where_clause = load_objects(where) if where else None
    base_url = f"/databases/{database_name}/collections/{collection_name}/attributes/{attribute_name}"
    count_tuples = collection.query_facets(where_clause, facet_columns=[attribute_name])[attribute_name]
    _count_objs = [{"value": v, "count": c} for v, c in count_tuples]
    cd = collection.class_definition()
    att = cd.attributes[attribute_name]
    att_dict = json_dumper.to_dict(att)
    items = [
        Item(
            name=str(v),
            type="Value",
            links=[
                Link(rel="self", href=f"{base_url}/equals/{v}"),
            ],
            data={"count": c},
        )
        for v, c in count_tuples
    ]

    links = [
        Link(rel="self", href=base_url),
        Link(rel="collection", href=f"/databases/{database_name}/collections/{collection_name}"),
        Link(rel="database", href=f"/databases/{database_name}"),
    ]

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
            "attribute": att_dict,
        },
        items=items,
        links=links,
    )


@app.get(
    "/databases/{database_name}/collections/{collection_name}/attributes/{attribute_name}/equals/{value}",
    response_model=APIResponse,
)
async def query_by_attribute(
    request: Request,
    database_name: str,
    collection_name: str,
    attribute_name: str,
    value: str,
    where: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    get_db: Callable[[str], Database] = Depends(get_database),
):
    database = get_db(database_name)
    collection = database.get_collection(collection_name)
    where_clause = {attribute_name: value}
    query = StoreQuery(from_table=collection.alias, where_clause=where_clause, limit=limit, offset=offset)
    result = collection.query(query)
    items = []
    for i, row in enumerate(result.rows):
        item = Item(name=str(i), type="X", data=row, links=[])
        items.append(item)

    total_count = collection.find({}).num_rows
    total_pages = (total_count + limit - 1) // limit
    current_page = offset // limit + 1

    base_url = f"/databases/{database_name}/collections/{collection_name}/attributes/{attribute_name}/equals/{value}"
    links = [Link(rel="self", href=f"{base_url}?limit={limit}&offset={offset}")]
    if current_page > 1:
        links.append(Link(rel="prev", href=f"{base_url}?limit={limit}&offset={offset - limit}"))
    if current_page < total_pages:
        links.append(Link(rel="next", href=f"{base_url}?limit={limit}&offset={offset + limit}"))

    links.extend(
        [
            Link(rel="first", href=f"{base_url}?limit={limit}&offset=0"),
            Link(rel="last", href=f"{base_url}?limit={limit}&offset={(total_pages - 1) * limit}"),
            Link(
                rel="parent",
                href=f"/databases/{database_name}/collections/{collection_name}/attributes/{attribute_name}",
            ),
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
            paged=True,
            item_count=total_count,
            page=current_page,
            page_size=limit,
        ),
        items=items,
        data={},
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


@app.get("/xxxpages/{path:path}", response_class=HTMLResponse)
async def xxxgeneric_page(request: Request, path: str, get_db: Callable[[str], Database] = Depends(get_database)):
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
    # api_response = APIResponse(**api_data)

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
    payload = APIResponse(**api_data)
    template_name = "generic.html.j2"
    params = payload.meta.params

    # data = payload.data
    data_html = None
    if "database_name" in params:
        db = get_db(params["database_name"])
        sv = db.schema_view
    else:
        sv = None
    if not payload.items:
        payload.items = []
    if payload.item_type and payload.items:
        cn = payload.item_type.name
        style_engine = StyleEngine(schemaview=sv)
        html_renderer.style_engine = style_engine
        for item in payload.items:
            if item.data:
                item.html = html_renderer.render(item.data, schemaview=sv, source_element_name=cn)

    # Render the appropriate template
    return templates.TemplateResponse(
        template_name,
        {
            "request": request,
            "response": payload,
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
