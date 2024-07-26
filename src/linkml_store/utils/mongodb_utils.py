import logging
import os
import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from pymongo import MongoClient
from pymongo.database import Database

logger = logging.getLogger(__name__)


def extract_connection_info(db: Database):
    client = db.client

    # Get the host and port
    host_info = client.address
    if host_info:
        host, port = host_info
    else:
        # For replica sets or sharded clusters, we might need to get this differently
        host = client.HOST
        port = client.PORT

    # Get the database name
    db_name = db.name

    # Get username if available
    username = None
    if hasattr(client, "options") and hasattr(client.options, "credentials"):
        credentials = client.options.credentials
        if credentials:
            username = credentials.username

    return {"host": host, "port": port, "db_name": db_name, "username": username}


def get_connection_string(client: MongoClient):
    """
    Extract a connection string from the MongoClient.
    This avoids triggering truth value testing on Database objects.
    """
    if client.address:
        host, port = client.address
        return f"{host}:{port}"
    if hasattr(client, "address") and client.address:
        host, port = client.address
        return f"{host}:{port}"
    elif client.hosts:
        # For replica sets, return all hosts
        return ",".join(f"{host}:{port}" for host, port in client.hosts)
    elif hasattr(client, "HOST"):
        # If we can't determine hosts, use the entire URI
        parsed_uri = urlparse(client.HOST)
        return f"{parsed_uri.hostname}:{parsed_uri.port}"
    else:
        raise ValueError("Unable to determine connection string from client")


def get_connection_info(db: Database):
    """
    Extract connection information from the Database object.
    """
    # Get the name of the database
    db_name = db.name

    # Get the client's node list (this should work for single nodes and replica sets)
    node_list = db.client.nodes

    if not node_list:
        raise ValueError("Unable to determine connection information from database")

    # Use the first node in the list (for single node setups, this will be the only node)
    first_node = node_list[0]
    host, port = first_node

    return host, port, db_name


def get_auth_from_client(client: MongoClient):
    """Extract authentication details from MongoClient."""
    if hasattr(client, "_MongoClient__options"):
        # For older versions of PyMongo
        options = client._MongoClient__options
    elif hasattr(client, "options"):
        # For newer versions of PyMongo
        options = client.options
    else:
        return None, None, None

    if hasattr(options, "credentials"):
        creds = options.credentials
        return creds.username, creds.password, creds.source
    return None, None, None


def connection_from_handle(handle: str):
    if handle.startswith("mongodb://"):
        handle = handle.replace("mongodb://", "")
    host, db = handle.split("/")
    return host, db


def export_mongodb(handle: str, location: str, password: Optional[str] = None):
    host, db_name = connection_from_handle(handle)

    # Construct the mongodump command
    cmd = ["mongodump", f"--host={host}", f"--db={db_name}"]
    logger.info(f"Exporting MongoDB database {db_name} from {host} to {location}")
    cmd.extend(["--out", location])
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    logger.info(f"MongoDB export completed successfully. Output: {result.stdout}")


def import_mongodb(handle: str, dump_dir: str, drop: bool = False):
    host, db_name = connection_from_handle(handle)

    # list dirs in dump_dir
    dir_path = Path(dump_dir)
    if not dir_path.is_dir():
        raise ValueError(f"{dir_path} is not a dir")
    directories = [name for name in os.listdir(dump_dir)]
    if len(directories) != 1:
        raise ValueError(f"Expected exactly one database in {dump_dir}, got: {directories}")
    src_db_name = directories[0]

    # Construct the mongorestore command
    cmd = [
        "mongorestore",
        f"--host={host}",
        f"--nsFrom={src_db_name}.*",
        f"--nsTo={db_name}.*",
        str(dump_dir),
    ]

    # Add drop option if specified
    if drop:
        cmd.append("--drop")
    logger.info(f"CMD={cmd}")
    # Execute mongorestore
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if result.stderr:
        logger.warning(result.stderr)
    logger.info(f"MongoDB import completed successfully. Output: {result.stdout} // {result.stderr}")
