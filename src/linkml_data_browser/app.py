import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition

from linkml_store import Client
from linkml_store.api import Collection
from linkml_store.api.queries import QueryResult

logger = logging.getLogger(__name__)

# Set page config to make layout "wide" by default
st.set_page_config(layout="wide")

config = None
if os.environ.get("LINKML_STORE_CONFIG"):
    with open(os.environ["LINKML_STORE_CONFIG"], "r") as f:
        config = yaml.safe_load(f)

# Initialize client
client = Client().from_config(config) if config else Client()

DEFAULT_LIMIT = 25

TABLES = ["gaf_association", "gaf_association_plus_violations_m", "gpi"]
DBS = {
    "mgi": TABLES,
    "goa_human": TABLES,
    "gcrp": TABLES,
}

DB_PATH = "db/{db}.db"


def init_reset_filters(cd: ClassDefinition, reset=False):
    for att_name in cd.attributes:
        key = f"filter_{att_name}"
        if reset or key not in st.session_state:
            st.session_state[key] = ""  # Assuming text input, adjust for other types


def apply_filters(collection: Collection, filters: Dict[str, Any], offset: int, limit: int, **kwargs) -> QueryResult:
    print(f"FILTERS={filters} // offset={offset}")
    qr = collection.find(filters, offset=offset, limit=limit, **kwargs)
    print(f"QR={qr.num_rows}")
    return qr


def render_filter_widget(collection: Collection, attribute: SlotDefinition):
    """Render appropriate Streamlit widget based on column type."""
    logger.info(f"Rendering filter widget: {attribute.name}")
    # print(f"{attribute.name} // RANGE={attribute.range}")
    # col_type = attribute.range
    col_name = attribute.name
    cols = st.sidebar.columns([1, 10])
    with cols[0]:
        if st.button("▼", key=f"facet_button_{col_name}", use_container_width=True):
            # Toggle the facet view state for this filter
            key = f"facet_view_{col_name}"
            if key not in st.session_state:
                st.session_state[key] = True
            else:
                st.session_state[key] = not st.session_state[key]
    with cols[1]:
        filter_value = st.text_input(f"Filter by {col_name}", key=f"filter_{col_name}")
    return filter_value
    # return st.sidebar.text_input(f"Filter by {col_name}")
    # elif col_type == "integer":
    #     max_value = con.execute(f"SELECT MAX({col_name}) FROM {tbl_name}").fetchall()[0][0]
    #     min_value = con.execute(f"SELECT MIN({col_name}) FROM {tbl_name}").fetchall()[0][0]
    #     return st.sidebar.slider(f"Filter by {col_name}", min_value, max_value, (min_value, max_value))
    # elif col_type in ['float', 'double']:
    #     max_value = con.execute(f"SELECT MAX({col_name}) FROM {tbl_name}").fetchall()[0][0]
    #     min_value = con.execute(f"SELECT MIN({col_name}) FROM {tbl_name}").fetchall()[0][0]
    #     return st.sidebar.slider(f"Filter by {col_name}", float(min_value), float(max_value),
    #                              (float(min_value), float(max_value)))
    # Add more data types as needed


# Main function to render the app
def main():
    st.title("LinkML Table Browser")
    db_names = list(client.databases.keys())
    selected_db = st.selectbox("Select a Database", db_names, key="db_selector")
    print(f"DB SELECTED={selected_db}")
    # con = duckdb.connect(DB_PATH.format(db=selected_db))
    # db_name = DB_PATH.format(db=selected_db)
    # database = DuckDBDatabase(f"duckdb:///{db_name}")
    database = client.get_database(selected_db)
    st.write(f"Connected to {selected_db}")
    candidate_tables = database.list_collection_names()
    print(f"COLLECtiONS={candidate_tables}")
    if len(candidate_tables) > 1:
        curr_table = st.selectbox("Select a Table", candidate_tables, key="table_selector")
    else:
        curr_table = candidate_tables[0]
    collection = database.get_collection(curr_table)
    print(f"CURR={collection.alias} // {collection.target_class_name}")
    cd = collection.class_definition()
    print(f"CD={cd.name} // {len(cd.attributes)}")
    filters = {}

    # Pagination setup
    session_state = st.session_state
    if "current_page" not in session_state:
        print(f"RESETTING CP// {session_state}")
        session_state.current_page = 0  # Start with page 0
    rows_per_page = DEFAULT_LIMIT

    init_reset_filters(cd)

    # Track if any filter has changed
    filter_changed = False
    for att in cd.attributes.values():
        att_name = att.name
        key = f"filter_{att_name}"
        prev_value = st.session_state[key]
        filter_widget = render_filter_widget(collection, att)
        if filter_widget is not None and filter_widget != "":
            filters[att_name] = filter_widget
        new_value = filters.get(att_name)
        if prev_value != new_value and not (not prev_value and not new_value):
            print(f"CHANGE FOR {att_name}: {prev_value} -> {new_value}")
            filter_changed = True
            # st.session_state[key] = new_value
        facet_key = f"facet_view_{att_name}"
        if facet_key in st.session_state and st.session_state[facet_key]:
            facet_results = collection.query_facets(filters, facet_columns=[att_name])
            facet_df = pd.DataFrame(facet_results[att_name])
            st.sidebar.write(facet_df)
    # If any filter has changed, reset pagination
    if filter_changed:
        print(f"FILTER CHANGED={filter_changed}")
        st.session_state.current_page = 0  # Reset offset
    result = apply_filters(collection, filters, session_state.current_page * rows_per_page, rows_per_page)
    # if filter_changed:
    #    facet_results = collection.query_facets(filters, facet_columns=["evidence_type"])
    #    print(f"FACET={facet_results}")
    st.write(f"Number of rows: {result.num_rows}")
    st.write(f"Page: {session_state.current_page + 1}")
    print(f"SESSION STATE: {session_state}")
    filtered_data = pd.DataFrame(result.rows)

    # Pagination buttons
    cols = st.columns(4)
    prev_button, next_button, _first_button, _last_button = cols[0], cols[1], cols[2], cols[3]

    if prev_button.button("Previous"):
        if session_state.current_page > 0:
            session_state.current_page -= 1
    if next_button.button("Next"):
        print(f"NEXT: CP={session_state.current_page} RPP={rows_per_page} NR={result.num_rows}")
        # Assuming result.num_rows gives the total number of rows after filtering, not just this page's rows
        if (session_state.current_page + 1) * rows_per_page < result.num_rows:
            session_state.current_page += 1
    if prev_button.button("First"):
        session_state.current_page = 0
    if prev_button.button("Last"):
        session_state.current_page = int(result.num_rows / rows_per_page)

    # Refresh the data after pagination change
    if "current_page" in session_state:
        result = apply_filters(collection, filters, session_state.current_page * rows_per_page, rows_per_page)
        filtered_data = pd.DataFrame(result.rows)

    # Add 'id' column to the DataFrame for easier tracking of changes; incremental
    filtered_data["id"] = range(1, len(filtered_data) + 1)
    edited_df = st.data_editor(filtered_data, width=2000, num_rows="dynamic")
    original_data = filtered_data

    for index, row in edited_df.iterrows():
        if np.isnan(row["id"]):
            print(f"INSERT: {row}")

    original_ids = set(original_data["id"])
    edited_ids = set(edited_df["id"])

    deleted_ids = original_ids - edited_ids
    for deleted_id in deleted_ids:
        print(f"DELETE: {original_data[original_data['id'] == deleted_id]}")

    # For modified rows, compare values row by row where ids match
    _modified_ids = []
    for id_val in original_ids & edited_ids:  # Intersection: ids present in both
        _original_row = original_data[original_data["id"] == id_val]
        _edited_row = edited_df[edited_df["id"] == id_val]
        # Assuming you have a function to compare rows and decide if they are different
        # if not rows_are_equal(original_row, edited_row):
        #    modified_ids.append(id_val)
        #    print(f"MODIFY: {original_row} -> {edited_row}")


if __name__ == "__main__":
    main()
