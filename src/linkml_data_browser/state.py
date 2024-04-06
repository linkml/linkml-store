from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from linkml_data_browser.cart import Cart
from linkml_store.api import Database


@dataclass
class PageState:
    predicted_object: Any = None
    results: List = None
    selected: Any = None


@dataclass
class ApplicationState:
    page: Optional[str] = None
    client: Database = None
    cart: Cart = field(default_factory=Cart)
    page_states: Dict[str, PageState] = field(default_factory=dict)
    # selected: Any = None

    def get_page_state(self, page_name: str) -> PageState:
        if page_name not in self.page_states:
            self.page_states[page_name] = PageState()
        return self.page_states[page_name]


def get_state(st) -> ApplicationState:
    """
    Gets the application state from the streamlit session state

    :param st:
    :return:
    """
    if "state" not in st.session_state:
        st.session_state["state"] = ApplicationState()
    return st.session_state["state"]
