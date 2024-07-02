from typing import List

from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import SlotDefinition


def path_to_attribute_list(class_name: str, path: str, schema_view: SchemaView) -> List[SlotDefinition]:
    """
    Convert a path to a list of attributes.

    :param path:
    :return:
    """
    parts = path.split(".")
    att_list = []
    while parts:
        part = parts.pop(0)
        att = schema_view.induced_slot(part, class_name)
        if not att:
            raise ValueError(f"Attribute {part} not found in class {class_name}")
        att_list.append(att)
        class_name = att.range
    return att_list
