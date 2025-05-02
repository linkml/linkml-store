from typing import Any, Dict, List, Optional, Tuple

ENTRY = Dict[str, Any]


def parse_sib_format(text) -> Tuple[Optional[ENTRY], List[ENTRY]]:
    """
    Parse SIB/Swiss-Prot format data into a structured dictionary.

    Args:
        text (str): The text in SIB/Swiss-Prot format

    Returns:
        dict: A dictionary with entry IDs as keys and parsed data as values
    """
    # Split the text into entries (separated by //)
    entries = text.split("//\n")
    header = None

    # Initialize results dictionary
    results = []

    # Parse each entry
    for entry in entries:
        if not entry.strip():
            continue

        # Initialize dictionary for current entry
        current_entry = {}
        current_code = None

        # Process each line
        for line in entry.strip().split("\n"):
            if not line.strip():
                continue

            # Check if this is a new field (starts with a 2-letter code followed by space)
            if len(line) > 2 and line[2] == " ":
                current_code = line[0:2]
                # Remove the code and the following space(s)
                value = line[3:].strip()

                # Initialize as list if needed for multi-line fields
                if current_code not in current_entry:
                    current_entry[current_code] = []

                current_entry[current_code].append(value)

            # Continuation of previous field
            elif current_code is not None:
                # Handle continuation lines (typically indented)
                if current_code == "CC":
                    # For comments, preserve the indentation
                    current_entry[current_code].append(line)
                else:
                    # For other fields, strip and append
                    current_entry[current_code].append(line.strip())

        # Combine multiline comments; e.g
        # -!- ...
        #     ...
        # -!- ...
        ccs = current_entry.get("CC", [])
        new_ccs = []
        for cc in ccs:
            if not cc.startswith("-!-") and new_ccs:
                new_ccs[-1] += " " + cc
            else:
                new_ccs.append(cc)
        current_entry["CC"] = new_ccs
        for k, vs in current_entry.items():
            if k != "CC":
                combined = "".join(vs)
                combined = combined.strip()
                if combined.endswith("."):
                    combined = combined.split(".")
                    combined = [c.strip() for c in combined if c.strip()]
                    if k == "DE":
                        combined = combined[0]
                current_entry[k] = combined

        if "ID" in current_entry:
            results.append(current_entry)
        else:
            header = current_entry

    return header, results


# Example usage:
# data = parse_sib_format(text)
# for entry_id, entry_data in data.items():
#     print(f"Entry: {entry_id}")
#     for code, values in entry_data.items():
#         print(f"  {code}: {values}")
