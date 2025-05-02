from linkml_store.utils.format_utils import Format, process_file
from tests import INPUT_DIR

DAT_FILE = INPUT_DIR / "expasy-subset.dat"


def test_parse_dat():
    entries = process_file(open(DAT_FILE), Format.DAT)
    assert len(entries) == 2
    e1 = entries[0]
    dr1 = e1["DR"]
    assert dr1.endswith("Q46856, YQHD_ECOLI ;")
    de1 = e1["DE"]
    assert de1 == "alcohol dehydrogenase (NADP(+))"
    cc1 = e1["CC"]
    assert len(cc1) == 4
