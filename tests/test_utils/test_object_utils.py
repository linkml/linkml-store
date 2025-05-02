import pytest

from linkml_store.utils.object_utils import object_path_update


@pytest.mark.parametrize(
    "input,path,value,expected",
    [
        ({}, "foo", 1, {"foo": 1}),
        ({"foo": 2}, "foo", 1, {"foo": 1}),
        ({"foo": None}, "foo", 1, {"foo": 1}),
        ({"foo": {"bar": 1}}, "foo", 1, {"foo": 1}),
        ({}, "foo.bar", 1, {"foo": {"bar": 1}}),
        ({"foo": {}}, "foo.bar", 1, {"foo": {"bar": 1}}),
        ({"foo": None}, "foo.bar", 1, {"foo": {"bar": 1}}),
        ({"foo": {"xyz": 2}}, "foo.bar", 1, {"foo": {"xyz": 2, "bar": 1}}),
        ({"foo": [2]}, "foo[0]", 1, {"foo": [1]}),
        ({}, "persons[0].foo.bar", 1, {"persons": [{"foo": {"bar": 1}}]}),
        ({}, "persons[1].foo.bar", 1, {"persons": [{}, {"foo": {"bar": 1}}]}),
        ({"persons": [{"foo": {"bar": 1}}]}, "persons[0].foo.bar", 1, {"persons": [{"foo": {"bar": 1}}]}),
        ({"persons": [{"foo": {"bar": 1}}]}, "persons[0].foo.baz", 2, {"persons": [{"foo": {"bar": 1, "baz": 2}}]}),
        (
            {"persons": [{"foo": {"bar": 1}}]},
            "persons[1].foo.baz",
            2,
            {"persons": [{"foo": {"bar": 1}}, {"foo": {"baz": 2}}]},
        ),
    ],
)
def test_update_nested_object(input, path, value, expected):
    result = object_path_update(input, path, value)
    assert result == expected
