import pandas as pd
import pytest

from asi_project.data import DataContractError
from asi_project.pipelines.data_processing.nodes import (
    filter_data,
    normalize_data,
    validate_data,
)


def test_validate_data(project):
    result = validate_data(project.sample)
    pd.testing.assert_frame_equal(result, project.sample)


def test_validate_data_missing_column(project):
    df = project.sample.drop(columns=[project.contract.target])

    with pytest.raises(DataContractError):
        validate_data(df)


def test_normalize_data():
    df = pd.DataFrame({"color": [" brown ", "white  ", ""]})

    result = normalize_data(df)

    assert result["color"].iloc[0] == "brown"
    assert result["color"].iloc[1] == "white"
    assert pd.isna(result["color"].iloc[2])


def test_filter_data(project):
    df = project.sample.copy()
    target = project.contract.target

    df.loc[df.index[0], target] = None

    result = filter_data(df)

    assert len(result) == len(df) - 1
    assert result[target].notna().all()
