
import pandas as pd

from asi_project.config import get_contract, load_project_config
from asi_project.data import assert_valid, filter_training_rows, normalize_input


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    config = load_project_config()
    contract = get_contract(config)

    assert_valid(df, contract, config)
    return df.copy()


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_input(df)


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    config = load_project_config()
    contract = get_contract(config)

    return filter_training_rows(df, contract)
