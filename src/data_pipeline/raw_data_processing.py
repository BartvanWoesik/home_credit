from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import pandas as pd
import os
import hydra
from hydra.utils import instantiate
from functools import partial
import numpy as np
from data_pipeline.pipes import get_last, get_mode  


DATA_PATH = Path(os.getcwd()) / "data/parquet_files/train"
BASE_FILE = "train_base.parquet"
ID = "case_id"
TARGET = "target"
DATE_DECISION = 'date_decision'

import os

os.environ['HYDRA_FULL_ERROR'] = '1'


def create_aggration_dataframe(cfg: dict, df: pd.DataFrame, itteration: int) -> pd.DataFrame:
    if len(cfg.time_col) > 0:
        df = df[df[cfg.time_col[0]] < df[DATE_DECISION]]
    aggregation_specs = {}
    for column in cfg.agg_columns:
        aggregation_specs[f'{column.base_feature_name}_{itteration}'] = (column.name , instantiate(column.aggregation))
    return df.groupby(ID).agg(**aggregation_specs)



def get_orderd_data_files(data_path: Path, all_file_sources: list[str]) -> list[dict]:
    file_dict = []
    all_files =  os.listdir(data_path)
    for source in all_file_sources:
        filtered_files = filter(lambda x, source=source: source in x, all_files)
        filtered_files = list(filter(None, filtered_files))
        file_dict.append({source: filtered_files})
        
    return file_dict


@hydra.main(config_path="../../", config_name="config.yaml")
def main(cfg: DictConfig) -> pd.DataFrame:
    all_file_sources = list(cfg.data.keys())
    data_structure = get_orderd_data_files(DATA_PATH, all_file_sources = all_file_sources)
    df_base = pd.read_parquet(DATA_PATH / BASE_FILE)
    df  = pd.DataFrame(df_base[[ID, TARGET]])
    for source_file in data_structure:

        source_name = list(source_file.keys())[0]
        files = list(source_file.values())[0]
        cfg_columns = cfg.data[source_name]
        for i, file in enumerate(files):
            print(file)
            df_file = pd.read_parquet(DATA_PATH / file)
            if len(cfg_columns.time_col) > 0:
                df_file = df_file[[ID] +[ col.name for col in cfg_columns.agg_columns] + [cfg_columns.time_col[0]]]
            else: 
                df_file = df_file[[ID] +[ col.name for col in cfg_columns.agg_columns]]
            df_combined = df_base.merge(df_file, on = ID, how = 'left', validate="one_to_many")
            df_agg = create_aggration_dataframe(cfg_columns, df_combined, i)
    
            df = df.merge(df_agg, on = ID, how ='left')
    print('Write df to feather')
    df.to_feather(DATA_PATH / 'processed_train.feather')

if __name__ == "__main__":
    main()
 