# dataclass for configuration settings
from dataclasses import dataclass, field
from typing import List, Optional
@dataclass
class Config:
    
    train_data_path: str = "../dataset/train.parquet"
    test_data_path: str = "../dataset/test.parquet"
    
    