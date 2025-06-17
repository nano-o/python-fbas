"""
Global config for the python_fbas package.
"""
from typing import Literal, Optional

sat_solver:str = 'cryptominisat5'
card_encoding:Literal['naive','totalizer'] = 'totalizer'
max_sat_algo:Literal['LSU','RC2'] = 'LSU'
output:Optional[str] = None
group_by:Optional[str] = None
