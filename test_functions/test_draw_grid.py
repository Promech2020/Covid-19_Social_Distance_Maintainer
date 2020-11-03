#Incomplete
#User inputs
#File comparison


import pytest
import filecmp
import os
import sys
sys.path.append("..")
from draw_grid import *

def test_get_essential_data():
    reference_outfile1 = "./SupportingFiles/background_data.txt"
    output_filename1 = "./SupportingFiles/test_background_data.txt"
    get_essential_data()

    assert is_same_file(output_filename1, reference_outfile1) is True
    os.remove("./SupportingFiles/test_background_data.txt")

def is_same_file(file1, file2):
    return filecmp.cmp(file1, file2)