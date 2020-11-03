#Incomplete
#User input -row, column and mouse clicks
#File comparison
import pytest
import sys
sys.path.append("..")
from get_points import *
import filecmp
import os
from input_test_base import *

def test_get_markings(mocker):
    mocker.patch('builtins.input', side_effect=["2", "2"])
    test = get_markings()

    reference_outfile1 = "./SupportingFiles/corner_points.txt"
    output_filename1 = "./SupportingFiles/test_corner_points.txt"

    assert is_same_file(output_filename1, reference_outfile1) is True
    os.remove("./SupportingFiles/test_corner_points.txt")

def is_same_file(file1, file2):
    return filecmp.cmp(file1, file2)