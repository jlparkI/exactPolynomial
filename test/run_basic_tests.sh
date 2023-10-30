#!/bin/bash

#Run data handler tests
cd basic_dataset_tests
python basic_dataset_tests.py
cd ..

#Run fht tests
cd fht_operations_tests
python exact_poly.py
cd ..
