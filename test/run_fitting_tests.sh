#!/bin/bash

#Run fitting tests
cd fitting_tests
python test_lbfgs_fit.py
python test_lsr1_fit.py
cd ..
