#!/bin/bash

# Run the training script for the 2000 era
python finetune-pyrain_2000_flood_era.py ;

# Run the training script for the 2000 era with "eranino" variant
python finetune-pyrain_2000_flood_eranino.py
