#!/bin/bash
##################################################
# Generate Data for Super-Resolution Experiments
# Generates train and test data for all 3 systems
##################################################

echo "=============================================="
echo "Data Generation for Super-Resolution"
echo "Started at: $(date)"
echo "=============================================="

# Systems to generate data for
SYSTEMS=("single_pendulum" "double_pendulum" "two_body")

for DATA_NAME in "${SYSTEMS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Generating data for: ${DATA_NAME}"
    echo "=========================================="
    
    # Generate training data (1000 trajectories)
    echo "--- Training Data ---"
    SAVE_DIR=data/${DATA_NAME}/train
    python data_gen/main.py \
        --seed=0 \
        --config=data_gen/configs/${DATA_NAME}.py \
        --config.save_dir=${SAVE_DIR}
    
    # Generate test data (200 trajectories)
    echo "--- Test Data ---"
    SAVE_DIR=data/${DATA_NAME}/test
    python data_gen/main.py \
        --seed=1 \
        --config=data_gen/configs/${DATA_NAME}.py \
        --config.save_dir=${SAVE_DIR} \
        --config.num_data=200
    
    echo "Completed: ${DATA_NAME}"
done

echo ""
echo "=============================================="
echo "Data generation finished at: $(date)"
echo "All systems completed!"
echo "=============================================="



