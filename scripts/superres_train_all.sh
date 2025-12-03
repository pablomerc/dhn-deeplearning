#!/bin/bash
##################################################
# Super-Resolution Training Script
# Trains all 6 models: 3 systems × 2 methods (DHN + CNN)
##################################################

EXP_CLASS=superres

# Number of epochs (set low for testing, 200 for full training)
NUM_EPOCHS=${1:-200}

# Array of all experiments
EXPERIMENTS=(
    "sinpend_4x"
    "sinpend_baseline_cnn_4x"
    "doupend_4x"
    "doupend_baseline_cnn_4x"
    "two_body_4x"
    "two_body_baseline_cnn_4x"
)

echo "=============================================="
echo "Super-Resolution Training"
echo "Epochs: ${NUM_EPOCHS}"
echo "Started at: $(date)"
echo "=============================================="

# Loop over all experiments
for EXP_NAME in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training: ${EXP_NAME}"
    echo "=========================================="
    
    RESULT_DIR=results/${EXP_CLASS}/${EXP_NAME}
    
    # Remove old results
    rm -rf ${RESULT_DIR}
    
    python main.py \
        --config=configs/${EXP_CLASS}/${EXP_NAME}.py \
        --mode=train \
        --config.workdir=${RESULT_DIR} \
        --config.optim.num_epochs=${NUM_EPOCHS}
    
    echo "Completed: ${EXP_NAME}"
done

echo ""
echo "=============================================="
echo "Training finished at: $(date)"
echo "All experiments completed!"
echo "=============================================="

