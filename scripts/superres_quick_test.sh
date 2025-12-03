#!/bin/bash
##################################################
# Quick Test: Super-Resolution Pipeline
# Runs a minimal test (5 epochs, single system) to verify everything works
##################################################

EXP_CLASS=superres
NUM_EPOCHS=5
EXP_NAME=sinpend_4x

echo "=============================================="
echo "Quick Test: Super-Resolution Pipeline"
echo "System: Single Pendulum (DHN)"
echo "Epochs: ${NUM_EPOCHS}"
echo "Started at: $(date)"
echo "=============================================="

RESULT_DIR=results/${EXP_CLASS}/${EXP_NAME}

# Clean up previous results
rm -rf ${RESULT_DIR}

# Train
echo ""
echo "--- Training ---"
python main.py \
    --config=configs/${EXP_CLASS}/${EXP_NAME}.py \
    --mode=train \
    --config.workdir=${RESULT_DIR} \
    --config.optim.num_epochs=${NUM_EPOCHS} \
    --config.logging.per_save_epochs=${NUM_EPOCHS} \
    --config.logging.per_eval_epochs=1

# Quick extraction test (fewer epochs)
echo ""
echo "--- Extraction (Same Init) ---"
rm -rf ${RESULT_DIR}/extract

python main.py \
    --config=configs/${EXP_CLASS}/${EXP_NAME}.py \
    --mode=extract \
    --config.workdir=${RESULT_DIR} \
    --config.model.num_embeddings=200 \
    --config.logging.num_eval_batches=10 \
    --config.data.batch_size=50 \
    --config.optim.num_epochs=10 \
    --config.optim.lr=1e-2

echo ""
echo "=============================================="
echo "Quick test finished at: $(date)"
echo "If no errors, pipeline is working!"
echo "=============================================="

