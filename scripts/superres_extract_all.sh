#!/bin/bash
##################################################
# Super-Resolution Extraction Script
# Runs extraction for both same-init and diff-init conditions
##################################################

EXP_CLASS=superres

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
echo "Super-Resolution Extraction"
echo "Started at: $(date)"
echo "=============================================="

# Loop over all experiments
for EXP_NAME in "${EXPERIMENTS[@]}"; do
    RESULT_DIR=results/${EXP_CLASS}/${EXP_NAME}
    
    # Check if model checkpoint exists
    if [ ! -f "${RESULT_DIR}/checkpoint.pth" ]; then
        echo "Skipping ${EXP_NAME}: No checkpoint found"
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "Extracting: ${EXP_NAME}"
    echo "=========================================="
    
    # Same-init extraction (in-distribution)
    echo "--- Same Init (In-Distribution) ---"
    rm -rf ${RESULT_DIR}/extract
    
    python main.py \
        --config=configs/${EXP_CLASS}/${EXP_NAME}.py \
        --mode=extract \
        --config.workdir=${RESULT_DIR} \
        --config.model.num_embeddings=200 \
        --config.logging.num_eval_batches=1000000 \
        --config.data.batch_size=100 \
        --config.optim.num_epochs=1000 \
        --config.optim.lr=1e-2
    
    # Diff-init extraction (out-of-distribution)
    echo "--- Diff Init (Out-of-Distribution) ---"
    rm -rf ${RESULT_DIR}/extract_ood
    
    python main.py \
        --config=configs/${EXP_CLASS}/${EXP_NAME}.py \
        --mode=extract \
        --config.workdir=${RESULT_DIR} \
        --config.model.num_embeddings=200 \
        --config.logging.num_eval_batches=1000000 \
        --config.data.batch_size=100 \
        --config.optim.num_epochs=1000 \
        --config.optim.lr=1e-2 \
        --config.model.train_step_span='(512,1025)' \
        --work_subdir=extract_ood
    
    echo "Completed extraction: ${EXP_NAME}"
done

echo ""
echo "=============================================="
echo "Extraction finished at: $(date)"
echo "All extractions completed!"
echo "=============================================="

