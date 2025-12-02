##################################################
# AutoRegression
##################################################

EXP_CLASS=ar

EXP_NAME=two_body_kernel2_stride1
#EXP_NAME=sinpend_kernel2_stride1
#EXP_NAME=sinpend_kernel4_stride2
#EXP_NAME=sinpend_kernel8_stride4

#EXP_NAME=sinpend_baseline_hnn_tf
#EXP_NAME=sinpend_baseline_vanilla_tf
#EXP_NAME=sinpend_baseline_vanilla_resnet_layer1
#EXP_NAME=sinpend_baseline_vanilla_resnet_layer2

#EXP_NAME=doupend_kernel2_stride1
#EXP_NAME=doupend_kernel4_stride2
#EXP_NAME=doupend_kernel8_stride4

#EXP_NAME=doupend_baseline_hnn_tf
#EXP_NAME=doupend_baseline_vanilla_tf
#EXP_NAME=doupend_baseline_vanilla_resnet_layer1
#EXP_NAME=doupend_baseline_vanilla_resnet_layer2


# Array of experiment names to run
EXPERIMENTS=(
    "two_body_baseline_hnn_tf"
    "two_body_kernel2_stride1"
    "two_body_kernel4_stride2"
    "two_body_kernel8_stride4"
    "two_body_baseline_vanilla_tf"
    "two_body_baseline_resnet_layer1"
    "two_body_baseline_resnet_layer2"
)

echo "Training started at: $(date)"
echo "Running ${#EXPERIMENTS[@]} experiments"


#new: dataset_split=test, work_subdir=gen_sequence_test, rm -rf ${RESULT_DIR}/gen_sequence_test (added _test)
# Loop over all experiments
for EXP_NAME in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Starting experiment: ${EXP_NAME}"
    echo "=========================================="

    RESULT_DIR=results/${EXP_CLASS}/${EXP_NAME}

    rm -rf ${RESULT_DIR}/gen_sequence_test

    echo "Generation started at: $(date)"

    python main.py \
    --config=configs/${EXP_CLASS}/${EXP_NAME}.py \
    --mode=generate \
    --dataset_split=test \
    --work_subdir=gen_sequence_test \
    --config.workdir=${RESULT_DIR} \
    --config.data.batch_size=1000

    echo "Generation finished at: $(date)"
    echo "Completed experiment: ${EXP_NAME}"
done

echo ""
echo "Generation finished at: $(date)"
echo "All experiments completed!"
