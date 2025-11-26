##################################################
# AutoRegression
##################################################

EXP_CLASS=ar

# EXP_NAME=sinpend_kernel2_stride1
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

# EXP_NAME=two_body_kernel2_stride1

##################################################
# Representation Learning
##################################################

#EXP_CLASS=repn

#EXP_NAME=doupend_kernel2_stride1
#EXP_NAME=doupend_kernel4_stride2
#EXP_NAME=doupend_kernel8_stride4

#EXP_NAME=doupend_kernel4_stride1
#EXP_NAME=doupend_kernel4_stride3

#EXP_NAME=doupend_kernel8_stride1
#EXP_NAME=doupend_kernel8_stride2
#EXP_NAME=doupend_kernel8_stride3
#EXP_NAME=doupend_kernel8_stride5
#EXP_NAME=doupend_kernel8_stride6
#EXP_NAME=doupend_kernel8_stride7

#EXP_NAME=doupend_baseline_hnn_tf
#EXP_NAME=doupend_baseline_vanilla_tf
#EXP_NAME=doupend_baseline_vanilla_resnet_layer1
#EXP_NAME=doupend_baseline_vanilla_resnet_layer2

##################################################
# Super-Resolution
##################################################

#EXP_CLASS=superres

#EXP_NAME=sinpend_4x
#EXP_NAME=sinpend_baseline_cnn_4x

#EXP_NAME=doupend_4x
#EXP_NAME=doupend_baseline_cnn_4x

# Array of experiment names to run
EXPERIMENTS=(
    "two_body_baseline_hnn_tf"
    "two_body_kernel4_stride2"
    "two_body_kernel8_stride4"
    "two_body_baseline_vanilla_tf"
    "two_body_baseline_resnet_layer1"
    "two_body_baseline_resnet_layer2"
    "two_body_kernel2_stride1"
)

echo "Training started at: $(date)"
echo "Running ${#EXPERIMENTS[@]} experiments"

# Loop over all experiments
for EXP_NAME in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Starting experiment: ${EXP_NAME}"
    echo "=========================================="

    RESULT_DIR=results/${EXP_CLASS}/${EXP_NAME}

    rm -rf ${RESULT_DIR}

    python main.py \
    --config=configs/${EXP_CLASS}/${EXP_NAME}.py \
    --mode=train \
    --config.workdir=${RESULT_DIR}

    echo "Completed experiment: ${EXP_NAME}"
done

echo ""
echo "Training finished at: $(date)"
echo "All experiments completed!"
