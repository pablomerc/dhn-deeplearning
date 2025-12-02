# DATA_NAME=single_pendulum
#DATA_NAME=double_pendulum
DATA_NAME=two_body

SAVE_DIR=data/${DATA_NAME}/test

echo "Data generation started at: $(date)"

python data_gen/main.py \
--seed=1 \
--config=data_gen/configs/${DATA_NAME}.py \
--config.save_dir=${SAVE_DIR} \
--config.num_data=200

echo "Data generation finished at: $(date)"
