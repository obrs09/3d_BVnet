#!/bin/sh

# Usage: ./toy_train_test_3d_laplace.sh [data_name] [config_file] [train_quan_type] [test_quan_type] [load_model_path]
# Example: ./toy_train_test_3d_laplace.sh toy_data_3d configs/toy_config_3d.json float32 float32
#          ./toy_train_test_3d_laplace.sh organmnist3d configs/toy_config_3d.json float16 float32
#          ./toy_train_test_3d_laplace.sh Task06_Lung configs/lung_CT_3d_config.json bfloat16 float32
#          ./toy_train_test_3d_laplace.sh toy_data_3d configs/toy_config_3d.json float32 float32 weights/pretrained.pth

# Default values
DATA_NAME=${1:-"toy_data_3d"}
CONFIG_FILE=${2:-"configs/toy_config_3d.json"}
TRAIN_QUAN_TYPE=${3:-"float32"}
TEST_QUAN_TYPE=${4:-"float32"}
LOAD_MODEL_PATH=${5:-""}

echo "=========================================="
echo "Training with dataset: $DATA_NAME"
echo "Config file: $CONFIG_FILE"
echo "Train precision: $TRAIN_QUAN_TYPE"
echo "Test precision: $TEST_QUAN_TYPE"
if [ -n "$LOAD_MODEL_PATH" ]; then
    echo "Load model from: $LOAD_MODEL_PATH"
fi
echo "=========================================="

# Clean up previous runs (only for toy data)
if [ "$DATA_NAME" = "toy_data_3d" ]; then
    rm -rf toy_data_3d toy_weights_3d toy_images_3d toy_predict_3d
    python generate_toy_data_3d.py
fi

# Run training with specified dataset and precision
if [ -n "$LOAD_MODEL_PATH" ]; then
    python Laplace_train.py with "$CONFIG_FILE" \
        "data_name=$DATA_NAME" \
        "train_quan_type=$TRAIN_QUAN_TYPE" \
        "test_quan_type=$TEST_QUAN_TYPE" \
        "load_model_path=$LOAD_MODEL_PATH"
else
    python Laplace_train.py with "$CONFIG_FILE" \
        "data_name=$DATA_NAME" \
        "train_quan_type=$TRAIN_QUAN_TYPE" \
        "test_quan_type=$TEST_QUAN_TYPE"
fi

