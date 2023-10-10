
#!/bin/bash

set -e


ID_GPU=0

EXPS=(
    ############################
    # Sticky Trap Insects
    ############################
    # DETECTION
    # sticky-trap-insects/detection/fasterrcnn/fasterrcnn_256
    # sticky-trap-insects/detection/fasterrcnn/fasterrcnn_320
    # sticky-trap-insects/detection/fasterrcnn/fasterrcnn_480
    # sticky-trap-insects/detection/fasterrcnn/fasterrcnn_640
    # sticky-trap-insects/detection/fasterrcnn/fasterrcnn_800
    # sticky-trap-insects/detection/fcos/fcos_256
    # sticky-trap-insects/detection/fcos/fcos_320
    # sticky-trap-insects/detection/fcos/fcos_480
    # sticky-trap-insects/detection/fcos/fcos_640
    # sticky-trap-insects/detection/fcos/fcos_800
    # DENSITY
    #sticky-trap-insects/density/csrnet/csrnet_256
    # sticky-trap-insects/density/csrnet/csrnet_320
    # sticky-trap-insects/density/csrnet/csrnet_480
    # sticky-trap-insects/density/csrnet/csrnet_640
    # sticky-trap-insects/density/csrnet/csrnet_800
    # sticky-trap-insects/density/fcrn/fcrn_256
    # sticky-trap-insects/density/fcrn/fcrn_320
    # sticky-trap-insects/density/fcrn/fcrn_480
    # sticky-trap-insects/density/fcrn/fcrn_640
    # sticky-trap-insects/density/fcrn/fcrn_800
    # # SEGMENTATION
    sticky-trap-insects/segmentation/unet/unet_256
    # sticky-trap-insects/segmentation/unet/unet_320
    # sticky-trap-insects/segmentation/unet/unet_480
    # sticky-trap-insects/segmentation/unet/unet_640
    # sticky-trap-insects/segmentation/unet/unet_800
)

# Train
for EXP in ${EXPS[@]}; do
    CUDA_VISIBLE_DEVICES=${ID_GPU} HYDRA_FULL_ERROR=1 python train.py experiment=$EXP
done
