# Hardware-Acceleration-of-FLITC-algorithm

This repository contains the implementation and hardware acceleration of a machine-learning model for fault detection in power distribution grids. The project focuses on deploying the Fault Location, Identification, and Type Classification (FLITC) algorithm on hardware devices to improve execution speed with minimal accuracy loss.

The algorithm employs Convolutional Neural Networks (CNNs) to detect faulty feeders and branches.
Fault types are classified into eleven categories, and the fault location is estimated.

Feeders Fault Detection: An FFNN model is trained with a dedicated dataset to detect faulty feeders.
A validation dataset is used to evaluate the model's accuracy.

Three quantization methods were tested to optimize model size and processing latency:
-TensorFlow-Lite Dynamic Range Quantization
-TensorFlow-Lite Full Integer Quantization
-Vitis-AI TensorFlow-2 Quantization

FPGA Implementation: The quantized models are deployed on the Xilinx Zynq UltraScale+ MPSoC ZCU104 FPGA platform.
Execution leverages the Deep Learning Processing Unit (DPU).

Performance Comparison: The FPGA implementation is compared against CPU and Raspberry Pi executions of the TensorFlow Lite models.
Multithreading is explored using two DPUs to further reduce latency.

Future work: 
For implementing the remaining models of the FLITC algorithm, the same process can be used. The models, responsible for fault location, identification, and classification, should be trained, validated, and quantized. The quantized models can be deployed on the Xilinx Zynq UltraScale+ MPSoC ZCU104 FPGA and tested for performance.
