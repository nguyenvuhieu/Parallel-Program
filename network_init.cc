/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-Lenet5-cpp
 * Copyright 2018 Kai Han
 *
 * Modified for the ECE 408 class project
 * Fall 2020
 */

#include "network_init.h"

Network createNetwork_CPU(bool customCPUConv)
{
  Network Lenet5;
  
  Layer* conv1 = customCPUConv ? (Layer*)(new Conv_CPU(1, 28, 28, 6, 5, 5)) : (Layer*)(new Conv(1, 28, 28, 6, 5, 5));
  Layer* pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
  Layer* conv2 = customCPUConv ? (Layer*)(new Conv_CPU(6, 12, 12, 16, 5, 5)) : (Layer*)(new Conv(6, 12, 12, 16, 5, 5));
  Layer* pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
  Layer* fc3 = new FullyConnected(pool2->output_dim(), 120);
  Layer* fc4 = new FullyConnected(120, 84);
  Layer* fc5 = new FullyConnected(84, 10);
  Layer* relu1 = new ReLU;
  Layer* relu2 = new ReLU;
  Layer* relu3 = new ReLU;
  Layer* relu4 = new ReLU;
  Layer* softmax = new Softmax;
  Lenet5.add_layer(conv1);
  Lenet5.add_layer(relu1);
  Lenet5.add_layer(pool1);
  Lenet5.add_layer(conv2);
  Lenet5.add_layer(relu2);
  Lenet5.add_layer(pool2);
  Lenet5.add_layer(fc3);
  Lenet5.add_layer(relu3);
  Lenet5.add_layer(fc4);
  Lenet5.add_layer(relu4);
  Lenet5.add_layer(fc5);
  Lenet5.add_layer(softmax);
  // loss
  Loss* loss = new CrossEntropy;
  Lenet5.add_loss(loss);

  //load weights
  Lenet5.load_parameters("./build/cpu-train-weights.bin");
  return Lenet5;
}

Network createNetwork_GPU()
{
  Network Lenet5;

  Layer* conv1 = new Conv_Custom(1, 28, 28, 6, 5, 5);
  Layer* pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
  Layer* conv2 = new Conv_Custom(6, 12, 12, 16, 5, 5);
  Layer* pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
  Layer* fc3 = new FullyConnected(pool2->output_dim(), 120);
  Layer* fc4 = new FullyConnected(120, 84);
  Layer* fc5 = new FullyConnected(84, 10);
  Layer* relu1 = new ReLU;
  Layer* relu2 = new ReLU;
  Layer* relu3 = new ReLU;
  Layer* relu4 = new ReLU;
  Layer* softmax = new Softmax;
  Lenet5.add_layer(conv1);
  Lenet5.add_layer(relu1);
  Lenet5.add_layer(pool1);
  Lenet5.add_layer(conv2);
  Lenet5.add_layer(relu2);
  Lenet5.add_layer(pool2);
  Lenet5.add_layer(fc3);
  Lenet5.add_layer(relu3);
  Lenet5.add_layer(fc4);
  Lenet5.add_layer(relu4);
  Lenet5.add_layer(fc5);
  Lenet5.add_layer(softmax);
  // loss
  Loss* loss = new CrossEntropy;
  Lenet5.add_loss(loss);

  //load weights
  Lenet5.load_parameters("./build/cpu-train-weights.bin");
  return Lenet5;
}
