cpu:		cpu.o custom-cpu
				nvcc -o cpu -lm -lcuda -lrt cpu.o network_init.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/layer/custom/cpu-forward.o src/layer/custom/Parallel_v1.o src/layer/custom/gpu-utils.o -I ../libgputk/ -I./

cpu.o:		cpu.cc
		nvcc --compile cpu.cc -I ../libgputk/ -I./

network_init.o:    network_init.cc
		nvcc --compile network_init.cc -I ../libgputk/ -I./

network.o:	src/network.cc
		nvcc --compile src/network.cc -o src/network.o -I ../libgputk/ -I./

mnist.o:	src/mnist.cc
		nvcc --compile src/mnist.cc -o src/mnist.o -I ../libgputk/ -I./

layer:		src/layer/conv.cc src/layer/ave_pooling.cc src/layer/conv_cpu.cc src/layer/conv_cust.cc src/layer/fully_connected.cc src/layer/max_pooling.cc src/layer/relu.cc src/layer/sigmoid.cc src/layer/softmax.cc 
		nvcc --compile src/layer/ave_pooling.cc -o src/layer/ave_pooling.o -I ../libgputk/ -I./
		nvcc --compile src/layer/conv.cc -o src/layer/conv.o -I ../libgputk/ -I./
		nvcc --compile src/layer/conv_cpu.cc -o src/layer/conv_cpu.o -I ../libgputk/ -I./
		nvcc --compile src/layer/conv_cust.cc -o src/layer/conv_cust.o -I ../libgputk/ -I./
		nvcc --compile src/layer/fully_connected.cc -o src/layer/fully_connected.o -I ../libgputk/ -I./
		nvcc --compile src/layer/max_pooling.cc -o src/layer/max_pooling.o -I ../libgputk/ -I./
		nvcc --compile src/layer/relu.cc -o src/layer/relu.o -I ../libgputk/ -I./
		nvcc --compile src/layer/sigmoid.cc -o src/layer/sigmoid.o -I ../libgputk/ -I./
		nvcc --compile src/layer/softmax.cc -o src/layer/softmax.o -I ../libgputk/ -I./

custom-cpu:
		nvcc --compile src/layer/custom/cpu-forward.cc -o src/layer/custom/cpu-forward.o -I ../libgputk/ -I./
		nvcc --compile src/layer/custom/gpu-utils.cu -o src/layer/custom/gpu-utils.o -I ../libgputk/ -I./
		nvcc --compile src/layer/custom/Parallel_v1.cu -o src/layer/custom/Parallel_v1.o -I ../libgputk/ -I./

		
loss:           src/loss/cross_entropy_loss.cc src/loss/mse_loss.cc
		nvcc --compile src/loss/cross_entropy_loss.cc -o src/loss/cross_entropy_loss.o -I ../libgputk/ -I./
		nvcc --compile src/loss/mse_loss.cc -o src/loss/mse_loss.o -I ../libgputk/ -I./


run: 		cpu
		./cpu 1000

custom_v1:
		nvcc --compile src/layer/custom/gpu-utils.cu -o src/layer/custom/gpu-utils.o -I ../libgputk/ -I./
		nvcc --compile src/layer/custom/Parallel_v1.cu -o src/layer/custom/Parallel_v1.o -I ../libgputk/ -I./

parallel_v1:		parallel_v1.o custom_v1
		nvcc -o parallel_v1 -lm -lcuda -lrt parallel_v1.o network_init.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/layer/custom/Parallel_v1.o src/layer/custom/gpu-utils.o src/layer/custom/cpu-forward.o -I ../libgputk/ -I./

parallel_v1.o:		parallel_v1.cc
		nvcc --compile parallel_v1.cc -I ../libgputk/ -I./

run_v1:	parallel_v1 parallel_v1.o
		./parallel_v1 1000

custom_v2:
		nvcc --compile src/layer/custom/gpu-utils.cu -o src/layer/custom/gpu-utils.o -I ../libgputk/ -I./
		nvcc --compile src/layer/custom/Parallel_v2.cu -o src/layer/custom/Parallel_v2.o -I ../libgputk/ -I./

parallel_v2:		parallel_v2.o custom_v2
		nvcc -o parallel_v2 -lm -lcuda -lrt parallel_v2.o network_init.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/layer/custom/Parallel_v2.o src/layer/custom/gpu-utils.o src/layer/custom/cpu-forward.o -I ../libgputk/ -I./

parallel_v2.o:		parallel_v2.cc
		nvcc --compile parallel_v2.cc -I ../libgputk/ -I./


run_v2:	parallel_v2 parallel_v2.o
		./parallel_v2 1000

custom_v3:
		nvcc --compile src/layer/custom/gpu-utils.cu -o src/layer/custom/gpu-utils.o -I ../libgputk/ -I./
		nvcc --compile src/layer/custom/Parallel_v3.cu -o src/layer/custom/Parallel_v3.o -I ../libgputk/ -I./

parallel_v3:		parallel_v3.o custom_v3
		nvcc -o parallel_v3 -lm -lcuda -lrt parallel_v3.o network_init.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/layer/custom/Parallel_v3.o src/layer/custom/gpu-utils.o src/layer/custom/cpu-forward.o -I ../libgputk/ -I./

parallel_v3.o:		parallel_v3.cc
		nvcc --compile parallel_v3.cc -I ../libgputk/ -I./


run_v3:	parallel_v3 parallel_v3.o
		./parallel_v3 1000

custom_v4:
		nvcc --compile src/layer/custom/gpu-utils.cu -o src/layer/custom/gpu-utils.o -I ../libgputk/ -I./
		nvcc --compile src/layer/custom/Parallel_v4.cu -o src/layer/custom/Parallel_v4.o -I ../libgputk/ -I./

parallel_v4:		parallel_v4.o custom_v4
		nvcc -o parallel_v4 -lm -lcuda -lrt parallel_v4.o network_init.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/layer/custom/Parallel_v4.o src/layer/custom/gpu-utils.o src/layer/custom/cpu-forward.o -I ../libgputk/ -I./

parallel_v4.o:		parallel_v4.cc
		nvcc --compile parallel_v4.cc -I ../libgputk/ -I./


run_v4:	parallel_v4 parallel_v4.o
		./parallel_v4 1000

