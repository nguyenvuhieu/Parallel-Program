#include "network_init.h"

void inference_only(int batch_size) {

  std::cout<<"Loading fashion-mnist data...";
  MNIST dataset("./data/");
  dataset.read_test_data(batch_size);
  std::cout<<"Done"<<std::endl;
  
  std::cout<<"Loading model...";
  Network Lenet5 = createNetwork_GPU();
  std::cout<<"Done"<<std::endl;

  Lenet5.forward(dataset.test_data);
  float acc = compute_accuracy(Lenet5.output(), dataset.test_labels);
  std::cout<<std::endl;
  std::cout<<"Test Accuracy: "<<acc<< std::endl;
  std::cout<<std::endl;
}

int main(int argc, char* argv[]) {

  int batch_size = 10000;
  
  if(argc == 2){
    batch_size = atoi(argv[1]);
  }

  std::cout<<"Test batch size: "<<batch_size<<std::endl;
  inference_only(batch_size);

  return 0;
}
