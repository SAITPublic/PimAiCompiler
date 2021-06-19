#pragma once

namespace nnr {

class NNRuntime {
  public:
    NNRuntime(){}

    NNRuntime(std::string torch_model_path);

    void inferenceModel();
  
  private:
    void compileModel();

    void preloadModel();

};

}
