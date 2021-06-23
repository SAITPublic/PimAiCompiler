#pragma once

namespace nnr {

class ModelBuilder {
  public:
    ModelBuilder(){}

    // TODO: NNIR(LLO) should be the output
    int compileModel(std::string torch_model_path);

    int preloadModel(/*NNIR(LLO) is the input */);
  private:

    // Runnable NNIR

};

} // namespace nnr
