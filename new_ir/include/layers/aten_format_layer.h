#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenFormatLayer : public NNLayer {
 public:
    AtenFormatLayer() {}

    AtenFormatLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenFormatLayer(const AtenFormatLayer &aten_format_layer) : NNLayer(aten_format_layer) {
        this->_assembly_format = aten_format_layer._assembly_format;
    }

    virtual ~AtenFormatLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<AtenFormatLayer>(new AtenFormatLayer(*this));
    }

    void setAssemblyFormat(std::string assembly_format) { _assembly_format = assembly_format; }

    std::string getAssemblyFormat() const { return _assembly_format; }

    void printAttr() {
        Log::IR::I() << "    AtenFormatAttr      ";
        Log::IR::I() << "    assembly_format is  "<< _assembly_format;
    }

 private:
    std::string _assembly_format;
};

} // namespace ir
} // namespace nn_compiler
