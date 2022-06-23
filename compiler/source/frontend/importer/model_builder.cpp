#include <set>

#include "frontend/importer/model_builder.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{
void ModelBuilder::build(std::unique_ptr<ir::NNModel>& nn_model, const std::string& torch_model_path)
{
    torch_model_ = parseTorchScript(torch_model_path);
    // convert torch node to nn-graph layer
    torchToNNGraph(nn_model, torch_model_);

    // ir::utils::printGraphModel(nn_model);
    return;
}

std::shared_ptr<torch::jit::Module> ModelBuilder::parseTorchScript(const std::string& torch_model_path)
{
    // ref: https://pytorch.org/tutorials/advanced/cpp_export.html
    torch::jit::Module script_model;
    try {
        script_model = torch::jit::load(torch_model_path, torch::kCPU);
    } catch (c10::Error& e) {
        DLOG(FATAL) << "error loading the model";
    }
    std::shared_ptr<torch::jit::Module> torch_model = std::make_shared<torch::jit::Module>(script_model);
    return torch_model;
}

uint32_t ModelBuilder::getUniqueBlockId()
{
    block_counter_++;
    return block_counter_;
}

uint32_t ModelBuilder::getUniqueTensorId(std::unique_ptr<ir::NNModel>& nn_model)
{
    auto shape_tensor = std::make_shared<ir::TSSTensor>();
    nn_model->addTSSTensor(std::make_pair(shape_tensor->getID(), shape_tensor));
    return shape_tensor->getID();
}

void ModelBuilder::getFinalType(c10::TypePtr t, std::set<c10::TypePtr>& type_set)
{
    if (t->containedTypes().empty()) {
        type_set.insert(t);
        return;
    }
    for (auto tt : t->containedTypes()) {
        getFinalType(tt, type_set);
    }
}

void ModelBuilder::isInplaceNode(std::string op_node, std::shared_ptr<ir::NNLayer>& layer)
{
    if (op_node == "aten::unsqueeze_") {
        std::shared_ptr<ir::AtenUnsqueezeLayer> attr = std::static_pointer_cast<ir::AtenUnsqueezeLayer>(layer);
        attr->setIsInplace(true);
        return;
    }
    if (op_node == "aten::masked_fill_") {
        std::shared_ptr<ir::AtenMaskedFillLayer> attr = std::static_pointer_cast<ir::AtenMaskedFillLayer>(layer);
        attr->setIsInplace(true);
        return;
    }
}

nn_compiler::ir::DataType ModelBuilder::convertTorchScriptType(const c10::TypePtr& torch_type)
{
    std::set<c10::TypePtr> type_set;
    type_set.clear();
    getFinalType(torch_type, type_set);
    if (type_set.size() == 1) {
        // all internal types are same
        auto type = type_set.extract(type_set.begin()).value()->str();
        if (type == "Tensor") {
            return nn_compiler::ir::DataType::TENSOR;
        }
        if (type == "int") {
            return nn_compiler::ir::DataType::INT64;
        }
        if (type == "Device") {
            return nn_compiler::ir::DataType::DEVICE;
        }
        if (type == "None") {
            return nn_compiler::ir::DataType::NONE;
        }
        if (type == "Scalar") {
            return nn_compiler::ir::DataType::INT64;
        }
        if (type == "bool") {
            return nn_compiler::ir::DataType::BOOL;
        }
        if (type == "float") {
            return nn_compiler::ir::DataType::FLOAT64;
        }
        if (type == "str") {
            return nn_compiler::ir::DataType::STRING;
        }
    } else {
        return nn_compiler::ir::DataType::LIST;
    }
    return nn_compiler::ir::DataType::UNDEFINED;
}

void addAttributeToLayer(c10::IValue ival, std::shared_ptr<nn_compiler::ir::PrimVariableLayer> layer)
{
    if (ival.isList()) {
        auto value = ival.toList();
        for (c10::IValue item : value) {
            addAttributeToLayer(item, layer);
        }
        return;
    } else if (ival.isTuple()) {
        auto value = ival.toTuple();
        for (c10::IValue item : value->elements()) {
            addAttributeToLayer(item, layer);
        }
        return;
    } else {
        auto data = std::make_shared<nn_compiler::ir::DTensor>();
        data->setDataType(nn_compiler::ir::DataType::UNDEFINED);
        if (ival.isBool()) {
            auto value = ival.toBool();
            data->setData(&value, 8);
            data->setTensorShape(nn_compiler::ir::STensor(0, 0, 0, 1));
            data->setDataType(nn_compiler::ir::DataType::INT64);
            DLOG(INFO) << "set bool as int64: " << value;

        } else if (ival.isInt()) {
            auto value = ival.toInt();
            data->setData(&value, 8);
            data->setTensorShape(nn_compiler::ir::STensor(0, 0, 0, 1));
            data->setDataType(nn_compiler::ir::DataType::INT64);
            DLOG(INFO) << "set int64: " << value;

        } else if (ival.isString()) {
            auto value = ival.toString()->string();
            auto len = value.length();
            data->setData(value.c_str(), len + 1);
            data->setTensorShape(nn_compiler::ir::STensor(0, 0, 0, 1));
            data->setDataType(nn_compiler::ir::DataType::UINT8);
            DLOG(INFO) << "set str: " << value;

        } else if (ival.isNone()) {
            auto value = ival.toNone();
            auto len = value.length();
            data->setData(value.c_str(), len + 1);
            data->setTensorShape(nn_compiler::ir::STensor(0, 0, 0, len));
            data->setDataType(nn_compiler::ir::DataType::UINT8);
            DLOG(INFO) << "set None: " << value;

        } else if (ival.isDouble()) {
            auto value = ival.toDouble();
            data->setData(&value, 1 * sizeof(value));
            data->setTensorShape(nn_compiler::ir::STensor(0, 0, 0, 1));
            data->setDataType(nn_compiler::ir::DataType::FLOAT64);
            DLOG(INFO) << "set float64: " << value;

        } else if (ival.isDevice()) {
            auto value = ival.toDevice().str();
            auto len = value.length();
            data->setData(value.c_str(), len + 1);
            data->setTensorShape(nn_compiler::ir::STensor(0, 0, 0, len));
            data->setDataType(nn_compiler::ir::DataType::UINT8);
            DLOG(INFO) << "set Device: " << value;

        } else if (ival.isTensor()) {
            auto value = ival.toTensor();
            c10::ScalarType dtype = value.scalar_type();
            nn_compiler::frontend::ptTensor2DTensor(value, data);
            DLOG(INFO) << "set Tensor with size: " << value.sizes() << " dtype: " << dtype;

        } else {
            DLOG(INFO) << ival.type()->repr_str() << " is not supported yet.";
            return;
        }
        layer->setAttr(data);
        return;
    }
}
template <typename T>
void ModelBuilder::importTorchScriptMethodBlock(std::unique_ptr<ir::NNModel>& nn_model, const std::string& name,
                                                const T& method_block, bool is_main_graph)
{
    std::shared_ptr<ir::NNGraph> graph = std::make_shared<ir::NNGraph>();
    graph->setName(name);
    // set graph inputs
    for (auto input : method_block->inputs()) {
        if (input->type()->is_module()) {
            continue;
        }
        if (value_tensor_map_.find(input) == value_tensor_map_.end()) {
            // create input, it is either method input or block input
            int tensor_id = getUniqueTensorId(nn_model);
            auto shape_tensor = nn_model->getTSSTensors()[tensor_id];
            shape_tensor->setFeaturemapType(convertTorchScriptType(input->type()));
            shape_tensor->setReprType(input->type()->repr_str());
            value_tensor_map_.emplace(input, tensor_id);
            graph->addGraphInTensorID(tensor_id);
        } else {
            // get and set
            graph->addGraphInTensorID(value_tensor_map_[input]);
        }
        // Add input layer (only for main graph)
        if (is_main_graph) {
            auto input_layer_builder = this->layer_builders_.get("prim::Input");
            auto layer =
                std::dynamic_pointer_cast<nn_compiler::ir::PrimInputLayer>(input_layer_builder->buildLayer(nullptr));
            layer->setName(convertLayerTypeToString(layer->getType()) + "_" + std::to_string(layer->getID()));
            layer->addOutSTensorID(value_tensor_map_[input]);

            graph->addLayer(layer);
            nn_model->addLayerRelationShips(value_tensor_map_[input], layer);
        }
    }

    // Import nodes in the method.
    for (const auto& node : method_block->nodes()) {
        auto kind = node->kind();
        switch (kind) {
            case c10::prim::CallMethod: {
                // TODO(SRCX):
                // Will be supported in the future since now we only
                // use frozen torchscript model which does not contain this node
                break;
            }
            case c10::prim::If: {
                auto builder = this->layer_builders_.get("prim::If");
                if (builder != nullptr) {
                    auto layer = createLayer(builder, node, nn_model);

                    // Add two NNGraphs(ThenNet, ElseNet) for then/else blocks.
                    auto then_block = node->blocks()[0];
                    std::string then_block_name = name + "_" + std::to_string(getUniqueBlockId());
                    importTorchScriptMethodBlock(nn_model, then_block_name, then_block);

                    auto else_block = node->blocks()[1];
                    std::string else_block_name = name + "_" + std::to_string(getUniqueBlockId());
                    importTorchScriptMethodBlock(nn_model, else_block_name, else_block);

                    // If builder will use two NNGraphs as atrributes (ThenNet, ElseNet)
                    auto if_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimIfLayer>(layer);
                    if_layer->setThenNet(then_block_name);
                    if_layer->setElseNet(else_block_name);

                    // Add layer to graph
                    graph->addLayer(layer);
                }
                break;
            }
            case c10::prim::Loop: {
                auto builder = this->layer_builders_.get("prim::Loop");
                if (builder != nullptr) {
                    // Add a NNGraph for the body of loop
                    // Loop builder will use NNGraph as an atrribute
                    auto body_block = node->blocks()[0];
                    std::string body_block_name = name + "_" + std::to_string(getUniqueBlockId());
                    importTorchScriptMethodBlock(nn_model, body_block_name, body_block);

                    auto layer = createLayer(builder, node, nn_model);
                    auto loop_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimLoopLayer>(layer);
                    loop_layer->setBodyNet(body_block_name);

                    // Add layer to graph
                    graph->addLayer(layer);
                }
                break;
            }
            case c10::prim::SetAttr: {
                auto attr_symbol = node->attributeNames().at(0);
                std::string attrMap_key = node->inputs()[0]->type()->str() + "@" + node->s(attr_symbol);
                auto attr = module_attributes_[attrMap_key];
                if (!attr.isModule()) {
                    // Module attr is not our target, we only handle attr that has values
                    if (attr_layer_map.count(attrMap_key) == 0) {
                        // Create variable layer
                        auto variable_builder = this->layer_builders_.get("prim::Variable");
                        if (variable_builder != nullptr) {
                            auto variable_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimVariableLayer>(
                                variable_builder->buildLayer(node));
                            variable_layer->setName(convertLayerTypeToString(variable_layer->getType()) + "_" +
                                                    std::to_string(variable_layer->getID()));
                            variable_layer->setNType(attr.type()->repr_str());

                            // Store attr
                            addAttributeToLayer(attr, variable_layer);

                            // No input
                            // Add output for prim::variable layer
                            int node_output_id = getUniqueTensorId(nn_model);
                            auto shape_tensor = nn_model->getTSSTensors()[node_output_id];
                            shape_tensor->setParentLayer(variable_layer->getID());
                            shape_tensor->setFeaturemapType(convertTorchScriptType(attr.type()));
                            shape_tensor->setReprType(variable_layer->getNType());
                            variable_layer->addOutSTensorID(node_output_id);
                            attr_layer_map.emplace(attrMap_key, variable_layer);

                            graph->addLayer(variable_layer);
                            nn_model->addLayerRelationShips(node_output_id, variable_layer);
                        }
                    }
                }

                // create SetAttr Layer
                auto set_attr_builder = this->layer_builders_.get("prim::SetAttr");
                if (set_attr_builder != nullptr) {
                    auto set_attr_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimSetAttrLayer>(
                        set_attr_builder->buildLayer(node));
                    set_attr_layer->setName(convertLayerTypeToString(set_attr_layer->getType()) + "_" +
                                            std::to_string(set_attr_layer->getID()));

                    // Connect SetAttr and variable
                    auto variable_layer = attr_layer_map[attrMap_key];

                    // Add input1, target attribute
                    for (auto oid : variable_layer->getOutSTensorID()) {
                        set_attr_layer->addInSTensorID(oid);
                        nn_model->addLayerRelationShips(oid, set_attr_layer);
                    }
                    // Add input2, target value
                    for (unsigned int i = 1; i < node->inputs().size(); i++) {
                        set_attr_layer->addInSTensorID(value_tensor_map_[node->inputs()[i]]);
                        nn_model->addLayerRelationShips(value_tensor_map_[node->inputs()[i]], set_attr_layer);
                    }
                    // No output
                    for (auto node_output : node->outputs()) {
                        // always create new
                        if (value_tensor_map_.find(node_output) == value_tensor_map_.end()) {
                            int node_output_id = getUniqueTensorId(nn_model);
                            auto shape_tensor = nn_model->getTSSTensors()[node_output_id];
                            shape_tensor->setFeaturemapType(convertTorchScriptType(node_output->type()));
                            shape_tensor->setParentLayer(set_attr_layer->getID());
                            shape_tensor->setReprType(node_output->type()->repr_str());
                            value_tensor_map_.emplace(node_output, node_output_id);
                            set_attr_layer->addOutSTensorID(node_output_id);
                        } else {
                            set_attr_layer->addOutSTensorID(value_tensor_map_[node_output]);
                        }
                        nn_model->addLayerRelationShips(value_tensor_map_[node_output], set_attr_layer);
                    }
                    graph->addLayer(set_attr_layer);
                }
                break;
            }
            case c10::prim::GetAttr: {
                // Create a prim::variable layer and store attribute values
                auto attr_symbol = node->attributeNames().at(0);
                std::string attrMap_key = node->input()->type()->str() + "@" + node->s(attr_symbol);
                auto attr = module_attributes_[attrMap_key];

                if (!attr.isModule()) {
                    // Module attr is not our target, we only handle attr that has values
                    if (attr_layer_map.count(attrMap_key) == 0) {
                        // Create variable layer
                        auto variable_builder = this->layer_builders_.get("prim::Variable");
                        if (variable_builder != nullptr) {
                            auto variable_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimVariableLayer>(
                                variable_builder->buildLayer(node));
                            variable_layer->setName(convertLayerTypeToString(variable_layer->getType()) + "_" +
                                                    std::to_string(variable_layer->getID()));
                            variable_layer->setNType(attr.type()->repr_str());

                            // Store attr
                            addAttributeToLayer(attr, variable_layer);

                            // No input
                            // Add output
                            for (auto node_output : node->outputs()) {
                                // always create new
                                if (value_tensor_map_.find(node_output) == value_tensor_map_.end()) {
                                    int node_output_id = getUniqueTensorId(nn_model);
                                    auto shape_tensor = nn_model->getTSSTensors()[node_output_id];
                                    shape_tensor->setFeaturemapType(convertTorchScriptType(node_output->type()));
                                    shape_tensor->setParentLayer(variable_layer->getID());
                                    shape_tensor->setReprType(node_output->type()->repr_str());
                                    value_tensor_map_.emplace(node_output, node_output_id);
                                    variable_layer->addOutSTensorID(node_output_id);
                                } else {
                                    variable_layer->addOutSTensorID(value_tensor_map_[node_output]);
                                }
                                nn_model->addLayerRelationShips(value_tensor_map_[node_output], variable_layer);
                            }
                            attr_layer_map.emplace(attrMap_key, variable_layer);
                            graph->addLayer(variable_layer);
                        }
                    }

                    // Create GetAttr layer
                    auto get_attr_builder = this->layer_builders_.get("prim::GetAttr");
                    if (get_attr_builder != nullptr) {
                        auto get_attr_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimGetAttrLayer>(
                            get_attr_builder->buildLayer(node));
                        get_attr_layer->setName(convertLayerTypeToString(get_attr_layer->getType()) + "_" +
                                                std::to_string(get_attr_layer->getID()));

                        // Connect GetAttr and variable
                        auto variable_layer = attr_layer_map[attrMap_key];

                        // Add input
                        for (auto oid : variable_layer->getOutSTensorID()) {
                            get_attr_layer->addInSTensorID(oid);
                            nn_model->addLayerRelationShips(oid, get_attr_layer);
                        }
                        // Add output
                        for (auto node_output : node->outputs()) {
                            // always create new
                            if (value_tensor_map_.find(node_output) == value_tensor_map_.end()) {
                                int node_output_id = getUniqueTensorId(nn_model);
                                auto shape_tensor = nn_model->getTSSTensors()[node_output_id];
                                shape_tensor->setFeaturemapType(convertTorchScriptType(node_output->type()));
                                shape_tensor->setParentLayer(get_attr_layer->getID());
                                shape_tensor->setReprType(node_output->type()->repr_str());
                                value_tensor_map_.emplace(node_output, node_output_id);
                                get_attr_layer->addOutSTensorID(node_output_id);
                            } else {
                                get_attr_layer->addOutSTensorID(value_tensor_map_[node_output]);
                            }
                            nn_model->addLayerRelationShips(value_tensor_map_[node_output], get_attr_layer);
                        }
                        graph->addLayer(get_attr_layer);
                    }
                }
                break;
            }
            case c10::aten::arange: {
                auto input_num = (node->inputs()).size();
                std::shared_ptr<frontend::LayerBuilder> builder = nullptr;
                if (input_num == 5) {
                    builder = this->layer_builders_.get("aten::arange1");
                } else if (input_num == 6) {
                    builder = this->layer_builders_.get("aten::arange2");
                } else if (input_num == 7) {
                    builder = this->layer_builders_.get("aten::arange3");
                } else {
                    DLOG(FATAL) << "Unsupported input number of aten::arange.";
                }

                if (builder != nullptr) {
                    graph->addLayer(createLayer(builder, node, nn_model));
                } else {
                    DLOG(FATAL) << kind.toQualString() << " layer builder not found.";
                }
                break;
            }
            case c10::aten::lstm: {
                int lstm_type = 1;
                std::shared_ptr<frontend::LayerBuilder> builder = nullptr;
                if (node->inputs()[3]->node()->kind() == c10::prim::ListConstruct) {
                    lstm_type = 2;
                }
                if (lstm_type == 1) {
                    builder = this->layer_builders_.get("aten::lstm1");
                } else if (lstm_type == 2) {
                    builder = this->layer_builders_.get("aten::lstm2");
                } else {
                    DLOG(FATAL) << "Unsupported input number of aten::lstm.";
                }

                if (builder != nullptr) {
                    graph->addLayer(createLayer(builder, node, nn_model));
                } else {
                    DLOG(FATAL) << kind.toQualString() << " layer builder not found.";
                }
                break;
            }
            case c10::aten::to: {
                auto input_num = (node->inputs()).size();
                std::shared_ptr<frontend::LayerBuilder> builder = nullptr;
                std::string type = c10::typeKindToString(node->inputs()[1]->type()->kind());
                if (type == "DeviceObjType") {
                    builder = this->layer_builders_.get("aten::to1");
                } else if (type == "IntType") {
                    builder = this->layer_builders_.get("aten::to2");
                } else if (type == "TensorType") {
                    builder = this->layer_builders_.get("aten::to3");
                } else {
                    DLOG(FATAL) << "Unsupported input type of aten::to.";
                }

                if (builder != nullptr) {
                    graph->addLayer(createLayer(builder, node, nn_model));
                } else {
                    DLOG(FATAL) << kind.toQualString() << " layer builder not found. "<<type;
                }
                break;
            }
            default: {
                auto builder = this->layer_builders_.get(std::string(kind.toQualString()));
                if (builder != nullptr) {
                    auto layer = createLayer(builder, node, nn_model);
                    isInplaceNode(std::string(kind.toQualString()), layer);
                    graph->addLayer(layer);
                } else {
                    // Must be error, ensure each Op has its LayerBuilder
                    DLOG(FATAL) << kind.toQualString() << " layer builder not found.";
                }
                break;
            }
        }
    }

    bool isEmptyInputAndNodes = (method_block->inputs().begin() == method_block->inputs().end()) &&
                                (method_block->nodes().begin() == method_block->nodes().end());

    // set graph outputs
    for (auto output : method_block->outputs()) {
        if (value_tensor_map_.find(output) == value_tensor_map_.end()) {
            int output_id = getUniqueTensorId(nn_model);
            auto shape_tensor = nn_model->getTSSTensors()[output_id];
            shape_tensor->setFeaturemapType(convertTorchScriptType(output->type()));
            shape_tensor->setReprType(output->type()->repr_str());
            value_tensor_map_.emplace(output, output_id);
            graph->addGraphOutTensorID(output_id);
        } else {
            graph->addGraphOutTensorID(value_tensor_map_[output]);
        }
        // if layer's then and else net has no layers and inputs
        if (isEmptyInputAndNodes && !is_main_graph) {
            nn_model->addLayerRelationShips(value_tensor_map_[output], nullptr);
        }

        // Add output layer (only for main graph)
        if (is_main_graph) {
            auto output_layer_builder = this->layer_builders_.get("prim::Output");
            auto layer =
                std::dynamic_pointer_cast<nn_compiler::ir::PrimOutputLayer>(output_layer_builder->buildLayer(nullptr));
            layer->setName(convertLayerTypeToString(layer->getType()) + "_" + std::to_string(layer->getID()));
            layer->addInSTensorID(value_tensor_map_[output]);

            graph->addLayer(layer);
            nn_model->addLayerRelationShips(value_tensor_map_[output], layer);
        }
    }
    nn_model->appendGraph(graph);
    graphs_.emplace(name, graph);
    return;
}

void ModelBuilder::importModuleAttributes(std::shared_ptr<torch::jit::Module> torch_model)
{
    for (auto module : torch_model->named_modules()) {
        std::string module_type = module.value.type()->str();
        for (auto attr : module.value.named_attributes()) {
            std::string attr_name = module_type + "@" + attr.name;
            module_attributes_.emplace(attr_name, attr.value);
        }
    }
}

std::shared_ptr<ir::NNLayer> ModelBuilder::createLayer(std::shared_ptr<frontend::LayerBuilder> builder,
                                                       const torch::jit::Node* node,
                                                       std::unique_ptr<ir::NNModel>& nn_model)
{
    auto layer = builder->buildLayer(node);
    layer->setName(convertLayerTypeToString(layer->getType()) + "_" + std::to_string(layer->getID()));

    // Add input
    for (auto node_input : node->inputs()) {
        layer->addInSTensorID(value_tensor_map_[node_input]);
        nn_model->addLayerRelationShips(value_tensor_map_[node_input], layer);
    }

    // Add output
    for (auto node_output : node->outputs()) {
        if (value_tensor_map_.find(node_output) == value_tensor_map_.end()) {
            int node_output_id = getUniqueTensorId(nn_model);
            auto shape_tensor = nn_model->getTSSTensors()[node_output_id];
            shape_tensor->setFeaturemapType(convertTorchScriptType(node_output->type()));
            shape_tensor->setParentLayer(layer->getID());
            shape_tensor->setReprType(node_output->type()->repr_str());
            value_tensor_map_.emplace(node_output, node_output_id);
            layer->addOutSTensorID(node_output_id);
        } else {
            layer->addOutSTensorID(value_tensor_map_[node_output]);
        }
        nn_model->addLayerRelationShips(value_tensor_map_[node_output], layer);
    }
    return layer;
}

void ModelBuilder::torchToNNGraph(std::unique_ptr<ir::NNModel>& nn_model,
                                  std::shared_ptr<torch::jit::Module>& torch_model)
{
    // Get and map all module attributes
    importModuleAttributes(torch_model);

    // Get and map all methods
    block_counter_ = 0;

    for (auto m : torch_model->named_modules()) {
        torch::jit::Module module = m.value;
        for (auto method : module.get_methods()) {
            // Construct unique method name
            std::string method_name = module.type()->str() + "@" + method.name();
            importTorchScriptMethodBlock(nn_model, method_name, method.graph(), true);
        }
    }

    // Make first Graph to the main graph
    nn_model->reverseGraphs();
    return;
}

}  // namespace frontend
}  // namespace nn_compiler
