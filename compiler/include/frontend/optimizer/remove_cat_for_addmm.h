#pragma once

#include "compiler/include/common/pass.hpp"
#include "half.hpp"

typedef half_float::half float16;

namespace nn_compiler {

namespace frontend {


/** @Details:
 *  1. Change structure 1 to structure 2.
 *  2. Remove: a Listconstruct Op and a cat Op. Add: an addmm Op.
 *  3. Although a new addmm Op is inserted, the computation runtime is increased little.
 *     These computation are always indispensable.
 *  4. However, both of Listconstruct and cat Op require memory processing (copy or move) which will spend much time.
 *     Remove them can surely decrease runtime.
 *
 *          input1        input2
 *            |              |
 *             \            /
 *          prim::ListConstruct (to remove)
 *                   |         alpha (a prim::Constant, might be removed)
 *                   |           |
 *                    \         /
 *                   aten::cat (to remove)
 *                         |
 * bias (prim::Constant)   |    weights (prim::Constant)
 *                    \    |     /
 *                     aten::addmm
 *                         |
 *
 *                    structure 1
 *
 *
 *                        input1
 *                          |
 *  bias (prim::Constant)   |   splitted_weights_1
 *                    \     |     /
 *                     aten::addmm
 *                            |     input2    splitted_weights_2
 *                             \       |       /      
 *                              aten::addmm (new)
 *                                     |
 *
 *                structure 2
 *
 */

class RemoveCatForAddmm : public Pass {
 public:
    RemoveCatForAddmm();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~RemoveCatForAddmm() = default;

 private:
    std::vector<std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>>> layers_;

    // shape with height and width
    // TODO: remove hard code of tensor shapes
    std::vector<std::vector<int>> shape_of_inputs_ = {{1, 1024}, {1, 320}};

    std::vector<int> shape_of_matmul_weight_ = {1344, 512};

    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>>
    create_new_constants(std::shared_ptr<nn_compiler::ir::PrimConstantLayer> old_constant_layer);

    void reorganize_graph(std::unique_ptr<nn_compiler::ir::NNModel>& graph_model,
                          std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> new_constants);
};

}  // namespace frontend

} // namespace nn_compiler

