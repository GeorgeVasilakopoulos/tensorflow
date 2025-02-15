/* Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/grappler/optimizers/function_transformation.h"
#include <set>
#include <iostream>
#include <unordered_map>
#include "tensorflow/core/util/event.pb.h"
#include "tensorflow/core/util/events_writer.h"

#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {
namespace grappler {
namespace {

static constexpr const char* const kCallOp = "Call";
static constexpr const char* const kRetOp = "Return";
static constexpr const char* const kIdentityOp = "Identity";
static constexpr const char* const kIdentityNOp = "IdentityN";
static constexpr const char* const kMergeOp = "Merge";
static constexpr const char* const kGradientOp =
    FunctionLibraryDefinition::kGradientOp;
static constexpr const char* const kFuncAttrName =
    FunctionLibraryDefinition::kFuncAttr;
static constexpr const char* kNoInlineAttr = "_noinline";

bool AttrIsTrue(const FunctionDef& func, const string& attr) {
  return func.attr().count(attr) != 0 && func.attr().at(attr).b();
}

bool MarkedNoInline(const FunctionDef& func) {
  return AttrIsTrue(func, kNoInlineAttr);
}

// There are two ways of calling a Tensorflow function:
//
// 1. Direct function call: node.op() is the name of the function.
//
// 2. Indirect function call: the function name is passed through a node
//    attribute, and special Tensorflow kernels are responsible for calling the
//    function through the FunctionLibraryRuntime. Example: PartitionedCallOp.

// Check if func_node.op() matches the name in FunctionDef signature.
bool IsDirectFunctionCall(const FunctionDef& func, const NodeDef& func_node) {
  return func_node.op() == func.signature().name();
}

// Check if func_node has function attribute with a function name matching
// FunctionDef signature.
bool IsIndirectFunctionCall(const FunctionDef& func, const NodeDef& func_node) {
  auto* func_attr = AttrSlice(func_node).Find(kFuncAttrName);
  return func_attr != nullptr && func_attr->has_func() &&
         func_attr->func().name() == func.signature().name();
}

// Copy input/output argument type to the type_list. Return error if argument
// type is not explicitly defined, and not specified in function attributes.
Status CopyArgType(const OpDef::ArgDef& arg,
                   const AttrSlice& func_attr,
                   DataType* type) {
    if (arg.type() != DT_INVALID) {
      *type = arg.type();
    } else {
      const AttrValue* it = func_attr.Find(arg.type_attr());
      if (it == nullptr || it->type() == DT_INVALID) {
        return errors::InvalidArgument(
                "Invalid argument ", arg.name());
      }
      *type = it->type();
    }
    return OkStatus();
}

// Copy input/output argument type to the type_list. Return error if argument
// type is not explicitly defined, and not specified in function attributes.
Status CopyArgType(const OpDef::ArgDef& arg,
                   const AttrSlice& func_attr,
                   AttrValue::ListValue* type_list) {
    if (arg.type() != DT_INVALID) {
      type_list->add_type(arg.type());
    } else {
      const AttrValue* it = func_attr.Find(arg.type_attr());
      if (it == nullptr || it->type() == DT_INVALID) {
        return errors::InvalidArgument("Invalid argument ", arg.name());
      }
      type_list->add_type(it->type());
    }
    return OkStatus();
}


AttrSlice FunctionInstantiationAttributes(const FunctionDef& func,
                                          const NodeDef& func_node) {
  if (IsDirectFunctionCall(func, func_node)) {
    return AttrSlice(func_node);

  } else if (IsIndirectFunctionCall(func, func_node)) {
    auto* func_attr = AttrSlice(func_node).Find(kFuncAttrName);
    return AttrSlice(&func_attr->func().attr());

  } else {
    LOG(WARNING) << "Can't resolve function instantiation attributes: "
                 << SummarizeNodeDef(func_node);
    return AttrSlice();
  }
}

struct FuncInfo { 
  DataTypeVector arg_types;
  DataTypeVector ret_types;
  std::vector<NodeDef*> args;
  std::vector<string> rets;
};

struct FuncGradInfo {
  FuncInfo f;
  FuncInfo g;
};

// same with commit a9a3b98 (possibly)
class FunctionInliningContext {
  public:
    explicit FunctionInliningContext(const GrapplerItem& item)
            : function_library_(FunctionLibraryDefinition(OpRegistry::Global(),
                                                    item.graph.library())) {
      InitializeInlinedFunctions(item);
      InitializeFetchNodes(item);
    }


    const FunctionLibraryDefinition& FunctionLibrary() const { 
      return function_library_; 
    }

    Status AddFunctionDef(const FunctionDef& fdef) {
      TF_RETURN_IF_ERROR(function_library_.AddFunctionDef(fdef));
      inlined_functions_[fdef.signature().name()] = function_library_.Find(fdef.signature().name());
      return OkStatus();
    }


    bool HasInlinedFunctions() const { return !inlined_functions_.empty(); }

    bool IsInlinedFunction(const string& name) const {
      return inlined_functions_.count(name) > 0;
    }

    // Find inlining candidate by name. Return nullptr if not found.
    const FunctionDef* FindInlinedFunction(const string& name) const {
      return gtl::FindWithDefault(inlined_functions_, name, nullptr);
    }

    bool IsFetchNode(const string& node_name) const {
      return fetch_nodes_.find(node_name) != fetch_nodes_.end();
    }

    const FunctionDef* FindInlinedFunctionAndGradient(const string& name) const {
      string grad_name = strings::StrCat(name, "Grad");
      return FindInlinedFunction(grad_name);
    }

  private:
    void InitializeInlinedFunctions(const GrapplerItem& item) {
      for (const FunctionDef& func : item.graph.library().function()) {

        printf("Func name %s\n",func.signature().name().c_str());

        bool marked_noinline = MarkedNoInline(func);
        // Don't inline functions marked as noinline
        if (marked_noinline) {
           continue;
        }
        // Don't touch anything marked XLA to prevent XLA failures further down
        // the road.
        if (func.attr().count("_XlaCompile") > 0 &&
            func.attr().at("_XlaCompile").b()) {
          continue;
        }
        // Can't create IdentityN nodes with no input or output: skip these
        // functions for now.
        if (func.signature().input_arg_size() == 0 ||
            func.signature().output_arg_size() == 0) {
          continue;
        }
        inlined_functions_[func.signature().name()] = &func;
      }
    }

    void InitializeFetchNodes(const GrapplerItem& item) {
      for (const string& fetch : item.fetch) {
        fetch_tensors_.insert(fetch);
        fetch_nodes_.insert(NodeName(fetch));
      }
    }

    FunctionLibraryDefinition function_library_;
    std::unordered_map<string, const FunctionDef*> inlined_functions_;	
    gtl::FlatSet<string> fetch_tensors_;  // format: node_name:port
    gtl::FlatSet<string> fetch_nodes_;    // format: node_name

    TF_DISALLOW_COPY_AND_ASSIGN(FunctionInliningContext);
};

struct CallInfo {
  int call_id;
  string call_frame;
  NodeDef* fcall = nullptr;
  NodeDef* gcall = nullptr;
  bool hasGradient() const { return (gcall != nullptr); }
};

struct TransformationResult {
  int call_id;
  string call_frame;
  NodeDef* transformed_node;
  std::vector<NodeDef*> call_nodes;
  std::vector<NodeDef*> ret_nodes;
};

class CallRewriter {

  public:
    explicit CallRewriter(const GrapplerItem& item_, GraphDef* graph_, FunctionInliningContext& ctx_)
        : graph(graph_), ctx(ctx_), item(item_) { }

    ~CallRewriter() {
        Flush();
    }

    Status CollectCalls(std::vector<CallInfo>& calls);

    Status TransformCall(const CallInfo& call_info);

    // Inlines a function to item.graph and if already inlined provide func_info
    Status FindCompatibleOrInlineFunction(const CallInfo& call,
                                          GraphDef* optimized_graph,
                                          FuncGradInfo& func_info);

    void Flush();

    inline int GetCallId(const NodeDef& node) { int call_id = id; id++; return call_id; }

  private:
    Status TransformNode(const CallInfo& info, 
            NodeDef* call, const FuncInfo& f, 
            std::vector<NodeDef*>& call_nodes,
            std::vector<NodeDef*>& ret_nodes,
            bool is_gradient_node);

    void ReplaceOutput(const string& old_output, const string& new_output) {
        // maybe some more checks
        output_map_[old_output] = new_output;
    }

    void MarkCallTransformed(const CallInfo& call_info) {
      CHECK_NOTNULL(call_info.fcall);
      MarkNodeDelete(call_info.fcall);
      
      if (call_info.gcall != nullptr) {
        MarkNodeDelete(call_info.gcall);
      }
    }

    void MarkTransformed(TransformationResult& result) {
      NodeDef* n = result.transformed_node;
      CHECK_NOTNULL(n);
      transformed_calls_[result.transformed_node->name()] = result;
      n->clear_input();
      n->set_op("NoOp");
      n->set_name(AddPrefixToNodeName(n->name(), "$MarkToDelete$"));
      nodes_to_delete.insert(n->name());
    }

    void MarkNodeDelete(NodeDef* n) {
      n->clear_input();
      n->set_op("NoOp");
      n->set_name(AddPrefixToNodeName(n->name(), "$MarkToDelete$"));
      nodes_to_delete.insert(n->name());
    }

    GraphDef* graph;
    FunctionInliningContext& ctx;
    const GrapplerItem& item;
    std::unordered_map<string, FuncGradInfo> transformed_functions_;
    std::unordered_map<string, string> output_map_;
    std::unordered_map<string, TransformationResult> transformed_calls_;
    std::set<string> nodes_to_delete;
    int id = 0;

    TF_DISALLOW_COPY_AND_ASSIGN(CallRewriter);
};

Status AddCallOp(const CallInfo& call_info,
               const DataType& type,
               const string& input,
               const string& prefix,
               int arg_id, NodeDef* call, bool is_gradient_call = false) {
    string call_name = strings::StrCat("Call", "_", arg_id);
    call->set_op(kCallOp);
    call->set_name(AddPrefixToNodeName(call_name, prefix));
    //call->set_device(node.device());
    call->add_input(input);

    auto& attr = *call->mutable_attr();
    attr["T"].set_type(type);
    attr["frame_name"].set_s(call_info.call_frame);
    attr["call_id"].set_i(call_info.call_id);
    attr["arg_id"].set_i(arg_id);
    attr["is_constant"].set_b(false);
    attr["is_gradient"].set_b(is_gradient_call);

    return OkStatus();
}

Status AddRetOp(const CallInfo& call_info,
              const DataType& type,
              const string& input,
              const string& prefix,
              int arg_id, NodeDef* ret, bool is_gradient_return = false) {
    string ret_name = strings::StrCat("Ret", "_", arg_id);
    ret->set_op(kRetOp);
    ret->set_name(AddPrefixToNodeName(ret_name, prefix));
    ret->add_input(input);

    auto& attr = *ret->mutable_attr();
    attr["T"].set_type(type);
    attr["frame_name"].set_s(call_info.call_frame);
    attr["call_id"].set_i(call_info.call_id);
    attr["arg_id"].set_i(arg_id);
    attr["is_gradient"].set_b(is_gradient_return);

    return OkStatus();
}

Status ConnectInput(NodeDef* from, NodeDef* to) {
    int to_input = to->input_size();
    if (to_input == 1) {
        // it is Identity and we convert it to Merge.
        CHECK(IsIdentity(*to));
        to->set_op(kMergeOp);
    }
    to->add_input(from->name());
    if (to->input_size() > 1) {
        (*to->mutable_attr())["N"].set_i(to->input_size());
    }
    return OkStatus();
}

Status InlineFunction(const FunctionDef& func_def,
                      const AttrSlice& func_instantiation_attr,
                      const FunctionInliningContext& ctx,
                      const string& device,
                      GraphDef* graph, FuncGradInfo& func_info) {
    GrapplerFunctionItem item;
    const int graph_version = graph->versions().producer();
    TF_RETURN_IF_ERROR(MakeGrapplerFunctionItem(func_def, func_instantiation_attr, ctx.FunctionLibrary(), graph_version, &item));

    string prefix = func_def.signature().name();
    int arg_size = func_def.signature().input_arg_size();
    // create an inverse map of arg to provide name -> argument number
    std::unordered_map<string, int> input_nodes;
    for (int i = 0; i < arg_size; ++i) {
        const OpDef::ArgDef& input_arg = func_def.signature().input_arg(i);
        input_nodes[input_arg.name()] = i;
    }
    func_info.f.args.resize(arg_size);
    func_info.f.arg_types.resize(arg_size);
    for (int i = 0; i < arg_size; ++i) {
        const OpDef::ArgDef& input_arg = func_def.signature().input_arg(i);
        NodeDef* merge = graph->add_node();
        merge->set_name(AddPrefixToNodeName(strings::StrCat("Input", "_", i), prefix));
        merge->set_op(kIdentityOp);
        merge->set_device(device);
        
        DataType type;
        TF_RETURN_IF_ERROR(CopyArgType(input_arg, func_instantiation_attr, &type));
        auto& attr = *merge->mutable_attr();
        attr["T"].set_type(type);

        func_info.f.args[i] = merge;
        func_info.f.arg_types[i] = type;
    }

    // prefix each node in function graph and place it to the global graph.
    // the inputs of each node need to be renamed as well to reflect the change.
    for (NodeDef& func_body_node : *item.mutable_function_body().mutable_node()) {
        const string& curr_name = func_body_node.name();
        // If the func body node is func's input argument
        auto input_it = input_nodes.find(curr_name);

        if (input_it != input_nodes.end()) {
            CHECK_EQ(0, func_body_node.input_size());
            // If the func body node is func's input argument
            // Turn input placeholders into identity nodes
            func_body_node.set_op(kIdentityOp);
            // Connect merge with input arg
            int idx = input_nodes[curr_name];
            func_body_node.add_input(func_info.f.args[idx]->name());
        } else {
            // Else if not an input_arg_node
            // Update the input names if any.
            for (string& input : *func_body_node.mutable_input()) {
                input = AddPrefixToNodeName(input, prefix);
            }
            // If this is a return node, change the op to KIdentityOp
            if(IsRetval(func_body_node)){
                func_body_node.set_op(kIdentityOp);
            }

            // If the node has no input, make hook it up to the Merge nodes to ensure
            // it runs in the same frame as the other nodes of the function body.
            if (func_body_node.input_size() == 0) {
                for (auto& func_input_node : func_info.f.args) {
                 *func_body_node.add_input() = AsControlDependency(func_input_node->name());
                }
            }
        }

        // Add the node name as a prefix to avoid collisions after inlining
        func_body_node.set_name(AddPrefixToNodeName(curr_name, prefix));

        // Make sure the node is placed
        if (func_body_node.device().empty())
          func_body_node.set_device(device);

        // Move the node to the main graph
        graph->add_node()->Swap(&func_body_node);
    }

    func_info.f.rets.clear();
    func_info.f.rets.resize(item.fetch.size());
    func_info.f.ret_types.resize(item.fetch.size());

    std::vector<string> fetch = item.fetch;
    for (unsigned int i = 0; i < fetch.size(); i++) {
        const OutputArgInstantiation& output_arg = item.output(i);
        func_info.f.rets[i] = AddPrefixToNodeName(output_arg.node_name, prefix);
        func_info.f.ret_types[i] = output_arg.data_type;
    }

    return OkStatus();
}

Status InlineFunctionAndGradient(const FunctionDef& fdef,
                      const AttrSlice& func_instantiation_attr,
                      FunctionInliningContext& ctx,
                      const string& device,
                      GraphDef* graph, 
                      FuncGradInfo& func_info) {
    // Get func_def's gradient graph
    
    const FunctionDef* fgdef = ctx.FindInlinedFunctionAndGradient(fdef.signature().name());
    if (fgdef == nullptr) {
        return errors::InvalidArgument(
                "Invalid argument, gradient of function ", fdef.signature().name(), "can not be found",
                "or not marked to be inlined");
    }
    
    


    GrapplerFunctionItem item;
    const int graph_version = graph->versions().producer();
    TF_RETURN_IF_ERROR(MakeGrapplerFunctionItem(*fgdef, func_instantiation_attr, ctx.FunctionLibrary(), graph_version, &item));

    string prefix = fdef.signature().name();
    size_t farg_size = fdef.signature().input_arg_size();
    size_t fret_size = fdef.signature().output_arg_size();
    size_t garg_size = fgdef->signature().input_arg_size();// - farg_size;
    size_t gret_size = fgdef->signature().output_arg_size();// - fret_size;

    CHECK_EQ(farg_size, gret_size - fret_size);
    CHECK_EQ(garg_size, fret_size + farg_size);

    func_info.f.arg_types.resize(farg_size);
    func_info.g.arg_types.resize(garg_size);
    func_info.g.ret_types.resize(gret_size);
    for (int i = 0; i < farg_size; i++) {
      const OpDef::ArgDef& input_arg = fdef.signature().input_arg(i);
      func_info.f.arg_types[i] = input_arg.type();
      func_info.g.arg_types[i] = input_arg.type();
    }

    func_info.f.ret_types.resize(fret_size);
    for (int i = 0; i < gret_size; i++) {
      // const OutputArgInstantiation& output_arg = item.output(i);
      if(i < fret_size){
        func_info.f.ret_types[i] = item.output(i).data_type;
        func_info.g.arg_types[farg_size + i] = item.output(i).data_type;
      }
      func_info.g.ret_types[i] = item.output(i).data_type;
    }

    // create an inverse map of arg to provide name -> argument number
    std::unordered_map<string, int> input_map;
    std::vector<string> input_names;
    input_names.resize(farg_size);
    for (int i = 0; i < garg_size; ++i) {
        input_map[item.input(i).node_name] = i;
        if (i < farg_size) {
          input_names[i] = item.input(i).node_name;
        }
    }
    func_info.f.args.resize(farg_size);
    func_info.f.rets.resize(fret_size);
    func_info.g.args.resize(garg_size);
    func_info.g.rets.resize(gret_size);

    // prefix each node in function graph and place it to the global graph.
    // the inputs of each node need to be renamed as well to reflect the change.
    for (NodeDef& n : *item.mutable_function_body().mutable_node()) {
        // If the func body node is func's input argument
        auto input_it = input_map.find(n.name());
        bool is_input = input_it != input_map.end();

        if (is_input) {
          CHECK_EQ(0, n.input_size());
          n.set_op(kIdentityOp);
        }

        // Add the node name as a prefix to avoid collisions after inlining
        n.set_name(AddPrefixToNodeName(n.name(), prefix));
        // Update the input names if any.
        for (string& input : *n.mutable_input()) {
            input = AddPrefixToNodeName(input, prefix);
        }

        // Make sure the node is placed
        if (n.device().empty())
          n.set_device(device);

        // if (n.op() == kGradientOp) {
        //   auto& attr = *n.mutable_attr();
        //   std::string& name = *attr.at("f").mutable_func()->mutable_name();
        //   name = AddPrefixToNodeName(name, prefix);
        // }
        if(IsRetval(n)){
          n.set_op(kIdentityOp);
        }

        // If the node has no input, make hook it up to the Merge nodes to ensure
        // it runs in the same frame as the other nodes of the function body.
        if (!is_input && n.input_size() == 0) {
          // CHECK: constants from both in function and gradient are connected 
          // with the inputs of the function only.
          for (const string& arg : input_names) {
            *n.add_input() = AsControlDependency(AddPrefixToNodeName(arg, prefix));
          }
        }

        // Move the node to the main graph
        NodeDef* nn = graph->add_node();
        nn->Swap(&n);
        
        if (is_input) {
          int i = input_it->second;
          if (i < farg_size) {
            func_info.f.args[i] = nn;
            func_info.g.args[i] = func_info.f.args[i];
          } else { 
            func_info.g.args[i] = nn;
          }
        }
    }

    CHECK_EQ(gret_size, item.fetch.size());

    for (unsigned int i = 0; i < gret_size; i++) {
        string output_port = AddPrefixToNodeName(item.output(i).node_name, prefix);
        if (i < fret_size) {
          func_info.f.rets[i] = output_port;
        }
        func_info.g.rets[i] = output_port;
    }

    return OkStatus();
}

Status CallRewriter::CollectCalls(std::vector<CallInfo>& calls) {

    std::unordered_map<string,CallInfo> call_map;
    std::vector<NodeDef*> gradients;

    // identify and collect calls in the graph
    for (NodeDef& node : *graph->mutable_node()) {
        if (node.op() == kGradientOp) {
            gradients.push_back(&node);
        } else {
            const FunctionDef* func_def = ctx.FindInlinedFunction(node.op());
            if (func_def != nullptr) {
                CallInfo& call = call_map[node.op()];
                call.call_id = GetCallId(node);
                call.call_frame = node.op();
                call.fcall  = &node;
            }
        }
    }
    for (NodeDef* gcall : gradients) {
        if (gcall->attr().count("f") > 0) {
          printf("Debug string: %s \n\n", gcall->attr().at("f").DebugString().c_str());
          const string& n = gcall->attr().at("f").func().name();
        
          auto fcall_it = call_map.find(n);
          if (fcall_it == call_map.end()) {
              // return errors::InvalidArgument("Cannot find forward node for gradient ",
              //         gcall->name());
            continue;
          }
          CallInfo& call = fcall_it->second;
          call.gcall = gcall;
        }
    }

    for (const auto& it : call_map) {
        calls.push_back(it.second);
    }
    return OkStatus();
}

Status CallRewriter::TransformNode(const CallInfo& info, 
        NodeDef* call, 
        const FuncInfo& f, 
        std::vector<NodeDef*>& call_nodes,
        std::vector<NodeDef*>& ret_nodes, bool is_gradient_node = false) {
  CHECK_EQ(call->input_size(), f.args.size());

  unsigned int next_return_node = is_gradient_node ? ret_nodes.size() : 0;  

  call_nodes.resize(f.args.size());
  for (unsigned int i = 0; i < f.args.size(); i++) {
      /* check if call node is already in place, if so, validate and skip */
      if (call_nodes[i] != nullptr) {
        // TODO: validate call_id
        // TODO: validate input
        //CHECK_EQ(call_nodes[i]->input(0), call->input(i));
      } else {
        call_nodes[i] = graph->add_node();
        TF_CHECK_OK(AddCallOp(info,
                f.arg_types[i],
                call->input(i),
                call->name(),
                i,
                call_nodes[i],
                is_gradient_node));

        call_nodes[i]->set_device(call->device());

        // connect the input of the inlined function to feed from call.
        TF_RETURN_IF_ERROR(ConnectInput(call_nodes[i], f.args[i]));
      }
  }

  // check for control edges in call
  gtl::FlatSet<string> control_inputs;
  for (const string& input : call->input()) {
    if (IsControlInput(input)) {
      control_inputs.insert(NodeName(input));
    }
  }

  for (NodeDef* call_node : call_nodes) {
    for (const string& control_input : control_inputs)
    *(call_node->add_input()) = AsControlDependency(control_input);
  }

  ret_nodes.resize(f.rets.size());
  for (unsigned int i = 0; i < f.rets.size(); i++) {
      if (ret_nodes[i] != nullptr) {
        // TODO: validate call_id
        // CHECK_EQ(ret_nodes[i]->input(0), f.rets[i]);
      } else {
        ret_nodes[i] = graph->add_node();
        TF_CHECK_OK(AddRetOp(info,
                f.ret_types[i],
                f.rets[i],
                call->name(),
                i,
                ret_nodes[i],
                is_gradient_node));
        ret_nodes[i]->set_device(call->device());
      }
  }

  if (ctx.IsFetchNode(call->name())) {
      // create an IdentityN with the same name of the initial function call
      // so as to preserve the naming of the outputs.
      // we re-use the initial node and we change (a) the op to IdentityN and
      // (b) the inputs to point to the outputs of the ret_nodes
      // The other information such as types, device placement etc remain the same.
      // The IdentityN node will sync the outputs and therefore may result to performance degradation.
      NodeDef* out = graph->add_node();
      out->set_op(kIdentityNOp);
      out->set_name(call->name());
      out->set_device(call->device());
      AttrValue::ListValue* type_list = (*out->mutable_attr())["T"].mutable_list();
      for (const DataType& type : f.ret_types) {
        type_list->add_type(type);
      }
      for (unsigned int i = 0; i < f.rets.size(); i++) {
          *out->add_input() = ret_nodes[i]->name();
      }
  } else {
      for (unsigned int i = next_return_node; i < f.rets.size(); i++) {
          ReplaceOutput(strings::StrCat(call->name(), ":", i - next_return_node), ret_nodes[i]->name());
          if(i == next_return_node)ReplaceOutput(call->name(), ret_nodes[i]->name());
      }
  }

  // for each call create a control dependency to each return
  // to facilitate dead propagation semantics
  for (NodeDef* ret : ret_nodes) {
      for (NodeDef* call : call_nodes){
        if(ret->attr().at("is_gradient").b() != call->attr().at("is_gradient").b()) continue;
        printf("Adding control edge from %s to %s\n",call->name().c_str(),ret->name().c_str());
        // TODO: Check if there is already a control dependency.
        *(ret->add_input()) = AsControlDependency(call->name());
        }
  }

  return OkStatus();
}

Status CallRewriter::TransformCall(const CallInfo& call_info) {
    FuncGradInfo func_info;
    TransformationResult result;

    // inlines the body of a function and provides a struct with func_info
    TF_RETURN_IF_ERROR(FindCompatibleOrInlineFunction(call_info, graph, func_info));

    result.call_id = call_info.call_id;
    result.call_frame = call_info.call_frame;
    result.transformed_node = call_info.fcall;

    TF_RETURN_IF_ERROR(TransformNode(call_info, call_info.fcall, func_info.f, result.call_nodes, result.ret_nodes,false));
    MarkTransformed(result);

    if (call_info.hasGradient()) {
      TransformationResult grad_result;
      grad_result.call_id = call_info.call_id;
      grad_result.call_frame = call_info.call_frame;
      grad_result.transformed_node = call_info.gcall;
      grad_result.call_nodes = result.call_nodes;
      grad_result.ret_nodes = result.ret_nodes;
      // keep all the inputs of the function
      TF_RETURN_IF_ERROR(TransformNode(call_info, call_info.gcall, func_info.g, grad_result.call_nodes, grad_result.ret_nodes,true));
      MarkTransformed(grad_result);
    }
    MarkCallTransformed(call_info);
    return OkStatus();
}

Status CallRewriter::FindCompatibleOrInlineFunction(
            const CallInfo& call,
            GraphDef* graph,
            FuncGradInfo& func_info) {
    CHECK_NOTNULL(call.fcall);
    const string& func_name = call.fcall->op();
    string device = call.fcall->device();
    const auto& it = transformed_functions_.find(func_name);
    // maybe it is not wise to discard call attributes
    // possible type specialization?
    if (it != transformed_functions_.end()) {
        func_info = it->second;
        return OkStatus();
    }
    const FunctionDef* func_def = ctx.FindInlinedFunction(func_name);
    if (func_def == nullptr) {
        return errors::InvalidArgument(
                        "Invalid argument, function ", func_name, "can not be found",
                        "or not marked to be inlined");
    }

    const AttrSlice func_instantiation_attr =
        FunctionInstantiationAttributes(*func_def, *call.fcall);

    if (call.hasGradient()) {
      TF_RETURN_IF_ERROR(
              InlineFunctionAndGradient(*func_def, func_instantiation_attr, ctx, device, graph, func_info));
    } else { 
      TF_RETURN_IF_ERROR(
              InlineFunction(*func_def, func_instantiation_attr, ctx, device, graph, func_info));
    }
    transformed_functions_[func_name] = func_info;
    printf("Store inlined function %s\n", func_name.c_str());
    return OkStatus();
}

void CallRewriter::Flush() {

    if (!transformed_calls_.empty()) {
        // garbage collect the transformed call nodes
        int last = graph->node_size() - 1;
        for (int i = graph->node_size() - 1; i >= 0; --i) {
            const NodeDef& node = graph->node(i);
            if (nodes_to_delete.find(node.name()) != nodes_to_delete.end()) {
                graph->mutable_node()->SwapElements(i,last);
                last--;
            }
        }
        graph->mutable_node()->DeleteSubrange(last + 1,
                                              graph->node_size() - last - 1);
    }


    // for(auto& p : output_map_){
    //     printf("%s -> %s\n",p.first.c_str(),p.second.c_str());

    // }

    if (!output_map_.empty()) {
      for (NodeDef& node : *graph->mutable_node()) {
        std::vector<TransformationResult> control_nodes;
        int last = node.input_size() - 1;

        for (int i = node.input_size() - 1; i >= 0; --i) {
          string& in = *node.mutable_input(i);
          auto it = output_map_.find(in);
          if (it != output_map_.end()) {
            in = it->second;
          }
          if (IsControlInput(in)) {
            auto it = transformed_calls_.find(NodeName(in));
            if (it != transformed_calls_.end()) {
              node.mutable_input()->SwapElements(i, last);
              control_nodes.push_back(it->second);
              last--;
            }
          }
          node.mutable_input()->DeleteSubrange(last + 1,
                                              node.input_size() - last - 1);
          for (TransformationResult& result : control_nodes) {
            for (NodeDef* ret_node : result.ret_nodes) {
              *node.add_input() = AsControlDependency(ret_node->name());
            }
          }
        }
      }
    }
    transformed_calls_.clear();
    nodes_to_delete.clear();
    output_map_.clear();
}

}  // namespace

Status FunctionTransformation::Optimize(Cluster* cluster, const GrapplerItem& item,
                                        GraphDef* output) {
    FunctionInliningContext ctx(item);
    CallRewriter call_rewriter(item, output, ctx);

    *output = item.graph;

    printf("Before optimizer: %s\n\n",SummarizeGraphDef(*output).c_str());
    if (!ctx.HasInlinedFunctions()) {
        return OkStatus();
    }

    std::vector<CallInfo> calls;
    while (1) {
        TF_RETURN_IF_ERROR(call_rewriter.CollectCalls(calls));
        if (calls.empty()) {
            break;
        }
        for (const CallInfo& call : calls) {
            Status s = call_rewriter.TransformCall(call);
            if (!s.ok()) {
              printf("Error: %s\n", tsl::NullTerminatedMessage(s));
              return s;
            }
            printf("After transforming call %s:\n %s\n", call.fcall->name().c_str(), SummarizeGraphDef(*output).c_str());
        }
        calls.clear();
        call_rewriter.Flush();
    }
    call_rewriter.Flush();
    

    printf("After finalizing:\n %s\n", SummarizeGraphDef(*output).c_str());

    // for (NodeDef& node : *output->mutable_node()){
      
    //   if(node.op() != kGradientOp)continue;
    //   NameAttrList func;
    //   TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(node), kFuncAttrName, &func));
    //    gradient::Creator creator;
    //   TF_RETURN_IF_ERROR(gradient::GetOpGradientCreator(func.name(), &creator));
    //   if (creator == nullptr) {
    //     return absl::InvalidArgumentError(
    //         absl::StrCat("No gradient is defined for ", func.name()));
    //   }
    //   FunctionDef grad_fdef;

    //   std::unique_ptr<FunctionBody>* fbody;
    //   TF_RETURN_IF_ERROR(creator(AttrSlice(&func.attr()), &grad_fdef));
    //   TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
    //       grad_fdef, AttrSlice(&func.attr()), &ctx.FunctionLibrary(), fbody));

    //   printf("Gradient of of %s:\n%s\n\n",func.name().c_str(),SummarizeGraphDef((*fbody)->graph->ToGraphDefDebug()).c_str());


    // }
    
    
    
    
    
    
    *output->mutable_versions() = item.graph.versions();

    // Function Library should be pruned of unreachable function definitions
    // cf. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/function_optimizer.cc#L428
    // however in this version there is a check in meta_optimizer that guarantees
    // that function library remains of the same length
    // cf. https://github.com/acharal/tensorflow/blob/r1.4_recursion/tensorflow/core/grappler/optimizers/meta_optimizer.cc#L132
    *output->mutable_library() = item.graph.library();



    /******************************************************************************************************/
    // Dumps optimized graph in a not so readable form
    // const GraphDef* tmp = optimized_graph;
    // printf("Summarize Optimized Graph\n %s\n", SummarizeGraphDef(*tmp).c_str());
    // Write an event, so that we can visualize this optimized graph in tensorboard
    EventsWriter writer("TRANSFORMATION");
    Event event;
    event.set_wall_time(1234);
    event.set_step(34);
    const size_t proto_size = output->ByteSizeLong();
    void* buf = port::Malloc(proto_size);
    if (buf == nullptr) {
    return errors::ResourceExhausted(
              "Failed to allocate memory to serialize message of type '" ,
              output->GetTypeName(), "' and size ", proto_size);
    }
    output->SerializeToArray(buf, proto_size);
    const void* bf = buf;
    event.set_graph_def(bf, proto_size);
    writer.WriteEvent(event);
    /******************************************************************************************************/

    return OkStatus();
}

}  // end namespace grappler
}  // end namespace tensorflow