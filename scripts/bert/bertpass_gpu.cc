/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2019 by Contributors
 * \file subgraph_lib.cc
 * \brief subgraph operator implementation library file
 */

#include <math.h>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <functional>
#include "mxnet/lib_api.h"

#if MX_LIBRARY_VERSION <= 7
class Node;
struct NodeEntry {
  Node* node;
  int entry;
};

class Node {
 public:
  std::string op,name;
  std::vector<NodeEntry> inputs;
  std::vector<NodeEntry> outputs;
  std::unordered_map<std::string, std::string> attrs;
};

class Graph {
 public:
  Graph() {}
  static Graph* fromString(const std::string& json) {
    JsonParser parser;
    JsonVal val = parser.parse_to_json(json);
    return fromJson(val);
  }
  ~Graph() {
    for(int i=0; i<nodes.size(); i++)
      delete nodes[i];
  }
  static Graph* fromJson(JsonVal val) {
    // get nodes list
    JsonVal nodes = val.map[JsonVal("nodes")];
    Graph *g = new Graph();

    std::map<int, Node*> nodeMap;
    // loop over nodes
    for(int i=0; i<nodes.list.size(); i++) {
      Node* n = new Node();
      g->nodes.push_back(n);
      JsonVal node = nodes.list[i];

      // set the op info
      n->op = node.map[JsonVal("op")].str;
      n->name = node.map[JsonVal("name")].str;

      // if op is null its an input to the graph
      if(n->op.compare("null") == 0)
        g->inputs.push_back(n);

      // set attrs
      JsonVal attributes = node.map[JsonVal("attrs")];
      for(auto& kv : attributes.map) {
        n->attrs[kv.first.str] = kv.second.str;
      }

      // set node inputs
      JsonVal node_inputs = node.map[JsonVal("inputs")];
      n->inputs.resize(node_inputs.list.size());
      for(int j=0; j<node_inputs.list.size(); j++) {
        JsonVal input = node_inputs.list[j];
        NodeEntry& entry = n->inputs[j];
        //get pointer to other node
        entry.node = nodeMap[input.list[0].num];
        //get the other node's output index
        entry.entry = input.list[1].num;
        //set other nodes output as connected to this node
        entry.node->outputs.push_back({n,j});
      }
      nodeMap[i] = n;
    }

    JsonVal& heads = val.map[JsonVal("heads")];
    g->outputs.resize(heads.list.size());
    for(int i=0; i<heads.list.size(); i++) {
      JsonVal head = heads.list[i];
      g->outputs[i].node = nodeMap[head.list[0].num];
      g->outputs[i].entry = head.list[1].num;
    }

    JsonParser parser;
    for(auto& kv : val.map) {
      if(kv.first.str.compare("nodes") != 0 &&
        kv.first.str.compare("heads") != 0 &&
        kv.first.str.compare("node_row_ptr") != 0 &&
        kv.first.str.compare("arg_nodes") != 0) {
        g->attrs[kv.first.str] = kv.second;
      }
    }
    return g;
  }
  JsonVal toJson() {
    JsonVal val(MAP);
    for(auto& kv : attrs) {
      val.map[JsonVal(kv.first)] = kv.second;
    }

    std::map<Node*, int> nodeMap;
    std::vector<Node*> sorted = topological_sort();
    for(int i=sorted.size()-1; i>=0; i--) {
      nodeMap[sorted[i]] = sorted.size()-1-i;
    }

    val.map[JsonVal("node_row_ptr")] = JsonVal(LIST);
    JsonVal& node_row_ptr = val.map[JsonVal("node_row_ptr")];
    for(int i=0; i<nodes.size(); i++)
      node_row_ptr.list.push_back(JsonVal(i));

    val.map[JsonVal("arg_nodes")] = JsonVal(LIST);
    JsonVal& arg_nodes = val.map[JsonVal("arg_nodes")];
    for(int i=0; i<inputs.size(); i++)
      arg_nodes.list.push_back(JsonVal(nodeMap[inputs[i]]));

    val.map[JsonVal("heads")] = JsonVal(LIST);
    JsonVal& heads = val.map[JsonVal("heads")];
    for(int i=0; i<outputs.size(); i++) {
      heads.list.push_back(JsonVal(LIST));
      JsonVal& out = heads.list[i];
      out.list.push_back(JsonVal(nodeMap[outputs[i].node]));
      out.list.push_back(JsonVal(outputs[i].entry));
      out.list.push_back(JsonVal(0));
    }

    val.map[JsonVal("nodes")] = JsonVal(LIST);
    JsonVal& nodes_ = val.map[JsonVal("nodes")];
    for(int i=sorted.size()-1; i>=0; i--) {
      nodes_.list.push_back(JsonVal(MAP));
      Node* n = sorted[i];
      JsonVal& n_ = nodes_.list[nodes_.list.size()-1];

      n_.map[JsonVal("op")] = JsonVal(n->op);
      n_.map[JsonVal("name")] = JsonVal(n->name);
      n_.map[JsonVal("inputs")] = JsonVal(LIST);

      JsonVal& inputs_ = n_.map[JsonVal("inputs")];
      for(int j=0; j<n->inputs.size(); j++) {
        inputs_.list.push_back(JsonVal(LIST));
        NodeEntry& entry = n->inputs[j];
        JsonVal& in = inputs_.list[j];
        in.list.push_back(JsonVal(nodeMap[entry.node]));
        in.list.push_back(JsonVal(entry.entry));
        in.list.push_back(JsonVal(0));
      }

      n_.map[JsonVal("attrs")] = JsonVal(MAP);
      JsonVal& attrs_ = n_.map[JsonVal("attrs")];
      for(auto& kv : n->attrs) {
        attrs_.map[JsonVal(kv.first)] = JsonVal(kv.second);
      }
    }
    return val;
  }
  std::string toString() {
    JsonParser parser;
    return parser.dump(toJson());
  }

  void _dfs_util(Node* n, std::unordered_set<Node*>* to_visit,
                 std::function<void(Node*)> handler) {
    to_visit->erase(n);
    for(NodeEntry& e : n->outputs) {
      Node* o = e.node;
      if(to_visit->count(o) != 0) {
        _dfs_util(o,to_visit,handler);
      }
    }
    handler(n);
  }

  void DFS(std::function<void(Node*)> handler) {
    std::unordered_set<Node*> to_visit;
    //put all nodes in set to visit
    for(auto& n : nodes)
      to_visit.insert(n);
    //visit all inputs first
    for(auto& i : inputs)
      if(to_visit.count(i) != 0)
        _dfs_util(i, &to_visit, handler);
    //visit any nodes left
    while(to_visit.size() > 0)
      _dfs_util(*(to_visit.begin()), &to_visit, handler);
  }

  std::vector<Node*> topological_sort() {
    std::vector<Node*> sorted;
    auto handler = [&](Node* n) {
      sorted.push_back(n);
    };
    DFS(handler);
    return sorted;
  }

  std::vector<Node*> nodes;
  std::vector<Node*> inputs;
  std::vector<NodeEntry> outputs;
  std::map<std::string, JsonVal> attrs;
};
#else
  using namespace mxnet::ext;
#endif

#if MX_LIBRARY_VERSION <= 7
MXReturnValue addbias_gelu(const std::string& in_graph, const std::string** out_graph,
                           const std::unordered_map<std::string, std::string>& options,
                           const std::unordered_map<std::string, MXTensor>& args,
                           const std::unordered_map<std::string, MXTensor>& aux,
                           const PassResource& res) {
  //convert graph from JSON string to Graph/Node data structure
  Graph *g = Graph::fromString(in_graph);
  for(Node* n : g->nodes) {
#else
MXReturnValue addbias_gelu(mxnet::ext::Graph *g,
                           const std::unordered_map<std::string, std::string>& options) {
  for(int i=0; i < g->size(); i++) {
    mxnet::ext::Node* n = g->getNode(i);
#endif

  /////////////////////// AddBias + GELU //////////////////////////
    std::string str_ffn1 = "ffn_1_fwd";
    if (n->name.find(str_ffn1) != std::string::npos) {
      Node* node_ffn1_fwd = n;
      Node* node_ffn1_bias = node_ffn1_fwd->inputs[2].node;
      Node* node_gelu = node_ffn1_fwd->outputs[0].node;

      std::size_t pos = n->name.find("fwd");
      std::string base_name = n->name.substr(0, pos - 1);

      // remove Bias terms in FC
      node_ffn1_fwd->attrs["no_bias"] = "True";
      node_ffn1_fwd->inputs.pop_back();

      // create 4 new nodes: 2 expand_dims nodes to expand bias dimensions
      // a broadcast_like node, and BiasAdd
#if MX_LIBRARY_VERSION <= 7
      Node* node_expand_1_bias = new Node();
      node_expand_1_bias->name = base_name + "_expand_1_bias";
      node_expand_1_bias->op = "expand_dims";
      Node* node_expand_2_bias = new Node();
      node_expand_2_bias->name = base_name + "_expand_2_bias";
      node_expand_2_bias->op = "expand_dims";
      Node* node_bcst_like = new Node();
      node_bcst_like->name = base_name + "_broadcast_like";
      node_bcst_like->op = "broadcast_like";
      Node* node_add_bias = new Node();
      node_add_bias->name = base_name + "_add_bias";
      node_add_bias->op = "elemwise_add";
      g->nodes.push_back(node_expand_1_bias);
      g->nodes.push_back(node_expand_2_bias);
      g->nodes.push_back(node_bcst_like);
      g->nodes.push_back(node_add_bias);
#else
      Node* node_expand_1_bias = g->addNode(base_name + "_expand_1_bias", "expand_dims");
      Node* node_expand_2_bias = g->addNode(base_name + "_expand_2_bias", "expand_dims");
      Node* node_bcst_like = g->addNode(base_name + "_broadcast_like", "broadcast_like");
      Node* node_add_bias = g->addNode(base_name + "_add_bias", "elemwise_add");
#endif
      node_expand_1_bias->attrs["axis"] = "0";
      node_expand_1_bias->inputs.resize(1);
      node_expand_1_bias->inputs[0].node = node_ffn1_bias;
      node_expand_1_bias->inputs[0].entry = 0;
      node_expand_2_bias->attrs["axis"] = "0";
      node_expand_2_bias->inputs.resize(1);
      node_expand_2_bias->inputs[0].node = node_expand_1_bias;
      node_expand_2_bias->inputs[0].entry = 0;
      node_bcst_like->inputs.resize(2);
      node_bcst_like->inputs[0].node = node_expand_2_bias;
      node_bcst_like->inputs[0].entry = 0;
      node_bcst_like->inputs[1].node = node_ffn1_fwd;
      node_bcst_like->inputs[1].entry = 0;
      node_add_bias->inputs.resize(2);
      node_add_bias->inputs[0].node = node_ffn1_fwd;
      node_add_bias->inputs[0].entry = 0;
      node_add_bias->inputs[1].node = node_bcst_like;
      node_add_bias->inputs[1].entry = 0;
      //set BiasAdd node as gelu input
      node_gelu->inputs[0].node = node_add_bias;
      node_gelu->inputs[0].entry = 0;
    }
  }
#if MX_LIBRARY_VERSION <= 7
  //convert back to JSON string from Graph/Node
  *out_graph = new std::string(g->toString());
#endif
  return MX_SUCCESS;
}

#if MX_LIBRARY_VERSION <= 7
MXReturnValue mha_interleave(const std::string& in_graph, const std::string** out_graph,
                             const std::unordered_map<std::string, std::string>& options,
                             const std::unordered_map<std::string, MXTensor>& args,
                             const std::unordered_map<std::string, MXTensor>& aux,
                             const PassResource& res) {
  //convert graph from JSON string to Graph/Node data structure
  Graph *g = Graph::fromString(in_graph);
#else
MXReturnValue mha_interleave(mxnet::ext::Graph *g,
                             const std::unordered_map<std::string, std::string>& options) {

#endif
  /////// MHA prepare weight & bias for interleaved strategy //////
  // find shapes, number of heads, and count number of MHA layers
  std::string query0_weight = "bertencoder0_transformer0_dotproductselfattentioncell0_query_weight";
  std::string mult_qk0 = "bertencoder0_transformer0_dotproductselfattentioncell0_interleaved_matmul_selfatt_qk0";
  std::string str_projection = "_dotproductselfattentioncell0_fullyconnected0";
  int num_mha_layers = 0;
  int num_heads = 0;
  int head_dimension = 0;
  int shape0, shape1;
#if MX_LIBRARY_VERSION <= 7
  for(Node* n : g->nodes) {
#else
  for(int i=0; i < g->size(); i++) {
    mxnet::ext::Node* n = g->getNode(i);
#endif
    if (n->name.find(query0_weight) != std::string::npos) {
      std::string shape = n->attrs["__shape__"];
      int pos_comma = shape.find(",");
      shape0 = stoi(shape.substr(1, pos_comma-1));
      shape1 = stoi(shape.substr(pos_comma + 2, shape.length() - pos_comma - 3));
    }
    if (n->name.find(mult_qk0) != std::string::npos) {
      std::string h = n->attrs["heads"];
      num_heads = stoi(h);
    }
    if (n->name.find(str_projection) != std::string::npos) {
      num_mha_layers++;
    }
  }
  head_dimension = shape0 / num_heads;

  // find projection nodes and set new interleaved intputs
#if MX_LIBRARY_VERSION <= 7
  for(Node* n : g->nodes) {
#else
  for(int i=0; i < g->size(); i++) {
    mxnet::ext::Node* n = g->getNode(i);
#endif
    if (n->name.find("_dotproductselfattentioncell0_fullyconnected0") != std::string::npos) {
      Node* node_projection = n;
      std::size_t pos = node_projection->name.find("_fullyconnected0");
      std::string base_name = n->name.substr(0, pos);

      //////////////////// WEIGHTS ////////////////////
      // create new input node with interleaved weights
      std::string name_qkv_weights_interleaved = base_name + "_qkv_interleaved_weight";
#if MX_LIBRARY_VERSION <= 7
      // create a new input Node
      Node* node_qkv_weights = new Node();
      node_qkv_weights->name = name_qkv_weights_interleaved;
      node_qkv_weights->op = "null";
      MXTensor* qkv_weights_interleaved = res.alloc_arg(name_qkv_weights_interleaved,
                                                        {3 * shape0, shape1},
                                                        MXContext::CPU(0), kFloat32);
      float* qkv_w_data = qkv_weights_interleaved->data<float>();
      // read from previous values and interleave them
      MXTensor query_w = args.at(base_name + "_query_weight");
      MXTensor key_w = args.at(base_name + "_key_weight");
      MXTensor value_w = args.at(base_name + "_value_weight");
      float* query_w_data = query_w.data<float>();
      float* key_w_data = key_w.data<float>();
      float* value_w_data = value_w.data<float>();
      g->nodes.push_back(node_qkv_weights);
      g->inputs.push_back(node_qkv_weights);
#else
      Node* node_qkv_weights = g->addNode(name_qkv_weights_interleaved, "null");
      node_qkv_weights->alloc_arg({3 * shape0, shape1}, MXContext::CPU(0), kFloat32);
      float* qkv_w_data = node_qkv_weights->tensor->data<float>();
      // look back for query, key, and value weights: original data
      float *query_w_data, *key_w_data, *value_w_data;
      int found = 0;
      for (int j=i; j >= 0; j--) {
        Node* n2 = g->getNode(j);
        if (n2->name.find(base_name + "_query_weight") != std::string::npos) {
          query_w_data = n2->tensor->data<float>();
          found++;
        }
        if (n2->name.find(base_name + "_key_weight") != std::string::npos) {
          key_w_data = n2->tensor->data<float>();
          found++;
        }
        if (n2->name.find(base_name + "_value_weight") != std::string::npos) {
          value_w_data = n2->tensor->data<float>();
          found++;
        }
        if (found >= 3)
          break;
      }
#endif
      // set interleave weights
      for (int h=0; h<num_heads; ++h) {
        for (int e=0; e < head_dimension * shape1; ++e) {
          qkv_w_data[h * head_dimension * shape1 * 3 + e] =
              query_w_data[h * head_dimension * shape1 + e];
        }
        for (int e=0; e < head_dimension * shape1; ++e) {
          qkv_w_data[h * head_dimension * shape1 * 3 + head_dimension * shape1 + e] =
              key_w_data[h * head_dimension * shape1 + e];
        }
        for (int e=0; e < head_dimension * shape1; ++e) {
          qkv_w_data[h * head_dimension * shape1 * 3 + 2 * head_dimension * shape1 + e] =
              value_w_data[h * head_dimension * shape1 + e];
        }
      }
      // set connection with new input
      node_projection->inputs[1].node = node_qkv_weights;
      node_projection->inputs[1].entry = 0;

      //////////////////// BIAS ////////////////////
      // create new input node with all bias
      std::string name_qkv_bias = base_name + "_qkv_bias";
#if MX_LIBRARY_VERSION <= 7
      Node* node_qkv_bias = new Node();
      node_qkv_bias->name = name_qkv_bias;
      node_qkv_bias->op = "null";
      MXTensor* qkv_bias = res.alloc_arg(name_qkv_bias, {3 * shape0, },
                                         MXContext::CPU(0), kFloat32);
      float* qkv_bias_data = qkv_bias->data<float>();
      MXTensor query_bias = args.at(base_name + "_query_bias");
      MXTensor key_bias = args.at(base_name + "_key_bias");
      MXTensor value_bias = args.at(base_name + "_value_bias");
      float* query_bias_data = query_bias.data<float>();
      float* key_bias_data = key_bias.data<float>();
      float* value_bias_data = value_bias.data<float>();
      g->nodes.push_back(node_qkv_bias);
      g->inputs.push_back(node_qkv_bias);
#else
      Node* node_qkv_bias = g->addNode(name_qkv_bias, "null");
      node_qkv_bias->alloc_arg({3 * shape0, }, MXContext::CPU(0), kFloat32);
      float* qkv_bias_data = node_qkv_bias->tensor->data<float>();
      // look back for query, key, and value bias: original data
      float *query_bias_data, *key_bias_data, *value_bias_data;
      found = 0;
      for (int j=i; j >= 0; j--) {
        Node* n2 = g->getNode(j);
        if (n2->name.find(base_name + "_query_bias") != std::string::npos) {
          query_bias_data = n2->tensor->data<float>();
          found++;
        }
        if (n2->name.find(base_name + "_key_bias") != std::string::npos) {
          key_bias_data = n2->tensor->data<float>();
          found++;
        }
        if (n2->name.find(base_name + "_value_weight") != std::string::npos) {
          value_bias_data = n2->tensor->data<float>();
          found++;
        }
        if (found >= 3)
          break;
      }
#endif
      // concatenate bias terms
      for (int e =0; e < shape0; ++e) {
        qkv_bias_data[e] = query_bias_data[e];
      }
      for (int e=0; e < shape0; ++e) {
        qkv_bias_data[shape0 + e] = key_bias_data[e];
      }
      for (int e=0; e < shape0; ++e) {
        qkv_bias_data[2 * shape0 + e] = value_bias_data[e];
      }
      // set connection with new input
      node_projection->inputs[2].node = node_qkv_bias;
      node_projection->inputs[2].entry = 0;
    }
  }

#if MX_LIBRARY_VERSION <= 7
  //convert back to JSON string from Graph/Node
  *out_graph = new std::string(g->toString());
#endif
  return MX_SUCCESS;
}

REGISTER_PASS(addbias_gelu)
.setBody(addbias_gelu);

REGISTER_PASS(mha_interleave)
.setBody(mha_interleave);

MXReturnValue initialize(int version) {
  if (version >= 10700) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    std::cout << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}
