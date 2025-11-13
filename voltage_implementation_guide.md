# Voltage算法在prima.cpp中的实现指南

## 论文概述

**Voltage: When the Edge Meets Transformers: Distributed Inference with Transformer Models**

论文提出了一种新的分布式Transformer推理方法，主要特点：

1. **位置级并行化（Position-wise Parallelism）**：利用Transformer的位置无关特性
2. **自注意力机制优化**：通过重新排列计算顺序减少通信开销
3. **自适应计算策略**：根据分区大小、输入大小和特征维度选择最优计算顺序

## Prima.cpp当前架构分析

### 当前分布式策略：层级并行（Layer Parallelism）
- 每个设备负责处理特定的Transformer层
- 通过`this_layer_is_mine()`函数确定层分配
- 使用ZMQ进行设备间通信

### 关键数据结构
```cpp
struct llama_cparams {
    uint32_t n_world;                    // 设备数量
    uint32_t rank;                       // 当前设备排名
    uint32_t n_layer_window[32];         // 每个设备的层窗口大小
    // ...
};
```

## Voltage算法核心实现

### 1. 位置级分区数据结构

```cpp
// 新增位置级分区参数
struct voltage_params {
    bool     enable_voltage;             // 启用Voltage算法
    uint32_t partition_size;             // 位置分区大小
    uint32_t n_partitions;               // 分区数量
    uint32_t my_partition_start;         // 当前设备的分区起始位置
    uint32_t my_partition_end;           // 当前设备的分区结束位置
};

// 扩展llama_cparams
struct llama_cparams {
    // ... 现有字段
    voltage_params voltage;
};
```

### 2. 位置级分区函数

```cpp
// 判断某个位置是否属于当前设备
static bool this_position_is_mine(
    uint32_t pos,
    uint32_t n_world,
    uint32_t my_rank,
    uint32_t seq_len) {
    
    uint32_t partition_size = seq_len / n_world;
    uint32_t remainder = seq_len % n_world;
    
    uint32_t start_pos, end_pos;
    if (my_rank < remainder) {
        start_pos = my_rank * (partition_size + 1);
        end_pos = start_pos + partition_size + 1;
    } else {
        start_pos = my_rank * partition_size + remainder;
        end_pos = start_pos + partition_size;
    }
    
    return pos >= start_pos && pos < end_pos;
}

// 获取当前设备负责的位置范围
static void get_my_position_range(
    uint32_t n_world,
    uint32_t my_rank,
    uint32_t seq_len,
    uint32_t* start_pos,
    uint32_t* end_pos) {
    
    uint32_t partition_size = seq_len / n_world;
    uint32_t remainder = seq_len % n_world;
    
    if (my_rank < remainder) {
        *start_pos = my_rank * (partition_size + 1);
        *end_pos = *start_pos + partition_size + 1;
    } else {
        *start_pos = my_rank * partition_size + remainder;
        *end_pos = *start_pos + partition_size;
    }
}
```

### 3. 自注意力机制优化

```cpp
// Voltage自注意力计算策略
enum voltage_attention_strategy {
    VOLTAGE_STRATEGY_QKV_FIRST,    // 先计算Q,K,V再计算注意力
    VOLTAGE_STRATEGY_QK_FIRST,     // 先计算QK^T再与V相乘
    VOLTAGE_STRATEGY_ADAPTIVE      // 自适应选择策略
};

// 选择最优计算策略
static voltage_attention_strategy select_optimal_strategy(
    uint32_t seq_len,
    uint32_t n_head,
    uint32_t head_dim,
    uint32_t partition_size) {
    
    // 根据论文中的复杂度分析选择策略
    float cost_qkv_first = 2.0f * seq_len * head_dim + 
                          seq_len * seq_len / partition_size;
    float cost_qk_first = seq_len * seq_len + 
                         seq_len * head_dim / partition_size;
    
    if (cost_qk_first < cost_qkv_first) {
        return VOLTAGE_STRATEGY_QK_FIRST;
    } else {
        return VOLTAGE_STRATEGY_QKV_FIRST;
    }
}
```

### 4. 修改Transformer层计算

```cpp
// 在llama_build_graph中添加Voltage支持
static std::vector<struct ggml_cgraph *> llama_build_graph_voltage(
         llama_context & lctx,
    const llama_ubatch & batch,
                  bool   worst_case) {
    
    const auto & model = lctx.model;
    const uint32_t n_world = lctx.cparams.n_world;
    const uint32_t my_rank = lctx.cparams.rank;
    const voltage_params & voltage = lctx.cparams.voltage;
    
    if (!voltage.enable_voltage) {
        // 回退到原始的层级并行
        return llama_build_graph(lctx, batch, worst_case);
    }
    
    // 获取当前设备负责的位置范围
    uint32_t start_pos, end_pos;
    get_my_position_range(n_world, my_rank, batch.n_tokens, 
                         &start_pos, &end_pos);
    
    // 为每一层创建位置级分区的计算图
    std::vector<struct ggml_cgraph *> sub_gfs;
    
    for (int il = 0; il < model.hparams.n_layer; ++il) {
        struct ggml_cgraph * sub_gf = ggml_new_graph_custom(
            ctx0, llama_model_max_nodes(model), false);
        
        // 实现位置级的注意力计算
        struct ggml_tensor * cur = voltage_attention_layer(
            lctx, il, inpL, start_pos, end_pos, sub_gf);
        
        sub_gfs.push_back(sub_gf);
    }
    
    return sub_gfs;
}
```

### 5. 位置级注意力计算

```cpp
static struct ggml_tensor * voltage_attention_layer(
    llama_context & lctx,
    int layer_idx,
    struct ggml_tensor * inp,
    uint32_t start_pos,
    uint32_t end_pos,
    struct ggml_cgraph * gf) {
    
    const auto & model = lctx.model;
    const auto & hparams = model.hparams;
    const auto & layer = model.layers[layer_idx];
    
    const int64_t n_embd = hparams.n_embd;
    const int64_t n_head = hparams.n_head;
    const int64_t n_head_kv = hparams.n_head_kv;
    const int64_t n_embd_head = hparams.n_embd_head();
    
    // 提取当前设备负责的位置切片
    struct ggml_tensor * inp_slice = ggml_view_2d(
        gf->ctx, inp,
        n_embd, end_pos - start_pos,
        inp->nb[1], start_pos * inp->nb[1]);
    
    // 计算Q, K, V (只计算当前分区)
    struct ggml_tensor * Qcur = ggml_mul_mat(gf->ctx, layer.wq, inp_slice);
    struct ggml_tensor * Kcur = ggml_mul_mat(gf->ctx, layer.wk, inp_slice);
    struct ggml_tensor * Vcur = ggml_mul_mat(gf->ctx, layer.wv, inp_slice);
    
    // 选择计算策略
    voltage_attention_strategy strategy = select_optimal_strategy(
        end_pos - start_pos, n_head, n_embd_head, end_pos - start_pos);
    
    struct ggml_tensor * attn_out;
    if (strategy == VOLTAGE_STRATEGY_QK_FIRST) {
        attn_out = voltage_attention_qk_first(gf->ctx, Qcur, Kcur, Vcur);
    } else {
        attn_out = voltage_attention_qkv_first(gf->ctx, Qcur, Kcur, Vcur);
    }
    
    // 输出投影
    struct ggml_tensor * result = ggml_mul_mat(gf->ctx, layer.wo, attn_out);
    
    ggml_build_forward_expand(gf, result);
    return result;
}
```

### 6. 通信协议扩展

```cpp
// 扩展同步元数据结构
struct voltage_sync_meta {
    uint32_t partition_start;
    uint32_t partition_end;
    uint32_t strategy;
};

// 发送位置级分区数据
static void llama_send_voltage_tensors(
    zmq::socket_t & socket,
    struct ggml_tensor * tensor,
    uint32_t start_pos,
    uint32_t end_pos) {
    
    std::vector<zmq::message_t> send_msgs;
    
    // 发送分区信息
    send_msgs.emplace_back("partition_start", strlen("partition_start"));
    send_msgs.emplace_back(&start_pos, sizeof(start_pos));
    
    send_msgs.emplace_back("partition_end", strlen("partition_end"));
    send_msgs.emplace_back(&end_pos, sizeof(end_pos));
    
    // 发送张量数据
    size_t tensor_size = ggml_nbytes(tensor);
    send_msgs.emplace_back("tensor_data", strlen("tensor_data"));
    send_msgs.emplace_back(ggml_get_data(tensor), tensor_size);
    
    zmq::send_multipart(socket, send_msgs);
}
```

## 实现步骤

### 第一阶段：基础框架
1. 在`common/common.h`中添加`voltage_params`结构
2. 在`common/arg.cpp`中添加Voltage相关命令行参数
3. 实现位置级分区的基础函数

### 第二阶段：核心算法
1. 实现`voltage_attention_layer`函数
2. 添加自适应策略选择逻辑
3. 修改`llama_build_graph`支持位置级并行

### 第三阶段：通信优化
1. 扩展ZMQ通信协议
2. 实现位置级数据传输
3. 优化通信开销

### 第四阶段：测试和优化
1. 添加性能测试
2. 与原始层级并行进行对比
3. 根据实验结果优化算法

## 使用示例

```bash
# 启用Voltage算法的分布式推理
./llama-cli -m model.gguf \
    --world 4 --rank 0 \
    --voltage \
    --partition-strategy adaptive \
    -p "What is AI?" -n 256
```

## 注意事项

1. **兼容性**：确保Voltage模式与现有的层级并行模式兼容
2. **内存管理**：位置级分区可能改变内存访问模式
3. **通信开销**：需要仔细平衡计算和通信的开销
4. **调试支持**：添加详细的日志和调试信息

这个实现指南提供了在prima.cpp中集成Voltage算法的完整框架。您可以根据具体需求逐步实现这些功能。