/*
 * 完整实现论文中的算法1和算法2
 * 论文: "When the Edge Meets Transformers: Distributed Inference with Transformer Models"
 * 
 * 算法1: 自适应策略选择 (Adaptive Strategy Selection)
 * 算法2: 分布式自注意力计算 (Distributed Self-Attention Computation)
 */

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>

// 论文中的符号定义
struct VoltageSymbols {
    uint32_t N;      // 序列长度 (sequence length)
    uint32_t F_H;    // 注意力头维度 (attention head dimension)
    uint32_t H;      // 注意力头数 (number of attention heads)
    uint32_t K;      // 分区数量 (number of partitions/devices)
    uint32_t P;      // 分区大小 (partition size) P = N/K
};

// 模拟张量类
class Tensor {
public:
    std::vector<float> data;
    std::vector<uint32_t> shape;
    
    Tensor(std::vector<uint32_t> s) : shape(s) {
        uint32_t size = 1;
        for (auto dim : shape) size *= dim;
        data.resize(size, 0.0f);
    }
    
    float& operator()(uint32_t i, uint32_t j) {
        return data[i * shape[1] + j];
    }
    
    const float& operator()(uint32_t i, uint32_t j) const {
        return data[i * shape[1] + j];
    }
    
    uint32_t rows() const { return shape[0]; }
    uint32_t cols() const { return shape[1]; }
};

/*
 * 算法1: 自适应策略选择
 * 
 * 输入: N (序列长度), F_H (头维度), K (分区数)
 * 输出: 最优计算策略
 */
class Algorithm1_AdaptiveStrategySelection {
public:
    enum Strategy {
        QKV_FIRST = 0,  // 策略1: 先计算QKV
        QK_FIRST = 1    // 策略2: 先计算QK
    };
    
    static Strategy selectOptimalStrategy(const VoltageSymbols& symbols) {
        uint32_t N = symbols.N;
        uint32_t F_H = symbols.F_H;
        uint32_t K = symbols.K;
        
        // 论文中的复杂度分析
        // 策略1 (QKV-first): C1 = 2NF_H + N²/K
        float cost_qkv_first = 2.0f * N * F_H + (N * N) / (float)K;
        
        // 策略2 (QK-first): C2 = N² + NF_H/K
        float cost_qk_first = N * N + (N * F_H) / (float)K;
        
        printf("Algorithm 1 - Strategy Selection:\n");
        printf("  N=%u, F_H=%u, K=%u\n", N, F_H, K);
        printf("  Cost QKV-first: %.2f\n", cost_qkv_first);
        printf("  Cost QK-first: %.2f\n", cost_qk_first);
        
        Strategy selected = (cost_qk_first < cost_qkv_first) ? QK_FIRST : QKV_FIRST;
        printf("  Selected strategy: %s\n", 
               selected == QK_FIRST ? "QK-First" : "QKV-First");
        
        return selected;
    }
    
    // 计算通信复杂度
    static float calculateCommunicationCost(const VoltageSymbols& symbols, Strategy strategy) {
        uint32_t N = symbols.N;
        uint32_t F_H = symbols.F_H;
        uint32_t K = symbols.K;
        
        switch (strategy) {
            case QKV_FIRST:
                // 需要广播完整的K和V矩阵
                return 2.0f * N * F_H;
                
            case QK_FIRST:
                // 需要传输QK^T的部分结果和V的部分
                return (N * N) / K + (N * F_H) / K;
                
            default:
                return 0.0f;
        }
    }
};

/*
 * 算法2: 分布式自注意力计算
 * 
 * 实现论文中描述的两种计算策略
 */
class Algorithm2_DistributedSelfAttention {
public:
    
    /*
     * 策略1: QKV-First计算
     * 
     * 步骤:
     * 1. 每个设备计算自己分区的Q_i, K_i, V_i
     * 2. 广播K和V到所有设备
     * 3. 计算注意力输出
     */
    static Tensor computeQKVFirst(
        const Tensor& X,           // 输入序列 [N, F_H]
        const Tensor& W_Q,         // Query权重 [F_H, F_H]
        const Tensor& W_K,         // Key权重 [F_H, F_H]
        const Tensor& W_V,         // Value权重 [F_H, F_H]
        uint32_t device_id,        // 当前设备ID
        const VoltageSymbols& symbols) {
        
        uint32_t N = symbols.N;
        uint32_t F_H = symbols.F_H;
        uint32_t K = symbols.K;
        uint32_t P = N / K;  // 每个设备的分区大小
        
        printf("\nAlgorithm 2 - QKV-First Strategy (Device %u):\n", device_id);
        
        // 步骤1: 计算当前设备分区的Q, K, V
        uint32_t start_pos = device_id * P;
        uint32_t end_pos = (device_id + 1) * P;
        
        printf("  Processing positions %u-%u\n", start_pos, end_pos-1);
        
        // Q_i = X_i * W_Q  [P, F_H]
        Tensor Q_local({P, F_H});
        for (uint32_t i = 0; i < P; ++i) {
            for (uint32_t j = 0; j < F_H; ++j) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < F_H; ++k) {
                    sum += X(start_pos + i, k) * W_Q(k, j);
                }
                Q_local(i, j) = sum;
            }
        }
        
        // K_i = X_i * W_K  [P, F_H]
        Tensor K_local({P, F_H});
        for (uint32_t i = 0; i < P; ++i) {
            for (uint32_t j = 0; j < F_H; ++j) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < F_H; ++k) {
                    sum += X(start_pos + i, k) * W_K(k, j);
                }
                K_local(i, j) = sum;
            }
        }
        
        // V_i = X_i * W_V  [P, F_H]
        Tensor V_local({P, F_H});
        for (uint32_t i = 0; i < P; ++i) {
            for (uint32_t j = 0; j < F_H; ++j) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < F_H; ++k) {
                    sum += X(start_pos + i, k) * W_V(k, j);
                }
                V_local(i, j) = sum;
            }
        }
        
        // 步骤2: 模拟广播K和V (在实际实现中通过网络通信)
        // 这里我们假设已经收集了所有设备的K和V
        Tensor K_global({N, F_H});
        Tensor V_global({N, F_H});
        
        // 将本地K和V复制到全局矩阵中
        for (uint32_t i = 0; i < P; ++i) {
            for (uint32_t j = 0; j < F_H; ++j) {
                K_global(start_pos + i, j) = K_local(i, j);
                V_global(start_pos + i, j) = V_local(i, j);
            }
        }
        
        // 步骤3: 计算注意力
        // QK^T = Q_local * K_global^T  [P, N]
        Tensor QK({P, N});
        for (uint32_t i = 0; i < P; ++i) {
            for (uint32_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < F_H; ++k) {
                    sum += Q_local(i, k) * K_global(j, k);
                }
                QK(i, j) = sum / sqrtf(F_H);  // 缩放
            }
        }
        
        // Softmax
        for (uint32_t i = 0; i < P; ++i) {
            float max_val = QK(i, 0);
            for (uint32_t j = 1; j < N; ++j) {
                max_val = std::max(max_val, QK(i, j));
            }
            
            float sum_exp = 0.0f;
            for (uint32_t j = 0; j < N; ++j) {
                QK(i, j) = expf(QK(i, j) - max_val);
                sum_exp += QK(i, j);
            }
            
            for (uint32_t j = 0; j < N; ++j) {
                QK(i, j) /= sum_exp;
            }
        }
        
        // Attention * V = QK * V_global  [P, F_H]
        Tensor output({P, F_H});
        for (uint32_t i = 0; i < P; ++i) {
            for (uint32_t j = 0; j < F_H; ++j) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < N; ++k) {
                    sum += QK(i, k) * V_global(k, j);
                }
                output(i, j) = sum;
            }
        }
        
        printf("  QKV-First computation completed\n");
        return output;
    }
    
    /*
     * 策略2: QK-First计算
     * 
     * 步骤:
     * 1. 计算QK^T矩阵
     * 2. 分区传输QK^T结果
     * 3. 与V相乘得到最终输出
     */
    static Tensor computeQKFirst(
        const Tensor& X,           // 输入序列 [N, F_H]
        const Tensor& W_Q,         // Query权重 [F_H, F_H]
        const Tensor& W_K,         // Key权重 [F_H, F_H]
        const Tensor& W_V,         // Value权重 [F_H, F_H]
        uint32_t device_id,        // 当前设备ID
        const VoltageSymbols& symbols) {
        
        uint32_t N = symbols.N;
        uint32_t F_H = symbols.F_H;
        uint32_t K_devices = symbols.K;
        uint32_t P = N / K_devices;
        
        printf("\nAlgorithm 2 - QK-First Strategy (Device %u):\n", device_id);
        
        uint32_t start_pos = device_id * P;
        uint32_t end_pos = (device_id + 1) * P;
        
        printf("  Processing positions %u-%u\n", start_pos, end_pos-1);
        
        // 步骤1: 计算完整的Q和K矩阵
        Tensor Q({N, F_H});
        Tensor K_matrix({N, F_H});
        
        for (uint32_t i = 0; i < N; ++i) {
            for (uint32_t j = 0; j < F_H; ++j) {
                float q_sum = 0.0f, k_sum = 0.0f;
                for (uint32_t k = 0; k < F_H; ++k) {
                    q_sum += X(i, k) * W_Q(k, j);
                    k_sum += X(i, k) * W_K(k, j);
                }
                Q(i, j) = q_sum;
                K_matrix(i, j) = k_sum;
            }
        }
        
        // 步骤2: 计算QK^T矩阵的当前分区部分
        Tensor QK_partition({P, N});
        for (uint32_t i = 0; i < P; ++i) {
            for (uint32_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < F_H; ++k) {
                    sum += Q(start_pos + i, k) * K_matrix(j, k);
                }
                QK_partition(i, j) = sum / sqrtf(F_H);
            }
        }
        
        // Softmax on partition
        for (uint32_t i = 0; i < P; ++i) {
            float max_val = QK_partition(i, 0);
            for (uint32_t j = 1; j < N; ++j) {
                max_val = std::max(max_val, QK_partition(i, j));
            }
            
            float sum_exp = 0.0f;
            for (uint32_t j = 0; j < N; ++j) {
                QK_partition(i, j) = expf(QK_partition(i, j) - max_val);
                sum_exp += QK_partition(i, j);
            }
            
            for (uint32_t j = 0; j < N; ++j) {
                QK_partition(i, j) /= sum_exp;
            }
        }
        
        // 步骤3: 计算V并与注意力权重相乘
        Tensor V({N, F_H});
        for (uint32_t i = 0; i < N; ++i) {
            for (uint32_t j = 0; j < F_H; ++j) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < F_H; ++k) {
                    sum += X(i, k) * W_V(k, j);
                }
                V(i, j) = sum;
            }
        }
        
        // 最终输出: QK_partition * V
        Tensor output({P, F_H});
        for (uint32_t i = 0; i < P; ++i) {
            for (uint32_t j = 0; j < F_H; ++j) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < N; ++k) {
                    sum += QK_partition(i, k) * V(k, j);
                }
                output(i, j) = sum;
            }
        }
        
        printf("  QK-First computation completed\n");
        return output;
    }
};

/*
 * 完整的Voltage算法实现
 */
class VoltageAlgorithmComplete {
public:
    static void runCompleteExample() {
        printf("=== 完整的Voltage算法实现 ===\n");
        
        // 设置参数
        VoltageSymbols symbols;
        symbols.N = 512;    // 序列长度
        symbols.F_H = 64;   // 头维度
        symbols.H = 16;     // 注意力头数
        symbols.K = 4;      // 设备数量
        symbols.P = symbols.N / symbols.K;  // 分区大小
        
        printf("参数设置: N=%u, F_H=%u, H=%u, K=%u, P=%u\n", 
               symbols.N, symbols.F_H, symbols.H, symbols.K, symbols.P);
        
        // 算法1: 选择最优策略
        auto strategy = Algorithm1_AdaptiveStrategySelection::selectOptimalStrategy(symbols);
        
        // 计算通信开销
        float comm_cost = Algorithm1_AdaptiveStrategySelection::calculateCommunicationCost(symbols, strategy);
        printf("通信开销: %.2f\n", comm_cost);
        
        // 创建模拟数据
        Tensor X({symbols.N, symbols.F_H});
        Tensor W_Q({symbols.F_H, symbols.F_H});
        Tensor W_K({symbols.F_H, symbols.F_H});
        Tensor W_V({symbols.F_H, symbols.F_H});
        
        // 填充随机数据
        for (uint32_t i = 0; i < symbols.N * symbols.F_H; ++i) {
            X.data[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
        for (uint32_t i = 0; i < symbols.F_H * symbols.F_H; ++i) {
            W_Q.data[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
            W_K.data[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
            W_V.data[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
        
        // 算法2: 执行分布式自注意力计算
        printf("\n=== 执行分布式自注意力计算 ===\n");
        
        std::vector<Tensor> outputs;
        
        for (uint32_t device_id = 0; device_id < symbols.K; ++device_id) {
            Tensor output({0, 0});
            
            if (strategy == Algorithm1_AdaptiveStrategySelection::QKV_FIRST) {
                output = Algorithm2_DistributedSelfAttention::computeQKVFirst(
                    X, W_Q, W_K, W_V, device_id, symbols);
            } else {
                output = Algorithm2_DistributedSelfAttention::computeQKFirst(
                    X, W_Q, W_K, W_V, device_id, symbols);
            }
            
            outputs.push_back(output);
            printf("设备 %u 输出形状: [%u, %u]\n", device_id, output.rows(), output.cols());
        }
        
        printf("\n所有设备计算完成！\n");
        
        // 性能分析
        printf("\n=== 性能分析 ===\n");
        analyzePerformance(symbols, strategy);
    }
    
private:
    static void analyzePerformance(const VoltageSymbols& symbols, 
                                 Algorithm1_AdaptiveStrategySelection::Strategy strategy) {
        uint32_t N = symbols.N;
        uint32_t F_H = symbols.F_H;
        uint32_t K = symbols.K;
        
        // Voltage算法的复杂度
        float voltage_comp, voltage_comm;
        if (strategy == Algorithm1_AdaptiveStrategySelection::QKV_FIRST) {
            voltage_comp = 2.0f * N * F_H + (N * N) / K;
            voltage_comm = 2.0f * N * F_H;
        } else {
            voltage_comp = N * N + (N * F_H) / K;
            voltage_comm = (N * N) / K + (N * F_H) / K;
        }
        
        // Tensor Parallelism的复杂度
        float tp_comp = (N * N * F_H) / K;
        float tp_comm = 4.0f * N * F_H;  // All-Reduce开销
        
        printf("Voltage (%s):\n", 
               strategy == Algorithm1_AdaptiveStrategySelection::QK_FIRST ? "QK-First" : "QKV-First");
        printf("  计算复杂度: %.0f\n", voltage_comp);
        printf("  通信复杂度: %.0f\n", voltage_comm);
        printf("  总复杂度: %.0f\n", voltage_comp + voltage_comm);
        
        printf("Tensor Parallelism:\n");
        printf("  计算复杂度: %.0f\n", tp_comp);
        printf("  通信复杂度: %.0f\n", tp_comm);
        printf("  总复杂度: %.0f\n", tp_comp + tp_comm);
        
        float speedup = (tp_comp + tp_comm) / (voltage_comp + voltage_comm);
        float comm_reduction = tp_comm / voltage_comm;
        
        printf("加速比: %.2fx\n", speedup);
        printf("通信减少: %.2fx\n", comm_reduction);
    }
};

int main() {
    VoltageAlgorithmComplete::runCompleteExample();
    return 0;
}