#!/bin/bash

# Voltage算法测试脚本
# 用于验证在prima.cpp中实现的Voltage分布式推理

echo "=== Voltage Algorithm Test Script ==="
echo "Testing Voltage position-wise parallelism implementation"
echo

# 检查是否有模型文件
MODEL_PATH="download/qwq-32b-q4_k_m.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model file not found at $MODEL_PATH"
    echo "Please download a model first:"
    echo "mkdir -p download"
    echo "wget https://huggingface.co/Qwen/QwQ-32B-GGUF/resolve/main/qwq-32b-q4_k_m.gguf -P download/"
    echo
fi

# 测试参数
WORLD_SIZE=4
PROMPT="What is the difference between distributed computing and parallel computing?"
MAX_TOKENS=128
CONTEXT_SIZE=1024

echo "Test Configuration:"
echo "  World Size: $WORLD_SIZE"
echo "  Prompt: $PROMPT"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Context Size: $CONTEXT_SIZE"
echo

# 函数：运行单设备测试（基准）
run_single_device_test() {
    echo "=== Single Device Baseline Test ==="
    if [ -f "$MODEL_PATH" ]; then
        time ./llama-cli -m "$MODEL_PATH" \
            -c $CONTEXT_SIZE \
            -n $MAX_TOKENS \
            -p "$PROMPT" \
            --temp 0.7 \
            --top-p 0.9
    else
        echo "Skipping single device test - no model file"
    fi
    echo
}

# 函数：运行Prima.cpp层级并行测试
run_layer_parallel_test() {
    echo "=== Prima.cpp Layer Parallelism Test ==="
    echo "Note: This requires multiple devices/containers to run properly"
    echo "Command for rank 0 (head device):"
    echo "./llama-cli -m $MODEL_PATH \\"
    echo "    -c $CONTEXT_SIZE -n $MAX_TOKENS \\"
    echo "    -p \"$PROMPT\" \\"
    echo "    --world $WORLD_SIZE --rank 0 \\"
    echo "    --master 192.168.1.2 --next 192.168.1.3 \\"
    echo "    --prefetch"
    echo
    echo "Commands for worker devices (rank 1-3):"
    for rank in {1..3}; do
        next_rank=$(( (rank + 1) % WORLD_SIZE ))
        if [ $next_rank -eq 0 ]; then
            next_ip="192.168.1.2"
        else
            next_ip="192.168.1.$((next_rank + 2))"
        fi
        echo "# Rank $rank:"
        echo "./llama-cli -m $MODEL_PATH \\"
        echo "    --world $WORLD_SIZE --rank $rank \\"
        echo "    --master 192.168.1.2 --next $next_ip \\"
        echo "    --prefetch"
        echo
    done
}

# 函数：运行Voltage位置级并行测试
run_voltage_test() {
    echo "=== Voltage Position-wise Parallelism Test ==="
    echo "Note: This is the proposed implementation"
    echo "Command for rank 0 (head device):"
    echo "./llama-cli -m $MODEL_PATH \\"
    echo "    -c $CONTEXT_SIZE -n $MAX_TOKENS \\"
    echo "    -p \"$PROMPT\" \\"
    echo "    --world $WORLD_SIZE --rank 0 \\"
    echo "    --master 192.168.1.2 --next 192.168.1.3 \\"
    echo "    --voltage \\"
    echo "    --voltage-strategy adaptive \\"
    echo "    --prefetch"
    echo
    echo "Commands for worker devices (rank 1-3):"
    for rank in {1..3}; do
        next_rank=$(( (rank + 1) % WORLD_SIZE ))
        if [ $next_rank -eq 0 ]; then
            next_ip="192.168.1.2"
        else
            next_ip="192.168.1.$((next_rank + 2))"
        fi
        echo "# Rank $rank:"
        echo "./llama-cli -m $MODEL_PATH \\"
        echo "    --world $WORLD_SIZE --rank $rank \\"
        echo "    --master 192.168.1.2 --next $next_ip \\"
        echo "    --voltage \\"
        echo "    --voltage-strategy adaptive \\"
        echo "    --prefetch"
        echo
    done
}

# 函数：性能对比分析
run_performance_analysis() {
    echo "=== Performance Analysis ==="
    echo "Running Voltage prototype performance analysis..."
    
    if [ -f "voltage_prototype" ]; then
        ./voltage_prototype
    else
        echo "Compiling voltage prototype..."
        g++ -o voltage_prototype voltage_prototype.cpp -std=c++11
        if [ $? -eq 0 ]; then
            ./voltage_prototype
        else
            echo "Failed to compile voltage prototype"
        fi
    fi
    echo
}

# 函数：验证实现正确性
verify_implementation() {
    echo "=== Implementation Verification ==="
    echo "Checking if Voltage parameters are properly integrated..."
    
    # 检查是否添加了Voltage参数
    if grep -q "enable_voltage" common/common.h; then
        echo "✓ Voltage parameters found in common.h"
    else
        echo "✗ Voltage parameters not found in common.h"
        echo "  Please apply the voltage_integration_patch.diff"
    fi
    
    if grep -q "voltage" common/arg.cpp; then
        echo "✓ Voltage command line arguments found"
    else
        echo "✗ Voltage command line arguments not found"
        echo "  Please apply the voltage_integration_patch.diff"
    fi
    
    if grep -q "voltage_" src/llama.cpp; then
        echo "✓ Voltage functions found in llama.cpp"
    else
        echo "✗ Voltage functions not found in llama.cpp"
        echo "  Please apply the voltage_integration_patch.diff"
    fi
    echo
}

# 函数：显示使用说明
show_usage() {
    echo "=== Usage Instructions ==="
    echo "To implement Voltage in prima.cpp:"
    echo
    echo "1. Apply the integration patch:"
    echo "   git apply voltage_integration_patch.diff"
    echo
    echo "2. Rebuild prima.cpp:"
    echo "   make clean"
    echo "   make USE_HIGHS=1 -j\$(nproc)  # For rank 0 device"
    echo "   make -j\$(nproc)              # For worker devices"
    echo
    echo "3. Test the implementation:"
    echo "   ./test_voltage.sh"
    echo
    echo "4. Run distributed inference with Voltage:"
    echo "   # On each device, use --voltage flag"
    echo "   ./llama-cli -m model.gguf --voltage --world 4 --rank 0 ..."
    echo
}

# 主执行流程
main() {
    case "${1:-all}" in
        "single")
            run_single_device_test
            ;;
        "layer")
            run_layer_parallel_test
            ;;
        "voltage")
            run_voltage_test
            ;;
        "perf")
            run_performance_analysis
            ;;
        "verify")
            verify_implementation
            ;;
        "usage")
            show_usage
            ;;
        "all")
            verify_implementation
            run_performance_analysis
            run_single_device_test
            run_layer_parallel_test
            run_voltage_test
            show_usage
            ;;
        *)
            echo "Usage: $0 [single|layer|voltage|perf|verify|usage|all]"
            echo "  single  - Run single device baseline test"
            echo "  layer   - Show layer parallelism commands"
            echo "  voltage - Show Voltage parallelism commands"
            echo "  perf    - Run performance analysis"
            echo "  verify  - Verify implementation status"
            echo "  usage   - Show usage instructions"
            echo "  all     - Run all tests (default)"
            ;;
    esac
}

# 运行主函数
main "$@"