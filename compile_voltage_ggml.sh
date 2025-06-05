#!/bin/bash

# VOLTAGE算法GGML集成版本编译脚本
# 解决编译依赖问题的简化版本

echo "=== VOLTAGE算法GGML集成版本编译脚本 ==="
echo

# 检查必要的源文件
if [ ! -f "voltage_prima_complete.cpp" ]; then
    echo "错误: voltage_prima_complete.cpp 文件不存在"
    exit 1
fi

if [ ! -f "ggml/src/ggml.c" ]; then
    echo "错误: ggml/src/ggml.c 文件不存在"
    exit 1
fi

echo "步骤1: 编译GGML核心库..."

# 编译GGML核心文件
gcc -std=c11 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c ggml/src/ggml.c -o ggml_core.o

if [ $? -ne 0 ]; then
    echo "错误: GGML核心库编译失败"
    exit 1
fi

echo "步骤2: 编译GGML后端..."

# 编译GGML后端
g++ -std=c++11 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c ggml/src/ggml-backend.cpp -o ggml_backend.o

if [ $? -ne 0 ]; then
    echo "错误: GGML后端编译失败"
    exit 1
fi

echo "步骤3: 编译GGML内存分配器..."

# 编译GGML内存分配器
gcc -std=c11 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c ggml/src/ggml-alloc.c -o ggml_alloc.o

if [ $? -ne 0 ]; then
    echo "错误: GGML内存分配器编译失败"
    exit 1
fi

echo "步骤4: 编译VOLTAGE算法主程序..."

# 编译VOLTAGE主程序（简化版本，移除ZMQ依赖）
g++ -std=c++11 -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src -Iinclude -Icommon \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -DVOLTAGE_STANDALONE_BUILD \
    -pthread -fopenmp \
    voltage_prima_complete.cpp \
    ggml_core.o ggml_backend.o ggml_alloc.o \
    -o voltage_ggml_standalone

if [ $? -ne 0 ]; then
    echo "错误: VOLTAGE主程序编译失败"
    exit 1
fi

echo
echo "✅ 编译成功！"
echo "可执行文件: voltage_ggml_standalone"
echo
echo "运行方法:"
echo "  ./voltage_ggml_standalone"
echo
echo "注意: 这是简化版本，移除了ZMQ和llama.cpp的完整依赖"
echo "      专注于展示VOLTAGE算法与GGML的集成"