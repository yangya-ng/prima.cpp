#!/bin/bash

# VOLTAGE算法GGML集成版本编译脚本 - 修复版本
# 包含所有必要的GGML组件和量化函数

echo "=== VOLTAGE算法GGML集成版本编译脚本 (修复版本) ==="
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

if [ ! -f "ggml/src/ggml-quants.c" ]; then
    echo "错误: ggml/src/ggml-quants.c 文件不存在"
    exit 1
fi

# Added: Check for llama.cpp source file
if [ ! -f "src/llama.cpp" ]; then
    echo "错误: llama.cpp 文件不存在. 请确保 llama.cpp 文件在脚本同一目录下."
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

echo "步骤2: 编译GGML量化函数..."

# 编译GGML量化函数
gcc -std=c11 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c ggml/src/ggml-quants.c -o ggml_quants.o

if [ $? -ne 0 ]; then
    echo "错误: GGML量化函数编译失败"
    exit 1
fi

echo "步骤3: 编译GGML后端..."

# 编译GGML后端
g++ -std=c++17 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c ggml/src/ggml-backend.cpp -o ggml_backend.o

if [ $? -ne 0 ]; then
    echo "错误: GGML后端编译失败"
    exit 1
fi

echo "步骤4: 编译GGML内存分配器..."

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

echo "步骤5: 编译GGML架构特定优化..."

# 编译GGML架构特定优化 (包含量化矩阵函数)
gcc -std=c11 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c ggml/src/ggml-aarch64.c -o ggml_aarch64.o

if [ $? -ne 0 ]; then
    echo "错误: GGML架构特定优化编译失败"
    exit 1
fi

echo "步骤6: 编译llamafile SGEMM优化..."

# 编译llamafile SGEMM优化
g++ -std=c++17 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c ggml/src/llamafile/sgemm.cpp -o sgemm.o

if [ $? -ne 0 ]; then
    echo "错误: llamafile SGEMM编译失败"
    exit 1
fi

# Compile required source files
echo "步骤7: 编译llama-vocab.cpp..."
g++ -std=c++17 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src -Iinclude -Icommon \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c src/llama-vocab.cpp -o llama_vocab.o

if [ $? -ne 0 ]; then
    echo "错误: llama-vocab.cpp 编译失败"
    exit 1
fi

echo "步骤8: 编译unicode.cpp..."
g++ -std=c++17 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src -Iinclude -Icommon \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c src/unicode.cpp -o unicode.o

if [ $? -ne 0 ]; then
    echo "错误: unicode.cpp 编译失败"
    exit 1
fi

echo "步骤9: 编译unicode-data.cpp..."
g++ -std=c++17 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src -Iinclude -Icommon \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c src/unicode-data.cpp -o unicode_data.o

if [ $? -ne 0 ]; then
    echo "错误: unicode-data.cpp 编译失败"
    exit 1
fi

echo "步骤10: 编译common.cpp..."
g++ -std=c++17 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src -Iinclude -Icommon \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c common/common.cpp -o common.o

if [ $? -ne 0 ]; then
    echo "错误: common.cpp 编译失败"
    exit 1
fi

echo "步骤11: 编译log.cpp..."
g++ -std=c++17 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src -Iinclude -Icommon \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c common/log.cpp -o log.o

if [ $? -ne 0 ]; then
    echo "错误: log.cpp编译失败"
    exit 1
fi

echo "步骤12: 编译profiler.cpp..."
g++ -std=c++17 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src -Iinclude -Icommon \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c common/profiler.cpp -o profiler.o

if [ $? -ne 0 ]; then
    echo "错误: profiler.cpp编译失败"
    exit 1
fi

echo "步骤13: 编译llama-grammar.cpp..."
g++ -std=c++17 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src -Iinclude -Icommon \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c src/llama-grammar.cpp -o llama-grammar.o

if [ $? -ne 0 ]; then
    echo "错误: llama-grammar.cpp编译失败"
    exit 1
fi

echo "步骤14: 编译llama-sampling.cpp..."
g++ -std=c++17 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src -Iinclude -Icommon \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c src/llama-sampling.cpp -o llama-sampling.o

if [ $? -ne 0 ]; then
    echo "错误: llama-sampling.cpp编译失败"
    exit 1
fi

echo "步骤15: 生成并编译build-info.cpp..."

# 生成build-info.cpp文件
echo "正在生成build-info.cpp..."
./scripts/build-info.sh g++ > build-info.cpp

if [ $? -ne 0 ]; then
    echo "错误: build-info.cpp生成失败"
    exit 1
fi

echo "正在编译build-info.cpp..."
g++ -std=c++17 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src -Iinclude -Icommon \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c build-info.cpp -o build-info.o

if [ $? -ne 0 ]; then
    echo "错误: build-info.cpp编译失败"
    exit 1
fi

echo "步骤16: 编译llama.cpp主程序..."
g++ -std=c++17 -fPIC -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src -Iinclude -Icommon \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -pthread -fopenmp \
    -c src/llama.cpp -o llama.o

if [ $? -ne 0 ]; then
    echo "错误: llama.cpp 编译失败"
    exit 1
fi


echo "步骤17: 链接所有组件生成可执行文件..."

# 编译VOLTAGE主程序（独立版本，包含完整的GGML支持和量化函数，并链接llama.cpp库）
g++ -std=c++17 -O3 -g -Wall -Wextra -Wpedantic \
    -Iggml/include -Iggml/src -Iinclude -Icommon \
    -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG \
    -DGGML_USE_OPENMP \
    -DVOLTAGE_STANDALONE_BUILD \
    -pthread -fopenmp \
    voltage_prima_complete.cpp llama.o llama_vocab.o unicode.o unicode_data.o common.o \
    log.o profiler.o llama-grammar.o llama-sampling.o build-info.o \
    ggml_core.o ggml_quants.o ggml_backend.o ggml_alloc.o ggml_aarch64.o sgemm.o \
    -lzmq -lz \
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
echo "注意: 这是独立版本，包含完整的GGML支持和量化函数"
echo "      专注于展示VOLTAGE算法与GGML的集成"
