/opt/homebrew/opt/llvm/bin/clang++ -march=native -ffast-math -fopenmp -ftree-vectorize -fslp-vectorize ../cpu/gemm_cpu.cpp -L/opt/homebrew/opt/llvm/lib -o mp1_cpu
for i in {1..20}; do
    echo "Run $i:"
    ./mp1_cpu 100 100 100
done