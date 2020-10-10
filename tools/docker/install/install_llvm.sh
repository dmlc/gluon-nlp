set -euo pipefail

wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 8  # Fix version
