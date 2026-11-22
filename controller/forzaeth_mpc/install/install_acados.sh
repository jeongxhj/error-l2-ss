#!/bin/bash
# Installs acados v0.4.4, its Python interface and the tera renderer that its code
# generation needs. Afterwards set:
#   export ACADOS_SOURCE_DIR=<ACADOS_ROOT>
#   export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
set -e

# Define the acados installation path
ACADOS_ROOT=${1:-/acados}
TERA_VERSION=0.0.34

# Clone acados into the target folder
git clone --branch v0.4.4 https://github.com/acados/acados.git "$ACADOS_ROOT"
cd "$ACADOS_ROOT"
git submodule update --init --recursive

# Build and install acados (also builds the shared libraries in $ACADOS_ROOT/lib)
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON .. # Add more options as needed
make install -j4

# Install Python interface
pip install "$ACADOS_ROOT"/interfaces/acados_template

# Tera renderer: without it acados asks y/n on stdin at the first code generation,
# which crashes any non-interactive process (e.g. a ROS node)
mkdir -p "$ACADOS_ROOT/bin"
if [[ "$(uname -m)" == "x86_64" ]]; then
    wget -q -O "$ACADOS_ROOT/bin/t_renderer" \
        "https://github.com/acados/tera_renderer/releases/download/v${TERA_VERSION}/t_renderer-v${TERA_VERSION}-linux"
    chmod +x "$ACADOS_ROOT/bin/t_renderer"
elif command -v cargo > /dev/null; then
    # only an x86_64 linux binary is released; build it elsewhere
    git clone --branch "v${TERA_VERSION}" https://github.com/acados/tera_renderer.git /tmp/tera_renderer
    (cd /tmp/tera_renderer && cargo build --release)
    cp /tmp/tera_renderer/target/release/t_renderer "$ACADOS_ROOT/bin/t_renderer"
    rm -rf /tmp/tera_renderer
else
    echo "WARNING: no tera renderer for $(uname -m): install cargo and rerun, or build" \
         "https://github.com/acados/tera_renderer into $ACADOS_ROOT/bin/t_renderer" >&2
fi

echo "Acados installation completed."
echo "Set ACADOS_SOURCE_DIR=$ACADOS_ROOT and add $ACADOS_ROOT/lib to LD_LIBRARY_PATH."
