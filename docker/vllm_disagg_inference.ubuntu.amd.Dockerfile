ARG BASE_IMAGE=vllm/vllm-openai-rocm:nightly-c133f3374625652c88e122fff995e4126c4635c0
#ARG BASE_IMAGE=rocm/vllm-dev:base_torch2.10_triton3.6_rocm7.2_torch_build_20260216
FROM ${BASE_IMAGE}

ENTRYPOINT []

WORKDIR /root

RUN sed -i 's/http/https/g' /etc/apt/sources.list

ENV _ROCM_DIR=/opt/rocm

ENV _UCX_SOURCE=https://github.com/ROCm/ucx.git
ENV _UCX_BRANCH=da3fac2a
ENV _UCX_INSTALL_DIR=/usr/local/ucx/

ENV _RIXL_SOURCE=https://github.com/ROCm/RIXL.git
ENV _RIXL_BRANCH=f33a5599
ENV _RIXL_INSTALL_DIR=/usr/local/RIXL/install
ENV _NIXLBENCH_INSTALL_DIR=/usr/local/RIXL

ARG GFX_COMPILATION_ARCH="gfx942"
ARG NIC_COMPILATION_ARCH="cx7"

RUN pip3 install meson==0.64.0
RUN pip3 install "pybind11[global]"

RUN apt-get update && apt-get install -y \
    autoconf \
    automake \
    libtool \
    autogen \
    pkg-config \
    m4 || apt --fix-broken install -y

RUN set -e && apt update

RUN set -e && apt -y install gcc make libtool autoconf librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool libibverbs-dev rdma-core strace
RUN apt install -y libgflags-dev

# Install UCX
RUN git clone ${_UCX_SOURCE} && \
    cd ucx && \
    git checkout ${_UCX_BRANCH} && \
    ./autogen.sh && \
    mkdir -p build && \
    cd build && \
    ../configure --prefix=${_UCX_INSTALL_DIR} --with-rocm=${_ROCM_DIR} --disable-go --disable-java --disable-assertions --enable-mt && \
    make -j && \
    make install && \
    echo "UCX installation completed."


ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/ucx/lib/
ENV PATH=$PATH:/usr/local/ucx/bin/

RUN set -e && apt update && \
    apt install -y libaio-dev liburing-dev etcd etcd-server etcd-client libcpprest-dev libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc wget && \
    wget https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz && \
    tar -xzf v1.14.0.tar.gz && \
    cd googletest-1.14.0 && \
    mkdir -p build && \
    cd build && \
    cmake -DBUILD_SHARED_LIBS=on .. && \
    make -j && \
    make install && \
    cd ../..

# Expected etcd at /usr/local/bin/etcd//etcd
RUN wget https://github.com/etcd-io/etcd/releases/download/v3.6.0-rc.5/etcd-v3.6.0-rc.5-linux-amd64.tar.gz -O /tmp/etcd.tar.gz && \
    mkdir -p /usr/local/bin/etcd && \
    tar -xvf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1 && \
    rm /tmp/etcd.tar.gz
ENV PATH=$PATH:/usr/local/bin/etcd/

RUN set -e && echo "Compiling etcd-cpp API" && \
    git clone https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git && \
    cd etcd-cpp-apiv3 && \
    mkdir build && cd build && \
    cmake -DCMAKE_FIND_ROOT_PATH=/usr/grpc .. && \
    make -j && \
    make install && \
    cd ../.. && \
    echo "etcd-cpp installation completed."

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
#ENV CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/usr/local/lib/cmake/etcd-cpp-api/
ENV PATH=/root/.local/bin:${_UCX_INSTALL_DIR}/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${_RIXL_INSTALL_DIR}/lib/x86_64-linux-gnu
#ENV CMAKE_PREFIX_PATH=/usr/local/lib/cmake/etcd-cpp-api/:/usr/grpc/lib/cmake/:/usr/local/lib/cmake

#git checkout ed772c8d0d8a47c7b4e1a622b13c4f6087a4972a && \
RUN set -e && git clone ${_RIXL_SOURCE} && \
    cd RIXL && \
    git checkout ${_RIXL_BRANCH} && \
    meson setup build/ --prefix=${_RIXL_INSTALL_DIR} \
        -Ducx_path=${_UCX_INSTALL_DIR} \
        -Ddisable_gds_backend=true \
        -Dcudapath_inc=${_ROCM_DIR}/include \
        -Dcudapath_lib=${_ROCM_DIR}/lib && \
    cd build && \
    ninja && \
    ninja install

RUN set -e && cd RIXL && \
    pip install --config-settings=setup-args="-Dcudapath_inc=${_ROCM_DIR}/include" \
                --config-settings=setup-args="-Dcudapath_lib=${_ROCM_DIR}/lib" \
                --config-settings=setup-args="-Ducx_path=${_UCX_INSTALL_DIR}" \
                --config-settings=setup-args="-Ddisable_gds_backend=true" .

ENV LD_LIBRARY_PATH=${_RIXL_INSTALL_DIR}/lib:$LD_LIBRARY_PATH

RUN set -e && echo "Compiling NixlBench" && \
    cd RIXL/benchmark/nixlbench && \
    meson setup build \
        -Dnixl_path=${_RIXL_INSTALL_DIR} \
        -Dcudapath_inc=${_ROCM_DIR}/include \
        -Dcudapath_lib=${_ROCM_DIR}/lib \
        --prefix=${_NIXLBENCH_INSTALL_DIR} && \
    cd build && \
    ninja && \
    ninja install && \
    echo "NixlBench compilation complete"


# Install Rust compiler (required for building vllm-router)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install vllm-router
RUN pip install vllm-router

WORKDIR /app

# not installing mori since its already installed in vllm container.
RUN pip install tqdm prettytable
RUN git clone --recursive $(grep '^MORI_REPO:' versions.txt | cut -d' ' -f2) && \
    cd mori && \
    git checkout $(grep '^MORI_BRANCH:' /app/versions.txt | cut -d' ' -f2)

RUN git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git && cd rocm-systems && \
    git sparse-checkout set --cone projects/rocshmem && \
    git checkout develop

WORKDIR /app/rocm-systems/projects/rocshmem
RUN echo "ROCSHMEM_REPO=\"https://github.com/ROCm/rocm-systems.git\"" >> /app/versions.txt
RUN echo "ROCSHMEM_BRANCH=\"$(git log | head -1 | awk '{print $2}' | cut -c1-8)\"" >> /app/versions.txt
RUN mkdir -p /app/rocshmem-build
WORKDIR /app/rocshmem-build
RUN /app/rocm-systems/projects/rocshmem/scripts/build_configs/all_backends -DUSE_EXTERNAL_MPI=OFF -DGPU_TARGETS=$GFX_COMPILATION_ARCH

WORKDIR /app
RUN git clone https://github.com/ROCm/DeepEP.git
WORKDIR /app/DeepEP
RUN echo "DEEPEP_REPO=\"https://github.com/ROCm/DeepEP.git\"" >> /app/versions.txt
RUN echo "DEEPEP_BRANCH=\"$(git log | head -1 | awk '{print $2}' | cut -c1-8)\"" >> /app/versions.txt
RUN PYTORCH_ROCM_ARCH=$GFX_COMPILATION_ARCH  CFLAGS="-O3 -fPIC" CXXFLAGS="-O3 -fPIC --offload-arch=$GFX_COMPILATION_ARCH" HIP_CXX_FLAGS="-O3 -fPIC" \
    python3 setup.py --variant rocm --nic $NIC_COMPILATION_ARCH build develop

# TODO: uninstall and re-install should be removed after upstream is stable.
# Only need tests/ for toy_proxy_server.py; base image already has vLLM installed
    #pip install -r requirements/rocm.txt && \
    #pip install -r requirements/kv_connectors_rocm.txt && \
    #PYTORCH_ROCM_ARCH=$GFX_COMPILATION_ARCH python setup.py bdist_wheel --dist-dir=dir && \
    #pip install dist/*.whl && \
#RUN pip uninstall -y vllm
#RUN git clone --recursive https://github.com/vllm-project/vllm.git /tmp/vllm-src && \
#    cd /tmp/vllm-src && \
#    git checkout 7d6917bef552d6aff70142ab9fb8af648081d4db && \
#    cp -r /tmp/vllm-src/tests /app/vllm/tests

# Only need tests/ for toy_proxy_server.py; base image already has vLLM installed
RUN git clone --depth 1 https://github.com/vllm-project/vllm.git /tmp/vllm-src && \
    cp -r /tmp/vllm-src/tests /app/vllm/tests && \
    rm -rf /tmp/vllm-src

WORKDIR /app    

ENV ROCSHMEM_TEST_UUID=1 
ENV ROCSHMEM_HEAP_SIZE=6442450944

RUN echo "UCX_REPO=${_UCX_SOURCE}" >> /app/versions.txt
RUN echo "UCX_BRANCH=${_UCX_BRANCH}" >> /app/versions.txt
RUN echo "RIXL_REPO=${_RIXL_SOURCE}" >> /app/versions.txt
RUN echo "RIXL_BRANCH=${_RIXL_BRANCH}" >> /app/versions.txt

RUN cat /app/versions.txt
