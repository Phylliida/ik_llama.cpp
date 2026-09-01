{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  packages = with pkgs; [
    cmake
    ninja
    pkg-config
    git
    ccache

    # Vulkan build deps
    vulkan-headers
    vulkan-loader
    shaderc # provides glslc for compiling the Vulkan shaders
    glslang
    spirv-tools
  ];

  # Make sure the loader can find system ICDs (NVIDIA) on NixOS
  shellHook = ''
    export VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json
    export VK_DRIVER_FILES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json
    export LD_LIBRARY_PATH=/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}
    # Home is read-only under the harness sandbox; keep ccache in the workspace
    export CCACHE_DIR=/home/bepis/prog/llm-tests/.ccache
    # ik_llama's IQK kernels need real SIMD flags; allow -march=native through the Nix wrapper
    export NIX_ENFORCE_NO_NATIVE=0
  '';
}
