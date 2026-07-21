{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  nlohmann_json,
  which,
  python3,
}:

let
  # ZenDNN version info matching ggml/src/ggml-zendnn/CMakeLists.txt
  zendnnVersion = "5.2.2";
  zendnnRev = "253b94ce0d7e9284c265fefb485714944caff9d3";

  # Dependency versions from ZenDNN's ZenDnnlDependenciesInfo.cmake
  aocldlpRev = "59b42ab1ae5e977ace905e27305fc775e6bb6737";
  aoclutilsRev = "5.0";
  onednnRev = "v3.12";
  libxsmmRev = "eedaa03d49a1dffe6048711598bc5a4da5a86008";

  # Fetch all dependency sources at evaluation time.
  aocldlpSrc = fetchFromGitHub {
    owner = "amd";
    repo = "aocl-dlp";
    rev = aocldlpRev;
    hash = "sha256-5uQaL8Ftm/3zw+j0mOrEzq58S6i/830VRnAmS42ZwjA=";
  };

  aoclutilsSrc = fetchFromGitHub {
    owner = "amd";
    repo = "aocl-utils";
    rev = aoclutilsRev;
    hash = "sha256-96j3Sw+Ts+CZzjPpUlt8cRYO5z0iASo+W/x1nrrAyQE=";
  };

  onednnSrc = fetchFromGitHub {
    owner = "oneapi-src";
    repo = "oneDNN";
    rev = onednnRev;
    hash = "sha256-t5+DF4/qgEYQpTY8Qox0BTfpykfs5kFqYy6HrEJaVu0=";
  };

  libxsmmSrc = fetchFromGitHub {
    owner = "libxsmm";
    repo = "libxsmm";
    rev = libxsmmRev;
    hash = "sha256-qm9/SqRH4AoZE0Y6YQyGi/7SSyvz9vytrkxZ8+L4z+4=";
  };
in

stdenv.mkDerivation {
  pname = "zendnn";
  version = zendnnVersion;

  src = fetchFromGitHub {
    owner = "amd";
    repo = "ZenDNN";
    rev = zendnnRev;
    hash = "sha256-dhNPxXSK+zPOXf/NjGKAAJUBgb0sXX3/uYUI5JMWKGk=";
  };

  nativeBuildInputs = [
    cmake
    which
    python3
  ];

  # Pre-populate the dependencies directory with pre-fetched sources
  # so that ExternalProject uses them instead of attempting git clone.
  # nlohmann_json is header-only and provided by nixpkgs, so we give
  # ZenDNN a minimal CMakeLists.txt that installs the nixpkgs headers.
  preConfigure = ''
    mkdir -p dependencies/json/include

    cat > dependencies/json/CMakeLists.txt << 'CMEOF'
cmake_minimum_required(VERSION 3.14)
project(nlohmann_json LANGUAGES CXX)
install(DIRECTORY ''${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
  $<BUILD_INTERFACE:''${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
install(TARGETS nlohmann_json EXPORT nlohmann_jsonTargets)
install(EXPORT nlohmann_jsonTargets
  NAMESPACE nlohmann_json::
  DESTINATION share/cmake/nlohmann_json)
file(WRITE ''${CMAKE_CURRENT_BINARY_DIR}/nlohmann_jsonConfig.cmake
  "include(\"\''${CMAKE_CURRENT_LIST_DIR}/nlohmann_jsonTargets.cmake\")\n")
install(FILES ''${CMAKE_CURRENT_BINARY_DIR}/nlohmann_jsonConfig.cmake
  DESTINATION share/cmake/nlohmann_json)
CMEOF

    cp -r ${nlohmann_json}/include/nlohmann dependencies/json/include/

    cp -r ${aocldlpSrc} dependencies/aocldlp
    cp -r ${aoclutilsSrc} dependencies/aoclutils
    cp -r ${onednnSrc} dependencies/onednn
    cp -r ${libxsmmSrc} dependencies/libxsmm

    chmod -R u+w dependencies/

    # Fix shebangs in libxsmm helper scripts
    # (libxsmm's old Makefile uses #!/usr/bin/env which doesn't exist in Nix sandbox)
    patchShebangs dependencies/libxsmm

    # libxsmm's Makefile.inc tries to locate tools (gsed) using `which`,
    # which may not be available. Replace fragile probe with direct name.
    # Also stub out the version() arithmetic function - it chokes on
    # non-numeric inputs from compiler name detection (Nix store paths).
    sed -i \
      -e '/^SED :=.*which.*gsed/c\SED := sed' \
      -e '/^  SED :=.*which.*sed/c\  SED := sed' \
      -e '/^version =/,/0))))")/{
            /^version =/c\version = 0
            /^version =/!d
          }' \
      dependencies/libxsmm/Makefile.inc
    export CXXFLAGS="-mavx512fp16 $CXXFLAGS"
    export CFLAGS="-mavx512fp16 $CFLAGS"

    # ZenDNN assumes GCC >= 14 has _mm512_maskz_loadu_ph/_mm512_mask_storeu_ph
    # but GCC 14 actually doesn't yet. Use the epi16 fallback path.
    substituteInPlace zendnnl/src/common/float16.hpp \
      --replace-fail '(__GNUC__ < 14)' '1'
  '';

  # Match the configuration used by ggml/src/ggml-zendnn/CMakeLists.txt
  cmakeFlags = [
    "-DZENDNNL_BUILD_EXAMPLES=OFF"
    "-DZENDNNL_BUILD_DOXYGEN=OFF"
    "-DZENDNNL_BUILD_GTEST=OFF"
    "-DZENDNNL_BUILD_BENCHDNN=OFF"
    "-DZENDNNL_DEPENDS_FBGEMM=OFF"
    "-DZENDNNL_LIB_BUILD_ARCHIVE=ON"
    "-DZENDNNL_LIB_BUILD_SHARED=OFF"
    "-DZENDNNL_DEPENDS_AOCLDLP=ON"
    "-DZENDNNL_DEPENDS_ONEDNN=ON"
    "-DZENDNNL_DEPENDS_LIBXSMM=ON"
    # Use locally placed sources, skip git downloads
    "-DZENDNNL_LOCAL_AOCLDLP=ON"
    "-DZENDNNL_LOCAL_AOCLUTILS=ON"
    "-DZENDNNL_LOCAL_JSON=ON"
    "-DZENDNNL_LOCAL_ONEDNN=ON"
    "-DZENDNNL_LOCAL_LIBXSMM=ON"
  ];

  # The ZenDNN build system uses ExternalProject for the library itself too.
  # Running cmake --build with the zendnnl target handles configure+build+install
  # for both dependencies and the library. The default target does not include
  # ExternalProject targets, so we must specify zendnnl explicitly.
  enableParallelBuilding = true;

  buildPhase = ''
    runHook preBuild
    cmake --build . --target zendnnl --parallel $NIX_BUILD_CORES
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out
    cp -r install/zendnnl $out/
    cp -r install/deps $out/
    # Remove shared libraries and binaries; we only need static libs for linking.
    # This also avoids fixupPhase RPATH issues from build-directory references.
    find $out \( -name '*.so' -o -name '*.so.*' \) -print0 | xargs -0 -r rm -f
    find $out/deps -name bin -type d -print0 | xargs -0 -r rm -rf
    # Remove broken symlinks (libxsmm creates dangling share symlinks)
    find -L $out -type l -delete
    # Remove empty share/doc directories we don't need
    find $out/deps -name share -empty -type d -delete
    runHook postInstall
  '';

  meta = {
    description = "Accelerated Deep Learning Inference on AMD Zen Architecture";
    homepage = "https://github.com/amd/ZenDNN";
    license = lib.licenses.asl20;
    # ZenDNN is specific to AMD Zen architecture (x86_64-linux)
    platforms = [ "x86_64-linux" ];
    maintainers = with lib.maintainers; [ ];
  };
}
