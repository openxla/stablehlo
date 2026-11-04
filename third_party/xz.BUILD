# Copyright 2026 The StableHLO Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:private"])

licenses(["unencumbered"])

exports_files(["COPYING"])

cc_library(
    name = "lzma",
    srcs = glob(
        include = [
            "src/**/*.h",
            "src/liblzma/**/*.c",
        ],
        exclude = [
            "src/**/*_tablegen.c",
            "src/liblzma/check/crc_clmul_consts_gen.c",
            "src/liblzma/check/sha256.c",
            "src/liblzma/check/crc*_small.c",
            "src/liblzma/common/stream_decoder_mt.c",
            "src/liblzma/common/stream_encoder_mt.c",
        ],
    ) + select({
        "@platforms//os:osx": [],
        "@platforms//os:ios": [],
        "//conditions:default": [
            "src/liblzma/check/sha256.c",
        ],
    }) + select({
        "@platforms//os:none": [],
        "//conditions:default": [
            "src/common/tuklib_cpucores.c",
            "src/common/tuklib_physmem.c",
            "src/liblzma/common/stream_decoder_mt.c",
            "src/liblzma/common/stream_encoder_mt.c",
        ],
    }),
    hdrs = glob(["src/liblzma/api/**/*.h"]),
    copts = select({
        "@platforms//os:windows": [],
        "//conditions:default": ["-std=c99"],
    }),
    defines = select({
        "@platforms//os:windows": ["LZMA_API_STATIC"],
        "//conditions:default": [],
    }),
    implementation_deps = [
        ":config",
        ":src_common",
        ":src_liblzma",
        ":src_liblzma_check",
        ":src_liblzma_common",
        ":src_liblzma_delta",
        ":src_liblzma_lz",
        ":src_liblzma_lzma",
        ":src_liblzma_rangecoder",
        ":src_liblzma_simple",
    ],
    linkopts = select({
        "@platforms//os:android": [],
        "@platforms//os:windows": [],
        "@platforms//os:none": [],
        "//conditions:default": ["-lpthread"],
    }),
    linkstatic = select({
        "@platforms//os:windows": True,
        "//conditions:default": False,
    }),
    local_defines = ["HAVE_CONFIG_H"],
    strip_include_prefix = "src/liblzma/api",
    visibility = ["//visibility:public"],
)

write_file(
    name = "config_gen",
    out = "lzma_config/config.h",
    content = [
        "#define ASSUME_RAM 128",
        "#ifndef _WIN32",
        "#define HAVE_BSWAP_16 1",
        "#define HAVE_BSWAP_32 1",
        "#define HAVE_BSWAP_64 1",
        "#endif",
        "#if defined(__APPLE__) || defined(_WIN32)",
        "#undef HAVE_BYTESWAP_H",
        "#else",
        "#define HAVE_BYTESWAP_H 1",
        "#endif",
        "#undef HAVE_CAPSICUM",
        "#if defined(__APPLE__)",
        "#define HAVE_CC_SHA256_CTX 1",
        "#define HAVE_CC_SHA256_INIT 1",
        "#define HAVE_COMMONCRYPTO_COMMONDIGEST_H 1",
        "#endif",
        "#define HAVE_CHECK_CRC32 1",
        "#define HAVE_CHECK_CRC64 1",
        "#define HAVE_CHECK_SHA256 1",
        "#if !defined(__APPLE__)",
        "#define HAVE_CLOCK_GETTIME 1",
        "#define HAVE_DECL_CLOCK_MONOTONIC 1",
        "#endif",
        "#if defined(__ANDROID__) || defined(__APPLE__) || defined(_WIN32)",
        "#define HAVE_DECL_PROGRAM_INVOCATION_NAME 0",
        "#else",
        "#define HAVE_DECL_PROGRAM_INVOCATION_NAME 1",
        "#endif",
        "#define HAVE_DECODERS 1",
        "#define HAVE_DECODER_ARM 1",
        "#define HAVE_DECODER_ARM64 1",
        "#define HAVE_DECODER_ARMTHUMB 1",
        "#define HAVE_DECODER_DELTA 1",
        "#define HAVE_DECODER_IA64 1",
        "#define HAVE_DECODER_LZMA1 1",
        "#define HAVE_DECODER_LZMA2 1",
        "#define HAVE_DECODER_POWERPC 1",
        "#define HAVE_DECODER_SPARC 1",
        "#define HAVE_DECODER_X86 1",
        "#define HAVE_DLFCN_H 1",
        "#define HAVE_ENCODERS 1",
        "#define HAVE_ENCODER_ARM 1",
        "#define HAVE_ENCODER_ARM64 1",
        "#define HAVE_ENCODER_ARMTHUMB 1",
        "#define HAVE_ENCODER_DELTA 1",
        "#define HAVE_ENCODER_IA64 1",
        "#define HAVE_ENCODER_LZMA1 1",
        "#define HAVE_ENCODER_LZMA2 1",
        "#define HAVE_ENCODER_POWERPC 1",
        "#define HAVE_ENCODER_SPARC 1",
        "#define HAVE_ENCODER_X86 1",
        "#define HAVE_FCNTL_H 1",
        "#define HAVE_FUTIMENS 1",
        "#define HAVE_GETOPT_H 1",
        "#define HAVE_GETOPT_LONG 1",
        "#define HAVE_INTTYPES_H 1",
        "#define HAVE_LIMITS_H 1",
        "#define HAVE_MBRTOWC 1",
        "#define HAVE_MEMORY_H 1",
        "#define HAVE_MF_BT2 1",
        "#define HAVE_MF_BT3 1",
        "#define HAVE_MF_BT4 1",
        "#define HAVE_MF_HC3 1",
        "#define HAVE_MF_HC4 1",
        "#if !defined(__APPLE__)",
        "#define HAVE_POSIX_FADVISE 1",
        "#endif",
        "#if !defined(__APPLE__) && !defined(__ANDROID__)",
        "#define HAVE_PTHREAD_CONDATTR_SETCLOCK 1",
        "#endif",
        "#define HAVE_PTHREAD_PRIO_INHERIT 1",
        "#undef HAVE_SMALL",
        "#define HAVE_STDBOOL_H 1",
        "#define HAVE_STDINT_H 1",
        "#define HAVE_STDLIB_H 1",
        "#define HAVE_STRING_H 1",
        "#define HAVE_STRUCT_STAT_ST_ATIM_TV_NSEC 1",
        "#undef HAVE_SYS_CAPSICUM_H",
        "#ifndef _WIN32",
        "#define HAVE_SYS_PARAM_H 1",
        "#define HAVE_SYS_STAT_H 1",
        "#define HAVE_SYS_TIME_H 1",
        "#define HAVE_SYS_TYPES_H 1",
        "#endif",
        "#define HAVE_UINTPTR_T 1",
        "#define HAVE_UNISTD_H 1",
        "#define HAVE_VISIBILITY 0",
        "#ifndef _WIN32",
        "#define HAVE_WCWIDTH 1",
        "#endif",
        "#define HAVE__BOOL 1",
        "#undef HAVE__FUTIME",
        "#define LT_OBJDIR \".libs/\"",
        "#ifdef _WIN32",
        "#define MYTHREAD_VISTA 1",
        "#else",
        "#define MYTHREAD_POSIX 1",
        "#endif",
        "#define NDEBUG 1",
        "#define PACKAGE \"xz\"",
        "#define PACKAGE_BUGREPORT \"lasse.collin@tukaani.org\"",
        "#define PACKAGE_NAME \"XZ Utils\"",
        "#define PACKAGE_STRING \"XZ Utils 5.8.3\"",
        "#define PACKAGE_TARNAME \"xz\"",
        "#define PACKAGE_URL \"https://tukaani.org/xz/\"",
        "#define PACKAGE_VERSION \"5.8.3\"",
        "#define SIZEOF_SIZE_T 8",
        "#define STDC_HEADERS 1",
        "#define TUKLIB_CPUCORES_SYSCONF 1",
        "#define TUKLIB_FAST_UNALIGNED_ACCESS 1",
        "#if defined(__APPLE__)",
        "#undef TUKLIB_PHYSMEM_SYSCONF",
        "#else",
        "#define TUKLIB_PHYSMEM_SYSCONF 1",
        "#endif",
        "#ifndef _ALL_SOURCE",
        "#define _ALL_SOURCE 1",
        "#endif",
        "#ifndef _GNU_SOURCE",
        "#define _GNU_SOURCE 1",
        "#endif",
        "#ifndef _POSIX_PTHREAD_SEMANTICS",
        "#define _POSIX_PTHREAD_SEMANTICS 1",
        "#endif",
        "#ifndef _TANDEM_SOURCE",
        "#define _TANDEM_SOURCE 1",
        "#endif",
        "#ifndef __EXTENSIONS__",
        "#define __EXTENSIONS__ 1",
        "#endif",
        "#define VERSION \"5.8.3\"",
        "#if defined AC_APPLE_UNIVERSAL_BUILD",
        "#if defined __BIG_ENDIAN__",
        "#define WORDS_BIGENDIAN 1",
        "#endif",
        "#endif",
        "#ifndef _DARWIN_USE_64_BIT_INODE",
        "#define _DARWIN_USE_64_BIT_INODE 1",
        "#endif",
        "",
    ],
)

cc_library(
    name = "config",
    hdrs = ["lzma_config/config.h"],
    strip_include_prefix = "lzma_config",
)

cc_library(
    name = "src_common",
    srcs = select({
        "@platforms//os:none": [],
        "//conditions:default": [
            "src/common/tuklib_exit.c",
            "src/common/tuklib_progname.c",
        ],
    }),
    hdrs = glob(["src/common/*.h"]),
    defines = select({
        "@platforms//os:windows": ["LZMA_API_STATIC"],
        "//conditions:default": [],
    }),
    local_defines = ["HAVE_CONFIG_H"] + select({
        "@platforms//os:windows": ["TUKLIB_GETTEXT=0"],
        "//conditions:default": [],
    }),
    strip_include_prefix = "src/common",
    deps = [
        ":config",
        ":src_liblzma",
        ":src_liblzma_api",
        ":src_liblzma_common",
    ],
)

cc_library(
    name = "src_liblzma",
    hdrs = glob([
        "src/liblzma/common/*.h",
        "src/liblzma/lzma/*.h",
    ]),
    strip_include_prefix = "src/liblzma",
)

cc_library(
    name = "src_liblzma_api",
    hdrs = glob(
        include = ["src/liblzma/api/**/*.h"],
        exclude = ["src/liblzma/api/lzma.h"],
    ),
    strip_include_prefix = "src/liblzma/api",
)

cc_library(
    name = "src_liblzma_check",
    hdrs = glob(["src/liblzma/check/*.h"]),
    strip_include_prefix = "src/liblzma/check",
)

cc_library(
    name = "src_liblzma_common",
    hdrs = glob(["src/liblzma/common/*.h"]),
    strip_include_prefix = "src/liblzma/common",
)

cc_library(
    name = "src_liblzma_delta",
    hdrs = glob(["src/liblzma/delta/*.h"]),
    strip_include_prefix = "src/liblzma/delta",
)

cc_library(
    name = "src_liblzma_lz",
    hdrs = glob(["src/liblzma/lz/*.h"]),
    strip_include_prefix = "src/liblzma/lz",
)

cc_library(
    name = "src_liblzma_lzma",
    hdrs = glob(["src/liblzma/lzma/*.h"]),
    strip_include_prefix = "src/liblzma/lzma",
)

cc_library(
    name = "src_liblzma_rangecoder",
    hdrs = glob(["src/liblzma/rangecoder/*.h"]),
    strip_include_prefix = "src/liblzma/rangecoder",
)

cc_library(
    name = "src_liblzma_simple",
    hdrs = glob(["src/liblzma/simple/*.h"]),
    strip_include_prefix = "src/liblzma/simple",
)
