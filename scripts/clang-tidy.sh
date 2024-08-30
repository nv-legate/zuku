#! /usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

nproc=${1:-16}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

$SCRIPT_DIR/get-clang-tidy-files \
  | xargs -P ${nproc} -I % sh -c 'echo % ; clang-tidy-17 -p compile_commands.json %'

if [ $? -ne 0 ]; then
    echo "At least one clang-tidy command failed"
    exit 1
fi

