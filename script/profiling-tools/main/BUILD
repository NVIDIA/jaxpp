# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "@tsl//tsl/platform:rules_cc.bzl",
    "cc_binary",
)

load("@tsl//tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

load("@tsl//tsl:tsl.bzl", "set_external_visibility")

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    copts = tf_profiler_copts(),
    visibility = set_external_visibility([
        "@tsl//tsl/profiler:internal",
    ]),
    deps = [
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/protobuf:trace_events_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@tsl//tsl/profiler/utils:tf_xplane_visitor",
        "@tsl//tsl/profiler/utils:trace_utils",
        "@tsl//tsl/profiler/utils:xplane_schema",
        "@tsl//tsl/profiler/utils:xplane_utils",
        "@tsl//tsl/profiler/utils:xplane_visitor",
        "@tsl//tsl/profiler/rpc/client:capture_profile",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)
