// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>


#include "tsl/profiler/protobuf/xplane.pb.h"
#include "tsl/profiler/utils/tf_xplane_visitor.h"
#include "tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/rpc/client/capture_profile.h"
#include "tsl/profiler/utils/xplane_schema.h"
#include <cmath>

using tensorflow::profiler::XPlane;
using tensorflow::profiler::XSpace;
using tensorflow::profiler::XEvent;

static bool verbose = false;

XPlane *load_xplane(XSpace *xspace, int node_idx, const std::string &name) {
  auto xplane = tsl::profiler::FindMutablePlaneWithName(xspace, name);
  if (!xplane) {
    std::cerr << "Failed to find plane with name " << name << ": in node "
              << node_idx << std::endl;
    exit(1);
  }
  return xplane;
}

XPlane *load_gpu_xplane(XSpace *xspace, int gpu_idx, int node_idx) {
  // kGpuPlanePrefix -> "/device:GPU:"
  std::string gpu_name = std::string(tsl::profiler::kGpuPlanePrefix) + std::to_string(gpu_idx);
  return load_xplane(xspace, node_idx, gpu_name);
}

XPlane *process_gpu_xplane(XPlane *gpu_xplane, int path_idx, int node_idx) {
  // std::cout << gpu_xplane->DebugString() << std::endl;
  auto name = gpu_xplane->name();
  auto gpu_id = name.substr(name.rfind(":") + 1);
  gpu_xplane->set_name(std::string(tsl::profiler::kGpuPlanePrefix) + "(" +
                       std::to_string(path_idx) + "," + gpu_id + ")");
  gpu_xplane->set_id(node_idx);
  return gpu_xplane;
}

XPlane *process_cpu_xplane(XPlane *cpu_xplane) {
  // kHostThreadsPlaneName -> "/host:CPU"
  cpu_xplane->set_name(std::string(tsl::profiler::kHostThreadsPlaneName));
  return cpu_xplane;
}

void RemoveLinesWithFewEvents(XPlane *xplane, int threshold) {
  auto lines = xplane->mutable_lines();
  auto m = xplane->event_metadata();

  for (int i = 0; i < lines->size(); ++i) {
    auto line = lines->Mutable(i);
    auto events = line->mutable_events();
    absl::flat_hash_set<const XEvent *> to_remove;
    for (int e_idx = 0; e_idx < events->size(); e_idx++) {
      auto e = events->Mutable(e_idx);
      auto meta_id = e->metadata_id();
      // std::cout << m[meta_id].name() << " " << line->timestamp_ns() << ": "
      // << e->offset_ps() << std::endl;
      if ((line->timestamp_ns() * 1000 + e->offset_ps()) <
          485 * std::pow(10, 12)) {
        to_remove.insert(e);
      }
    }

    if (events->size() < threshold) {
      events->Clear();
    } else {
      std::cout << to_remove.size() << std::endl;
      tsl::profiler::RemoveEvents(line, to_remove);
    }
  }

  tsl::profiler::RemoveEmptyLines(xplane);
}

int main(int argc, char **argv) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  // TO-DO: use argparse
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << "<m_dim> <shapes> <files>" << std::endl;
    std::cerr << "  Expect at least m_dim, one shape, and one pb file" << std::endl;
    return 1;
  }
  bool parsingIntegers = true;
  std::vector<int> shapes;
  std::vector<std::string> paths;
  int m_dim = std::stoi(argv[1]); // the dimension for merging
  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    if (parsingIntegers) {
      try {
        shapes.push_back(std::stoi(arg));
      } catch (const std::invalid_argument&) {
        parsingIntegers = false;
        paths.push_back(arg);
      }
    } else {
      paths.push_back(arg);
    }
  }
  int factor = 1;
  for (int i = m_dim + 1; i < shapes.size(); i++) {
    factor *= shapes[i];
  }

  XSpace out_xspace;

  // add a dummy cpu xplane, needed by tensorboard trace_viewer
  XPlane cpu_xplane;
  *out_xspace.add_planes() = *process_cpu_xplane(&cpu_xplane);

  int path_idx = 0, gpu_idx = 0, node_idx = 0, gpu_offset = 0;
  for (int i = 0; i < shapes[m_dim]; i++) {
    std::string file = paths[path_idx];
    if (verbose) {
      std::cout << "input: path_idx = " << path_idx << ", file = " << file << std::endl;
    }

    // load xspace from source
    XSpace xspace;
    {
      std::fstream input(file, std::ios::in | std::ios::binary);
      if (!input) {
        std::cout << file << ": File not found." << std::endl;
        return 1;
      } else if (!xspace.ParseFromIstream(&input)) {
        std::cerr << "Failed to read xspace" << std::endl;
        return -1;
      }
    }

    std::cout << "[Profile] Merging profile " << node_idx
              << " from node " << path_idx << ", gpu " << gpu_idx
              << " (offset=" << gpu_offset << ")" << std::endl;
    if (verbose) {
      for (auto h : xspace.hostnames()) {
        std::cout << "hostname: " << h << std::endl;
      }

      for (auto plane : xspace.planes()) {
        std::cout << "plane: " << plane.name() << std::endl;
      }
    }

    // load gpu xplane
    XPlane *gpu_xplane = load_gpu_xplane(&xspace, gpu_idx, node_idx);
    *out_xspace.add_planes() = *process_gpu_xplane(gpu_xplane, path_idx, node_idx);

    // set hostnames
    out_xspace.add_hostnames(xspace.hostnames(0));

    node_idx++;
    gpu_offset += factor;
    gpu_idx = gpu_offset % 8;
    path_idx = gpu_offset / 8;
  }

  if (verbose) {
    std::cout << "--- output xspace ---" << std::endl;
    for (auto plane : out_xspace.planes()) {
      std::cout << "plane: " << plane.name() << std::endl;
    }
    std::cout << "--- output xspace debug string ---" << std::endl;
    std::cout << out_xspace.DebugString() << std::endl;
  }

  auto log_dir = "merged_profiles";
  setenv("TF_PROFILER_TRACE_VIEWER_MAX_EVENTS", "10000000", 1);
  std::cout << tsl::profiler::ExportToTensorBoard(out_xspace, log_dir, true)
            << std::endl;

  return 0;
}
