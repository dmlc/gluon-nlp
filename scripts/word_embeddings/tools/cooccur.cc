//  Tool to calculate word-word cooccurrence statistics
//
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// * Includes and definitions
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "CLI/CLI.hpp"    // command line parser
#include "cnpy.h"         // numpy
#include "sparsepp/spp.h" // fast sparse hash map

#include "utils.h"

namespace fs = std::filesystem;
using Vocab = spp::sparse_hash_map<std::string, std::pair<uint32_t, uint32_t>>;
using count_type = float;
using Matrix = spp::sparse_hash_map<uint64_t, count_type>;

// * Arguments
enum class ContextWeight { Harmonic, DistanceOverSize, None };

std::istream &operator>>(std::istream &in, ContextWeight &context_weight) {
  int i;
  in >> i;
  context_weight = static_cast<ContextWeight>(i);
  return in;
}

std::ostream &operator<<(std::ostream &in,
                         const ContextWeight &context_weight) {
  return in << static_cast<int>(context_weight);
}

// Arguments specified via command line options. See ParseArgs for
// documentation.
struct Arguments {
  unsigned int num_threads = 1;
  unsigned int window_size = 15;
  bool no_symmetric = false;
  bool subsample = false;
  ContextWeight context_weight;
};

auto ParseArgs(int argc, char **argv) {
  // Performance optimizations for writing to stdout
  std::ios::sync_with_stdio(false);

  Arguments args;
  CLI::App app("Simple tool to calculate word-word cooccurrence statistics");
  std::vector<fs::path> files;
  app.add_option("FILES", files, "File names")->check(CLI::ExistingPath);
  std::string output = "cooccurrences.npz";
  app.add_option("-o,--output", output,
                 "Output file name. Co-occurence matrix is saved as "
                 "scipy.sparse compatible CSR matrix in a numpy .npz archive");
  app.add_option("-w,--window-size", args.window_size,
                 "Window size in which to count co-occurences.");
  app.add_flag("--no-symmetric", args.no_symmetric,
               "If not specified, a symmetric context window is used.");
  app.add_flag("--subsample", args.subsample,
               "Apply subsampling during co-occurence matrix construction as "
               "in Word2Vec .");
  app.add_set("-c,--context-weights", args.context_weight,
              {ContextWeight::Harmonic, ContextWeight::DistanceOverSize,
               ContextWeight::None},
              "Weighting scheme for contexts.")
      ->type_name("ContextWeight in {Harmonic=0, DistanceOverSize=1, None=2}");
  app.add_option(
         "-j,--numThreads", args.num_threads,
         "Number of threads to use. Each thread constructs an "
         "independent vocabulary which are finally merged. Only appropriate "
         "when multiple, sufficiently large input files are specified.")
      ->check(CLI::Range(1U, std::numeric_limits<unsigned int>::max()));

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    std::exit(app.exit(e));
  }

  std::queue<fs::path> paths;
  for (auto &file : files) {
    paths.emplace(file);
  }

  return std::make_tuple(paths, output, args);
}

// * Input
auto ReadVocab() {
  std::string word;
  std::string count;
  int rank{0};
  Vocab vocab;
  while (std::cin >> word) {
    std::cin >> count;
    vocab[word] = {rank, std::stoi(count)};
    rank++;
  }
  return vocab;
}
// * Co-occurence matrix construction
std::mutex paths_m;
std::mutex matrices_m;

void ReadMatrix(std::queue<fs::path> &paths, queue<Matrix> &matrices,
                const Vocab &vocab, const Arguments &args, uint32_t seed) {
  assert(seed > 0);
  std::string line;
  std::deque<uint32_t> history;
  std::unique_ptr<Matrix> m = std::make_unique<Matrix>();

  // Prepare subsampling
  std::random_device r;
  std::default_random_engine random_engine(r());
  std::uniform_real_distribution<float> uniform_dist(0, 1);
  std::vector<double> idx_to_pdiscard;
  if (args.subsample) {
    double sum_counts = std::accumulate(vocab.begin(), vocab.end(), 0,
                                        [](const auto &sum, const auto &e) {
                                          const auto count = e.second.second;
                                          return sum + count;
                                        });
    double t = 1E-4;
    for (const auto &e : vocab) {
      const auto count = e.second.second;
      idx_to_pdiscard.push_back(1 - std::sqrt(t / (count / sum_counts)));
    }
  }

  while (true) {
    fs::path path;
    {
      std::scoped_lock lock(paths_m);
      if (paths.empty()) {
        break;
      }
      path = paths.front();
      paths.pop();
    }

    std::ifstream in{path};
    if (!in.is_open()) {
      throw std::invalid_argument(path.string() + " cannot be opened!");
    }
    while (std::getline(in, line)) {
      history.clear(); // Discard context from other lines
      std::stringstream stream(line);
      std::string word;
      while (stream >> word) {
        // TODO We must construct an extra std::string for every word due to
        // missing support for heterogenous lookup in unordered map. Once
        // https://wg21.link/P0919 is merged construct a string_view instead.
        // std::string_view(&*word.begin(), ranges::distance(word))
        auto word_rank_it = vocab.find(word);
        // Skip words not contained in the vocabulary
        if (word_rank_it != vocab.end()) {
          uint32_t word_rank = word_rank_it->second.first;

          if (args.subsample &&
              uniform_dist(random_engine) <= idx_to_pdiscard[word_rank]) {
            continue;
          }

          for (unsigned int distance = 1; distance <= history.size();
               distance++) {
            const auto &context_word_rank = history[distance - 1];
            uint64_t key; // We merge 32 bit row and col indices to a single 64
                          // bit key
            // For symmetric contexts, only store one direction.
            if (!args.no_symmetric) {
              if (word_rank <= context_word_rank) {
                key = (static_cast<uint64_t>(word_rank) << 32) |
                      context_word_rank;
              } else {
                key = word_rank |
                      (static_cast<uint64_t>(context_word_rank) << 32);
              }
            } else {
              key =
                  (static_cast<uint64_t>(word_rank) << 32) | context_word_rank;
            }

            if (args.context_weight == ContextWeight::Harmonic) {
              (*m)[key] += 1.0f / static_cast<count_type>(distance);
            } else if (args.context_weight == ContextWeight::DistanceOverSize) {
              (*m)[key] += (args.window_size - distance - 1) / args.window_size;
            } else {
              (*m)[key]++;
            }
          }

          // Update circular history buffer
          if (history.size() == args.window_size) {
            history.pop_front();
          }
          history.push_back(word_rank);
        }
      }
    }
  }
  {
    std::scoped_lock lock(matrices_m);
    matrices.push(std::move(m));
  }
}

std::unique_ptr<Matrix> CombineMatrices(queue<Matrix> &matrices,
                                        int num_threads) {
  std::unique_ptr<Matrix> m1 = matrices.pop();
  for (int i = 1; i < num_threads; i++) {
    std::unique_ptr<Matrix> m2 = matrices.pop();
    if (m1->size() < m2->size()) {
      for (const auto &e : *m1) {
        (*m2)[e.first] += e.second;
      }
      std::swap(m1, m2);
    } else {
      for (const auto &e : *m2) {
        (*m1)[e.first] += e.second;
      }
    }
  }
  return m1;
}

auto ComputeCooccurrenceMatrix(Vocab &vocab, std::queue<fs::path> &paths,
                               const Arguments &args) {
  std::vector<std::thread> threads;
  queue<Matrix> matrices;
  for (unsigned int i = 0; i < args.num_threads; i++) {
    threads.push_back(std::thread([&paths, &matrices, &vocab, &args, i]() {
      ReadMatrix(std::ref(paths), std::ref(matrices), std::ref(vocab),
                 std::ref(args), i + 1);
    }));
  }
  std::unique_ptr<Matrix> m = CombineMatrices(matrices, args.num_threads);
  for (unsigned int i = 0; i < args.num_threads; i++) {
    threads[i].join();
  }
  return m;
}

auto ToCOO(const Vocab &vocab, std::unique_ptr<Matrix> m) {
  size_t num_tokens = vocab.size();
  size_t nnz = m->size();
  std::cout << "Got " << nnz
            << " non-zero entries in cooccurrence matrix of shape ("
            << num_tokens << ", " << num_tokens << ")" << std::endl;
  std::vector<uint32_t> row;
  std::vector<uint32_t> col;
  std::vector<count_type> data;
  row.reserve(nnz);
  col.reserve(nnz);
  data.reserve(nnz);
  for (const auto &e : *m) {
    row.push_back(e.first >> 32);
    col.push_back(e.first & 0xffffffff);
    data.push_back(e.second);
  }
  return std::make_tuple(row, col, data);
}

// * Output
void WriteNumpy(const std::string output, const std::vector<uint32_t> &row,
                const std::vector<uint32_t> &col,
                const std::vector<count_type> &data, const bool symmetric,
                const uint32_t num_tokens) {

  assert(row.size() == data.size());
  assert(col.size() == data.size());
  cnpy::npz_save(output, "row", &row[0], {row.size()}, "w");
  cnpy::npz_save(output, "col", &col[0], {col.size()}, "a");
  cnpy::npz_save(output, "data", &data[0], {data.size()}, "a");
  cnpy::npz_save(output, "num_tokens", &num_tokens, {1}, "a");
  cnpy::npz_save(output, "symmetric", &symmetric, {1}, "a");
}

// * Main
int main(int argc, char **argv) {
  auto [paths, output, args] = ParseArgs(argc, argv);
  auto vocab = ReadVocab();
  auto cooccurenceMatrix = ComputeCooccurrenceMatrix(vocab, paths, args);
  auto [row, col, data] = ToCOO(vocab, std::move(cooccurenceMatrix));
  WriteNumpy(output, row, col, data, !args.no_symmetric, vocab.size());
  return 0;
}
