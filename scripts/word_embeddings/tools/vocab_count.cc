//  Tool to extract unigram counts
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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "CLI/CLI.hpp"    // command line parser
#include "sparsepp/spp.h" // fast sparse hash map

#include "./utils.h"

namespace fs = std::filesystem;
using Vocab = spp::sparse_hash_map<std::string, uint32_t>;

std::mutex paths_m;
std::mutex vocabs_m;

void ReadVocab(std::queue<fs::path> &paths, queue<Vocab> &vocabs) {
  std::unique_ptr<Vocab> vocab = std::make_unique<Vocab>();
  std::string word;
  fs::path path;

  while (true) {
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
    while (in >> word) {
      (*vocab)[word]++;
    }
  }
  {
    std::scoped_lock lock(vocabs_m);
    vocabs.push(std::move(vocab));
  }
}

std::unique_ptr<Vocab> CombineVocabs(queue<Vocab> &vocabs, int num_threads) {
  std::unique_ptr<Vocab> vocab1 = vocabs.pop();
  for (int i = 1; i < num_threads; i++) {
    std::unique_ptr<Vocab> vocab2 = vocabs.pop();
    if (vocab1->size() < vocab2->size()) {
      for (const auto &e : *vocab1) {
        (*vocab2)[e.first] += e.second;
      }
      std::swap(vocab1, vocab2);
    } else {
      for (const auto &e : *vocab2) {
        (*vocab1)[e.first] += e.second;
      }
    }
  }
  return vocab1;
}

int main(int argc, char **argv) {
  // Performance optimizations for writing to stdout
  std::ios::sync_with_stdio(false);

  CLI::App app("Simple tool to extract unigram counts");
  std::vector<std::string> files;
  app.add_option("FILES", files, "File names")->check(CLI::ExistingPath);
  unsigned int minCount = 10;
  app.add_option("-c,--minCount", minCount,
                 "Minimum number of occurences required for a word to be "
                 "included in the vocabulary.");
  unsigned int num_threads = 1;
  app.add_option(
         "-j,--num_threads", num_threads,
         "Number of threads to use. Each thread constructs an "
         "independent vocabulary which are finally merged. Only appropriate "
         "when multiple, sufficiently large input files are specified.")
      ->check(CLI::Range(1U, std::numeric_limits<unsigned int>::max()));
  CLI11_PARSE(app, argc, argv);

  std::queue<fs::path> paths;
  for (auto &file : files) {
    paths.emplace(file);
  }
  std::vector<std::thread> threads;
  queue<Vocab> vocabs;
  for (unsigned int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(
        [&paths, &vocabs]() { ReadVocab(std::ref(paths), std::ref(vocabs)); }));
  }
  std::unique_ptr<Vocab> vocab = CombineVocabs(vocabs, num_threads);
  for (unsigned int i = 0; i < num_threads; i++) {
    threads[i].join();
  }

  // Sort
  typedef std::function<bool(std::pair<std::string, int>,
                             std::pair<std::string, int>)>
      Comparator;
  Comparator CompFunctor = [](std::pair<std::string, int> elem1,
                              std::pair<std::string, int> elem2) {
    return (elem1.second > elem2.second) ||
           (elem1.second == elem2.second && elem1.first < elem2.first);
  };
  std::set<std::pair<std::string, uint32_t>, Comparator> sorted_vocab(
      vocab->begin(), vocab->end(), CompFunctor);
  vocab.reset(); // Release ownership

  // Output
  for (const auto &e : sorted_vocab) {
    if (e.second < minCount) {
      break;
    }
    std::cout << e.first << "\t" << e.second << "\n";
  }

  return 0;
}
