#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <stddef.h>
#include <thread>
#include <iostream>

#include "Log_Uniform_Sampler.h"

Log_Uniform_Sampler::Log_Uniform_Sampler(const int range_max) : N(range_max), generator(), distribution(0.0, 1.0) {}

std::unordered_set<long> Log_Uniform_Sampler::sample_unique(const size_t size, int* num_tries) {
  std::unordered_set<long> data;
  const double log_N = log(N);
  int n = 0;
  while (data.size() != size) {
    n += 1;
    double x = distribution(generator);
    long value = lround(exp(x * log_N)) - 1;
    if (data.find(value) == data.end()) {
      data.emplace(value);
    }
  }
  *num_tries = n;
  return data;
}
