#ifndef _LOG_UNIFORM_SAMPLER_H
#define _LOG_UNIFORM_SAMPLER_H

#include <unordered_set>
#include <vector>
#include <utility>
#include <random>

class Log_Uniform_Sampler {
private:
  const int N;
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution;
public:
  Log_Uniform_Sampler(const int);
  std::unordered_set<long> sample_unique(const size_t, int*);
};

#endif // _LOG_UNIFORM_SAMPLER_H
