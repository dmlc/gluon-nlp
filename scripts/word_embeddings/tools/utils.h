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
#pragma once

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>

// Based on https://stackoverflow.com/a/12805690
template <typename T> class queue {
private:
  std::mutex d_mutex;
  std::condition_variable d_condition;
  std::deque<std::unique_ptr<T>> d_queue;

public:
  // Add a value to queue in a thread-safe manner.
  void push(std::unique_ptr<T> value) {
    {
      std::unique_lock<std::mutex> lock(this->d_mutex);
      d_queue.push_front(std::move(value));
    }
    this->d_condition.notify_one();
  }

  // Remove and return a value from the queue in a thread-safe manner (FIFO).
  // Blocks if there is no value in the Queue.
  std::unique_ptr<T> pop() {
    std::unique_lock<std::mutex> lock(this->d_mutex);
    this->d_condition.wait(lock, [&] { return !this->d_queue.empty(); });
    std::unique_ptr<T> rc(std::move(this->d_queue.back()));
    this->d_queue.pop_back();
    return rc;
  }
};
