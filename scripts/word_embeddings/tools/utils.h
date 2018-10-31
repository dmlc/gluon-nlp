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
