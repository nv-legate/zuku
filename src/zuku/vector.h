/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_VECTOR_H_
#define _POC_SRC_VECTOR_H_

#include <functional>
#include <variant>
#include <vector>

namespace zuku {

template <typename T, typename IterType>
class Iterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T*;    // or also value_type*
  using reference = T&;  // or also value_type&

  Iterator(IterType&& iter) : iter_(std::move(iter)) {}

  Iterator& operator++() {
    iter_++;
    return *this;
  }

  Iterator operator++(int) {
    Iterator tmp = *this;
    iter_++;
    return tmp;
  }

  reference operator*() const { return (*iter_).get(); }
  pointer operator->() { return &(*iter_); }

  friend bool operator==(const Iterator& a, const Iterator& b) {
    return a.iter_ == b.iter_;
  };
  friend bool operator!=(const Iterator& a, const Iterator& b) {
    return a.iter_ != b.iter_;
  };

 private:
  IterType iter_;
};

template <class T>
class ro_vector {
 public:
  using iterator_type =
      typename std::vector<std::reference_wrapper<const T>>::const_iterator;
  using iterator = Iterator<const T, iterator_type>;
  using const_iterator = iterator;

  void reserve(std::size_t n) { values_.reserve(n); }

  void push_back(const T& t) { values_.push_back(t); }

  template <class... Args>
  void emplace_back(Args&&... args) {
    values_.emplace_back(std::forward<Args>(args)...);
  }

  const_iterator begin() const { return const_iterator(values_.begin()); }

  const_iterator end() const { return const_iterator(values_.end()); }

  const T& operator[](std::size_t idx) { return values_[idx].get(); }

  std::size_t size() const { return values_.size(); }

 private:
  std::vector<std::reference_wrapper<const T>> values_;
};

template <class T>
class rw_vector {
 public:
  using const_iterator_type =
      typename std::vector<std::reference_wrapper<const T>>::const_iterator;
  using iterator_type =
      typename std::vector<std::reference_wrapper<T>>::iterator;

  rw_vector() = default;
  rw_vector(rw_vector&&) = default;
  rw_vector(const rw_vector&) = delete;

  template <class... Args>
  rw_vector(Args&&... args) {
    values_.reserve(sizeof...(args));
    (push_back(args), ...);
  }

  using iterator = Iterator<T, iterator_type>;
  using const_iterator = Iterator<const T, iterator_type>;

  const_iterator begin() const {
    return Iterator<T, const_iterator_type>(values_.begin());
  }

  const_iterator end() const {
    return Iterator<T, const_iterator_type>(values_.end());
  }

  iterator begin() { return Iterator<T, iterator_type>(values_.begin()); }

  iterator end() { return Iterator<T, iterator_type>(values_.end()); }

  std::size_t size() const { return values_.size(); }

  void reserve(std::size_t n) { values_.reserve(n); }

  void push_back(T& t) { values_.push_back(t); }

  template <class... Args>
  void emplace_back(Args&&... args) {
    values_.emplace_back(std::forward<Args>(args)...);
  }

  T& operator[](std::size_t idx) { return values_[idx].get(); }

 private:
  std::vector<std::reference_wrapper<T>> values_;
};

}  // namespace zuku

#endif
