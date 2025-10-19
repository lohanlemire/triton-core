# Optimization Strategies for GetNextSteps Function

## Overview

This document outlines comprehensive optimization strategies to eliminate the performance bottleneck in the `GetNextSteps` function, specifically targeting the O(U × S × I) triple nested loop.

## Strategy 1: Algorithmic Redesign (HIGH IMPACT)

### 1.1 Reverse Indexing Approach

**Current Algorithm**:
```
For each updated tensor → Find dependent steps → Check step readiness
```

**Optimized Algorithm**:
```
For each step → Check if all dependencies are satisfied
```

**Implementation**:
```cpp
Status GetNextSteps(
    const std::set<std::pair<std::string, IterationCount>>& updated_tensors,
    StepList* steps)
{
  steps->clear();
  
  // Create a set of updated tensor names for O(1) lookup
  std::unordered_set<std::string> updated_tensor_names;
  std::unordered_map<std::string, IterationCount> updated_iterations;
  
  for (const auto& updated_tensor : updated_tensors) {
    updated_tensor_names.insert(updated_tensor.first);
    updated_iterations[updated_tensor.first] = updated_tensor.second;
  }
  
  std::set<std::pair<size_t, IterationCount>> next_step_idx;
  
  // Iterate through all steps instead of updated tensors
  for (size_t step_idx = 0; step_idx < info_->steps_.size(); ++step_idx) {
    const auto& step = info_->steps_[step_idx];
    bool ready = true;
    IterationCount required_iteration = 0;
    
    // Check if all inputs for this step are satisfied
    for (const auto& input_pair : step.input_to_tensor_) {
      const std::string& tensor_name = input_pair.second;
      
      // Check if this tensor was updated
      if (updated_tensor_names.find(tensor_name) != updated_tensor_names.end()) {
        // Check if tensor data exists and has required iteration
        auto& tensor_data = tensor_data_[tensor_name];
        auto& tensor = tensor_data.tensor_;
        
        if (tensor.empty()) {
          ready = false;
          break;
        }
        
        IterationCount updated_iteration = updated_iterations[tensor_name];
        if (tensor.find(updated_iteration) == tensor.end()) {
          ready = false;
          break;
        }
        
        required_iteration = updated_iteration;
      } else {
        // Tensor wasn't updated, check if it has any data
        auto& tensor_data = tensor_data_[tensor_name];
        auto& tensor = tensor_data.tensor_;
        
        if (tensor.empty()) {
          ready = false;
          break;
        }
      }
    }
    
    if (ready) {
      next_step_idx.emplace(step_idx, required_iteration);
    }
  }
  
  // Rest of the function remains the same...
}
```

**Complexity**: O(S × I) where S = total steps, I = average inputs per step
**Improvement**: Eliminates the U factor from complexity

### 1.2 Dependency Tracking with Bitmasks

**Concept**: Use bitmasks to track partial satisfaction of step dependencies.

**Implementation**:
```cpp
struct StepDependencyTracker {
  std::vector<bool> step_ready_;
  std::unordered_map<std::string, std::vector<size_t>> tensor_to_steps_;
  std::unordered_map<size_t, std::unordered_set<std::string>> step_dependencies_;
  std::unordered_map<size_t, std::unordered_set<std::string>> step_satisfied_deps_;
  
  void Initialize(const EnsembleInfo* info) {
    step_ready_.resize(info->steps_.size(), false);
    
    for (size_t i = 0; i < info->steps_.size(); ++i) {
      const auto& step = info->steps_[i];
      for (const auto& input_pair : step.input_to_tensor_) {
        const std::string& tensor_name = input_pair.second;
        tensor_to_steps_[tensor_name].push_back(i);
        step_dependencies_[i].insert(tensor_name);
      }
    }
  }
  
  void UpdateTensor(const std::string& tensor_name, IterationCount iteration) {
    auto it = tensor_to_steps_.find(tensor_name);
    if (it != tensor_to_steps_.end()) {
      for (size_t step_idx : it->second) {
        step_satisfied_deps_[step_idx].insert(tensor_name);
        
        // Check if step is now ready
        if (step_satisfied_deps_[step_idx].size() == step_dependencies_[step_idx].size()) {
          step_ready_[step_idx] = true;
        }
      }
    }
  }
  
  std::vector<size_t> GetReadySteps() const {
    std::vector<size_t> ready_steps;
    for (size_t i = 0; i < step_ready_.size(); ++i) {
      if (step_ready_[step_idx]) {
        ready_steps.push_back(i);
      }
    }
    return ready_steps;
  }
};
```

**Complexity**: O(1) per tensor update, O(S) for final check
**Improvement**: Eliminates nested loops entirely

## Strategy 2: Data Structure Optimizations (MEDIUM IMPACT)

### 2.1 Pre-computed Step Dependencies

**Concept**: Cache step dependency information during initialization to avoid repeated hash map lookups.

**Implementation**:
```cpp
struct CachedStepInfo {
  std::vector<std::string> input_tensors_;
  std::vector<std::string> output_tensors_;
  size_t dependency_count_;
  
  CachedStepInfo(const StepInfo& step_info) {
    for (const auto& input_pair : step_info.input_to_tensor_) {
      input_tensors_.push_back(input_pair.second);
    }
    for (const auto& output_pair : step_info.output_to_tensor_) {
      output_tensors_.push_back(output_pair.second);
    }
    dependency_count_ = input_tensors_.size();
  }
};

class OptimizedEnsembleContext {
private:
  std::vector<CachedStepInfo> cached_step_info_;
  std::unordered_map<std::string, std::vector<size_t>> tensor_to_steps_cache_;
  
public:
  void InitializeCaches() {
    // Pre-compute step information
    cached_step_info_.reserve(info_->steps_.size());
    for (const auto& step : info_->steps_) {
      cached_step_info_.emplace_back(step);
    }
    
    // Pre-compute tensor-to-steps mapping
    for (size_t i = 0; i < info_->steps_.size(); ++i) {
      const auto& step = info_->steps_[i];
      for (const auto& input_pair : step.input_to_tensor_) {
        tensor_to_steps_cache_[input_pair.second].push_back(i);
      }
    }
  }
};
```

**Benefits**:
- Eliminates hash map lookups during execution
- Better cache locality with pre-computed data
- Reduces memory allocation overhead

### 2.2 Memory Pool Allocation

**Concept**: Use memory pools for frequently allocated objects to reduce allocation overhead.

**Implementation**:
```cpp
class StepAllocationPool {
private:
  std::vector<std::unique_ptr<Step>> step_pool_;
  std::queue<Step*> available_steps_;
  size_t pool_size_;
  
public:
  StepAllocationPool(size_t initial_size = 100) : pool_size_(initial_size) {
    step_pool_.reserve(initial_size);
    for (size_t i = 0; i < initial_size; ++i) {
      step_pool_.push_back(std::make_unique<Step>());
      available_steps_.push(step_pool_.back().get());
    }
  }
  
  Step* AllocateStep() {
    if (available_steps_.empty()) {
      // Expand pool if needed
      ExpandPool();
    }
    
    Step* step = available_steps_.front();
    available_steps_.pop();
    return step;
  }
  
  void DeallocateStep(Step* step) {
    // Reset step state
    step->Reset();
    available_steps_.push(step);
  }
  
private:
  void ExpandPool() {
    size_t new_size = pool_size_ * 2;
    step_pool_.reserve(new_size);
    
    for (size_t i = pool_size_; i < new_size; ++i) {
      step_pool_.push_back(std::make_unique<Step>());
      available_steps_.push(step_pool_.back().get());
    }
    
    pool_size_ = new_size;
  }
};
```

## Strategy 3: Parallel Processing (HIGH IMPACT)

### 3.1 Parallel Step Readiness Checking

**Concept**: Check step readiness in parallel for independent steps.

**Implementation**:
```cpp
Status GetNextSteps(
    const std::set<std::pair<std::string, IterationCount>>& updated_tensors,
    StepList* steps)
{
  steps->clear();
  
  // Create thread-safe containers
  std::mutex ready_steps_mutex;
  std::vector<std::pair<size_t, IterationCount>> ready_steps;
  
  // Parallel step checking
  std::vector<std::future<void>> futures;
  
  for (size_t step_idx = 0; step_idx < info_->steps_.size(); ++step_idx) {
    futures.push_back(std::async(std::launch::async, [&, step_idx]() {
      if (IsStepReady(step_idx, updated_tensors)) {
        std::lock_guard<std::mutex> lock(ready_steps_mutex);
        ready_steps.emplace_back(step_idx, GetRequiredIteration(step_idx, updated_tensors));
      }
    }));
  }
  
  // Wait for all checks to complete
  for (auto& future : futures) {
    future.wait();
  }
  
  // Create steps from ready steps
  for (const auto& ready_step : ready_steps) {
    steps->emplace_back();
    RETURN_IF_ERROR(InitStep(ready_step.first, ready_step.second, &(steps->back())));
  }
  
  return Status::Success;
}
```

**Benefits**:
- Parallel execution of independent step checks
- Better utilization of multi-core systems
- Reduced wall-clock time

### 3.2 SIMD Optimizations

**Concept**: Use SIMD instructions for vectorized operations where possible.

**Implementation**:
```cpp
// Example: Vectorized tensor iteration count checking
bool CheckTensorIterationsSIMD(
    const std::unordered_map<IterationCount, TensorData::Metadata>& tensor,
    IterationCount required_iteration)
{
  // Convert to vector for SIMD operations
  std::vector<IterationCount> iterations;
  iterations.reserve(tensor.size());
  
  for (const auto& pair : tensor) {
    iterations.push_back(pair.first);
  }
  
  // Use SIMD to find required iteration
  return std::find(iterations.begin(), iterations.end(), required_iteration) != iterations.end();
}
```

## Strategy 4: Caching and Memoization (MEDIUM IMPACT)

### 4.1 Step Readiness Cache

**Concept**: Cache step readiness results to avoid recomputation.

**Implementation**:
```cpp
class StepReadinessCache {
private:
  struct CacheEntry {
    std::unordered_set<std::string> satisfied_tensors_;
    bool is_ready_;
    uint64_t timestamp_;
  };
  
  std::unordered_map<size_t, CacheEntry> cache_;
  uint64_t current_timestamp_;
  
public:
  bool IsStepReady(size_t step_idx, const std::set<std::pair<std::string, IterationCount>>& updated_tensors) {
    auto it = cache_.find(step_idx);
    if (it != cache_.end() && it->second.timestamp_ == current_timestamp_) {
      // Check if cache is still valid
      bool cache_valid = true;
      for (const auto& updated_tensor : updated_tensors) {
        if (it->second.satisfied_tensors_.find(updated_tensor.first) == it->second.satisfied_tensors_.end()) {
          cache_valid = false;
          break;
        }
      }
      
      if (cache_valid) {
        return it->second.is_ready_;
      }
    }
    
    // Compute step readiness
    bool is_ready = ComputeStepReadiness(step_idx, updated_tensors);
    
    // Update cache
    CacheEntry entry;
    for (const auto& updated_tensor : updated_tensors) {
      entry.satisfied_tensors_.insert(updated_tensor.first);
    }
    entry.is_ready_ = is_ready;
    entry.timestamp_ = current_timestamp_;
    
    cache_[step_idx] = std::move(entry);
    
    return is_ready;
  }
  
  void InvalidateCache() {
    current_timestamp_++;
  }
};
```

### 4.2 Tensor Dependency Cache

**Concept**: Cache tensor-to-step mappings to reduce hash map lookups.

**Implementation**:
```cpp
class TensorDependencyCache {
private:
  std::unordered_map<std::string, std::vector<size_t>> tensor_to_steps_;
  std::unordered_map<size_t, std::vector<std::string>> step_to_tensors_;
  
public:
  void Initialize(const EnsembleInfo* info) {
    for (size_t i = 0; i < info->steps_.size(); ++i) {
      const auto& step = info->steps_[i];
      for (const auto& input_pair : step.input_to_tensor_) {
        const std::string& tensor_name = input_pair.second;
        tensor_to_steps_[tensor_name].push_back(i);
        step_to_tensors_[i].push_back(tensor_name);
      }
    }
  }
  
  const std::vector<size_t>& GetStepsForTensor(const std::string& tensor_name) const {
    auto it = tensor_to_steps_.find(tensor_name);
    if (it != tensor_to_steps_.end()) {
      return it->second;
    }
    return empty_vector_; // Return empty vector for non-existent tensors
  }
  
  const std::vector<std::string>& GetTensorsForStep(size_t step_idx) const {
    auto it = step_to_tensors_.find(step_idx);
    if (it != step_to_tensors_.end()) {
      return it->second;
    }
    return empty_vector_; // Return empty vector for non-existent steps
  }
  
private:
  static const std::vector<size_t> empty_vector_;
  static const std::vector<std::string> empty_vector_strings_;
};
```

## Strategy 5: Event-Driven Architecture (HIGH IMPACT)

### 5.1 Event-Driven Step Scheduling

**Concept**: Use an event-driven system where tensor updates trigger step readiness events.

**Implementation**:
```cpp
class EventDrivenScheduler {
private:
  struct StepEvent {
    size_t step_idx_;
    std::string tensor_name_;
    IterationCount iteration_;
    bool is_ready_;
  };
  
  std::queue<StepEvent> event_queue_;
  std::unordered_map<size_t, std::unordered_set<std::string>> step_dependencies_;
  std::unordered_map<size_t, std::unordered_set<std::string>> step_satisfied_deps_;
  std::unordered_map<size_t, bool> step_ready_status_;
  
public:
  void OnTensorUpdated(const std::string& tensor_name, IterationCount iteration) {
    // Find all steps that depend on this tensor
    auto it = tensor_to_steps_.find(tensor_name);
    if (it != tensor_to_steps_.end()) {
      for (size_t step_idx : it->second) {
        // Mark dependency as satisfied
        step_satisfied_deps_[step_idx].insert(tensor_name);
        
        // Check if step is now ready
        if (step_satisfied_deps_[step_idx].size() == step_dependencies_[step_idx].size()) {
          if (!step_ready_status_[step_idx]) {
            step_ready_status_[step_idx] = true;
            event_queue_.push({step_idx, tensor_name, iteration, true});
          }
        }
      }
    }
  }
  
  std::vector<size_t> ProcessEvents() {
    std::vector<size_t> ready_steps;
    
    while (!event_queue_.empty()) {
      const StepEvent& event = event_queue_.front();
      if (event.is_ready_) {
        ready_steps.push_back(event.step_idx_);
      }
      event_queue_.pop();
    }
    
    return ready_steps;
  }
};
```

## Implementation Priority

### Phase 1: Immediate Optimizations (Low Risk)
1. **Pre-computed step dependencies**
2. **Memory pool allocation**
3. **Early termination conditions**
4. **Hash map access optimization**

### Phase 2: Algorithmic Improvements (Medium Risk)
1. **Reverse indexing approach**
2. **Dependency tracking with bitmasks**
3. **Step readiness caching**
4. **Tensor dependency caching**

### Phase 3: Advanced Optimizations (High Risk)
1. **Parallel processing**
2. **Event-driven architecture**
3. **SIMD optimizations**
4. **Complete algorithm redesign**

## Expected Performance Improvements

| Optimization | Complexity Improvement | Expected Speedup | Risk Level |
|--------------|----------------------|------------------|------------|
| Reverse Indexing | O(U×S×I) → O(S×I) | 10-100x | Medium |
| Dependency Tracking | O(U×S×I) → O(1) | 100-1000x | High |
| Parallel Processing | O(S×I) → O(S×I/P) | 2-8x | Medium |
| Caching | O(U×S×I) → O(S×I) | 5-50x | Low |
| Memory Pools | No complexity change | 1.5-3x | Low |

## Conclusion

The optimization strategies outlined above provide a comprehensive approach to eliminating the performance bottleneck in the `GetNextSteps` function. The key is to start with low-risk optimizations and gradually implement more complex solutions while maintaining correctness and reliability.

**Recommended Implementation Order**:
1. Start with caching and memory pool optimizations
2. Implement reverse indexing approach
3. Add parallel processing capabilities
4. Consider event-driven architecture for maximum performance
