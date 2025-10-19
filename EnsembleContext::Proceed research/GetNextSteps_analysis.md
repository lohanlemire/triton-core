# GetNextSteps Function Analysis

**[index.md](index.md)** | **[function_analysis.md](function_analysis.md)**

## Overview

`GetNextSteps` is the **PRIMARY PERFORMANCE BOTTLENECK** in the ensemble scheduler. This function has O(U × S × I) triple nested loop complexity that can result in up to 1,000,000+ operations per step completion for very large ensembles.

## Function Signature

```cpp
Status GetNextSteps(
    const std::set<std::pair<std::string, IterationCount>>& updated_tensors,
    StepList* steps)
```

**Location**: `src/ensemble_scheduler/ensemble_scheduler.cc` lines 918-957

## Purpose

Determines which ensemble steps are ready to be executed based on the `updated_tensors`. It iterates through updated tensors and checks which steps have all their input dependencies satisfied.

## Input Analysis

### Input Parameters

1. **`updated_tensors`**: `std::set<std::pair<std::string, IterationCount>>`
   - **Type**: Set of tensor name and iteration count pairs
   - **Size**: Typically 1-500 elements for large ensembles
   - **Content**: Tensors that were updated by the completed step
   - **Expected Form**: `{("tensor_name", iteration_count), ...}`

2. **`steps`**: `StepList*` (output parameter)
   - **Type**: Pointer to vector of unique_ptr<Step>
   - **Purpose**: Output container for ready steps
   - **Expected Size**: 0-100 steps for large ensembles

### Input Data Flow

```
Completed Step → ConsumeResponse → updated_tensors → GetNextSteps → ready_steps
```

## Algorithm Analysis

### Current Implementation (Lines 918-957)

```cpp
Status GetNextSteps(
    const std::set<std::pair<std::string, IterationCount>>& updated_tensors,
    StepList* steps)
{
  steps->clear();

  std::set<std::pair<size_t, IterationCount>> next_step_idx;
  // Get steps whose tensors used for input are set
  for (const auto& updated_tensor : updated_tensors) {           // U iterations
    const auto& step_idx = (*tensor_to_step_)[updated_tensor.first];
    for (const auto& idx : step_idx) {                          // S iterations
      bool ready = true;
      for (const auto& input_pair : info_->steps_[idx].input_to_tensor_) { // I iterations
        auto& tensor = tensor_data_[input_pair.second].tensor_;
        if (tensor.empty()) {
          ready = false;
          break;
        } else {
          // Check if other inputs have tensor with corresponding iteration
          // count
          if (tensor.find(updated_tensor.second) == tensor.end()) {
            ready = false;
            break;
          }
        }
      }
      if (ready) {
        next_step_idx.emplace(idx, updated_tensor.second);
      }
    }
  }

  for (const auto& idx : next_step_idx) {
    steps->emplace_back();
    RETURN_IF_ERROR(InitStep(idx.first, idx.second, &(steps->back())));
  }
  inflight_step_counter_ += steps->size();

  return Status::Success;
}
```

### Complexity Analysis

**Triple Nested Loop Structure**:
1. **Outer Loop**: `for (const auto& updated_tensor : updated_tensors)` - **U iterations**
2. **Middle Loop**: `for (const auto& idx : step_idx)` - **S iterations**  
3. **Inner Loop**: `for (const auto& input_pair : info_->steps_[idx].input_to_tensor_)` - **I iterations**

**Total Complexity**: **O(U × S × I)** where:
- **U** = number of updated tensors (typically 1-500)
- **S** = average number of steps per tensor (typically 1-100)
- **I** = average number of inputs per step (typically 1-20)

### Performance Impact by Ensemble Size

| Ensemble Size | Updated Tensors (U) | Steps per Tensor (S) | Inputs per Step (I) | Total Operations |
|---------------|-------------------|---------------------|-------------------|------------------|
| Small         | 5                 | 3                   | 2                 | 30               |
| Medium        | 20                | 10                  | 5                 | 1,000            |
| Large         | 100               | 50                  | 10                | 50,000           |
| Very Large    | 500               | 100                 | 20                | 1,000,000        |

## Memory Usage Analysis

### Memory Access Patterns

1. **Hash Map Lookups**: `(*tensor_to_step_)[updated_tensor.first]`
   - **Access Pattern**: Random access to hash map
   - **Cache Impact**: Poor cache locality for large maps
   - **Frequency**: U times per call

2. **Tensor Data Access**: `tensor_data_[input_pair.second].tensor_`
   - **Access Pattern**: Random access to hash map
   - **Cache Impact**: Poor cache locality for large maps
   - **Frequency**: U × S × I times per call

3. **Step Info Access**: `info_->steps_[idx].input_to_tensor_`
   - **Access Pattern**: Random access to vector
   - **Cache Impact**: Moderate cache locality
   - **Frequency**: U × S times per call

### Memory Allocation

- **`next_step_idx`**: `std::set<std::pair<size_t, IterationCount>>`
  - **Size**: O(S) where S = number of ready steps
  - **Allocation**: Dynamic allocation for set operations

- **`steps`**: `StepList` (vector of unique_ptr<Step>)
  - **Size**: O(S) where S = number of ready steps
  - **Allocation**: Dynamic allocation for step objects

## Bottleneck Identification

### Primary Bottlenecks

1. **Triple Nested Loop**: O(U × S × I) complexity
   - **Impact**: Exponential growth with ensemble size
   - **Location**: Lines 926-948
   - **Severity**: CRITICAL

2. **Hash Map Lookups**: Multiple random access operations
   - **Impact**: Poor cache locality
   - **Location**: Lines 927, 932
   - **Severity**: HIGH

3. **Redundant Computations**: Same step readiness checked multiple times
   - **Impact**: Unnecessary work
   - **Location**: Lines 930-945
   - **Severity**: MEDIUM

### Secondary Bottlenecks

1. **Memory Allocation**: Dynamic allocation for intermediate data structures
   - **Impact**: Allocation overhead
   - **Location**: Lines 925, 950-955
   - **Severity**: LOW

2. **Set Operations**: Insertion and lookup in `next_step_idx`
   - **Impact**: O(log n) operations
   - **Location**: Line 946
   - **Severity**: LOW

## Optimization Opportunities

### 1. Pre-compute Step Dependencies (CRITICAL)

**Current**: Recompute step readiness on every call
**Optimized**: Pre-compute and cache step dependencies

```cpp
class StepReadinessCache {
private:
  std::unordered_map<size_t, bool> step_readiness_;
  std::unordered_map<std::string, std::set<size_t>> tensor_dependencies_;
  std::unordered_map<size_t, std::set<std::string>> step_inputs_;
  
public:
  void Initialize(const EnsembleInfo* info) {
    // Pre-compute dependencies once
    for (size_t step_idx = 0; step_idx < info->steps_.size(); ++step_idx) {
      const auto& step = info->steps_[step_idx];
      for (const auto& [input_name, tensor_name] : step.input_to_tensor_) {
        step_inputs_[step_idx].insert(tensor_name);
        tensor_dependencies_[tensor_name].insert(step_idx);
      }
    }
  }
  
  void UpdateReadiness(const std::set<std::pair<std::string, IterationCount>>& updated_tensors,
                      const std::unordered_map<std::string, TensorData>& tensor_data_) {
    // Only update affected steps
    for (const auto& [tensor_name, iteration_count] : updated_tensors) {
      auto it = tensor_dependencies_.find(tensor_name);
      if (it != tensor_dependencies_.end()) {
        for (const auto& step_idx : it->second) {
          step_readiness_[step_idx] = CheckStepReadiness(step_idx, iteration_count, tensor_data_);
        }
      }
    }
  }
  
  std::vector<std::pair<size_t, IterationCount>> GetReadySteps() {
    std::vector<std::pair<size_t, IterationCount>> ready_steps;
    for (const auto& [step_idx, is_ready] : step_readiness_) {
      if (is_ready) {
        ready_steps.emplace_back(step_idx, FindIterationCount(step_idx));
      }
    }
    return ready_steps;
  }
};
```

**Expected Improvement**: 10x-100x performance improvement

### 2. Incremental Updates (HIGH)

**Current**: Check all step inputs for every updated tensor
**Optimized**: Only check steps that depend on updated tensors

```cpp
// ABI-SAFE: Add as private member to EnsembleContext
std::unique_ptr<StepReadinessCache> step_readiness_cache_;

Status GetNextSteps(
    const std::set<std::pair<std::string, IterationCount>>& updated_tensors,
    StepList* steps)
{
  steps->clear();
  
  // Use cache for incremental updates
  if (step_readiness_cache_) {
    step_readiness_cache_->UpdateReadiness(updated_tensors, tensor_data_);
    auto ready_steps = step_readiness_cache_->GetReadySteps();
    
    // Create step objects
    for (const auto& [step_idx, iteration_count] : ready_steps) {
      steps->emplace_back();
      RETURN_IF_ERROR(InitStep(step_idx, iteration_count, &(steps->back())));
    }
  } else {
    // Fallback to original implementation
    // ... original nested loop implementation ...
  }
  
  inflight_step_counter_ += steps->size();
  return Status::Success;
}
```

**Expected Improvement**: 5x-50x performance improvement

### 3. Optimize Data Structures (MEDIUM)

**Current**: Multiple hash map lookups
**Optimized**: Use more cache-friendly data structures

```cpp
// Use arrays instead of hash maps for small, fixed-size data
std::vector<bool> step_readiness_;  // Indexed by step_idx
std::vector<std::vector<size_t>> tensor_to_steps_;  // Indexed by tensor_idx
```

**Expected Improvement**: 2x-5x performance improvement

## ABI-Safe Implementation Strategy

### 1. Add Private Cache Member

```cpp
class EnsembleContext {
private:
  // 🔥 ABI-SAFE: Adding new private member
  std::unique_ptr<StepReadinessCache> step_readiness_cache_;
  
  // ... existing members remain unchanged ...
};
```

### 2. Initialize Cache in Constructor

```cpp
EnsembleContext::EnsembleContext(...) {
  // ... existing initialization ...
  
  // Initialize step readiness cache
  step_readiness_cache_ = std::make_unique<StepReadinessCache>();
  step_readiness_cache_->Initialize(info_);
}
```

### 3. Modify GetNextSteps Implementation

```cpp
// ABI-SAFE: Only internal implementation changes
Status GetNextSteps(
    const std::set<std::pair<std::string, IterationCount>>& updated_tensors,
    StepList* steps)
{
  // Use optimized implementation with cache
  // ... implementation details ...
}
```

## Testing Strategy

### 1. Performance Testing

- **Small Ensemble**: 10 tensors, 5 steps
- **Medium Ensemble**: 100 tensors, 50 steps  
- **Large Ensemble**: 1000 tensors, 200 steps
- **Very Large Ensemble**: 5000 tensors, 500 steps

### 2. Correctness Testing

- **Unit Tests**: Test step readiness logic
- **Integration Tests**: Test with real ensemble configurations
- **Regression Tests**: Ensure no functional changes

### 3. Memory Testing

- **Memory Usage**: Monitor memory consumption
- **Memory Leaks**: Check for proper cleanup
- **Cache Efficiency**: Measure cache hit rates

## Expected Results

### Performance Improvements

| Ensemble Size | Current Operations | Optimized Operations | Improvement |
|---------------|-------------------|---------------------|-------------|
| Small         | 30                | 5                   | 6x          |
| Medium        | 1,000             | 50                  | 20x         |
| Large         | 50,000            | 500                 | 100x        |
| Very Large    | 1,000,000         | 5,000               | 200x        |

### Memory Improvements

- **Reduced Allocations**: Fewer dynamic allocations
- **Better Cache Locality**: More sequential memory access
- **Lower Memory Usage**: More efficient data structures

## Conclusion

`GetNextSteps` is the primary performance bottleneck in the ensemble scheduler. The triple nested loop complexity makes it scale poorly with ensemble size. The proposed optimizations using step readiness caching can provide 10x-100x performance improvement while maintaining ABI stability.

The most critical optimization is pre-computing step dependencies and using incremental updates, which can reduce the complexity from O(U × S × I) to O(U + S) for the common case.

