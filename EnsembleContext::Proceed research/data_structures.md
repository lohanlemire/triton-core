# Data Structures Analysis for EnsembleContext::Proceed

**[index.md](index.md)**

## Overview

This document provides a comprehensive analysis of the data structures that enter the `EnsembleContext::Proceed` function, their expected forms, sizes, and how they contribute to the scaling issues identified in the ensemble scheduler.

## Function Signature Analysis

### EnsembleContext::Proceed Input Parameters

```cpp
static void Proceed(
    const std::shared_ptr<EnsembleContext>& context,
    const std::unique_ptr<Step>& completed_step = nullptr);
```

**Input Data:**
1. **`context`**: `std::shared_ptr<EnsembleContext>` - The ensemble execution context
2. **`completed_step`**: `std::unique_ptr<Step>` - The completed step (can be nullptr for initialization)

## Core Data Structures

### 1. EnsembleContext Structure

The `EnsembleContext` class contains the following key data members that affect performance:

#### Critical Performance Data Members

```cpp
class EnsembleContext {
private:
  // 🔥 CRITICAL: Main tensor data storage - O(N) iteration bottleneck
  std::unordered_map<std::string, TensorData> tensor_data_;
  
  // 🔥 CRITICAL: Tensor to step mapping - used in nested loops
  std::unordered_map<std::string, std::set<size_t>>* tensor_to_step_;
  std::unordered_map<std::string, std::set<size_t>> pruned_tensor_to_step_;
  
  // 🔥 CRITICAL: Model handles - affects memory usage
  std::unordered_map<ModelIdentifier, VersionMap> handles_;
  
  // 🔥 CRITICAL: Ensemble information - contains step definitions
  EnsembleInfo* info_;
  
  // 🔥 CRITICAL: Synchronization bottleneck
  std::mutex mutex_;
  
  // Performance counters
  size_t inflight_step_counter_;
  
  // Request metadata
  uint32_t flags_;
  std::string request_id_;
  InferenceRequest::SequenceId correlation_id_;
  uint64_t priority_;
  uint64_t timeout_;
  std::deque<InferenceParameter> parameters_;
  
  // Status and tracking
  Status ensemble_status_;
  RequestTracker* request_tracker_;
  bool response_sent_{false};
  
  // Memory management
  std::unique_ptr<TRITONSERVER_ResponseAllocator, decltype(&TRITONSERVER_ResponseAllocatorDelete)> allocator_;
  triton::common::ThreadPool* const callback_pool_;
};
```

### 2. Step Structure

The `Step` struct contains data that flows through the ensemble system:

```cpp
struct Step {
  // Context reference
  std::shared_ptr<EnsembleContext> ctx_;
  
  // Request data
  std::unique_ptr<InferenceRequest> request_;
  InferenceRequest::SequenceId correlation_id_;
  uint32_t flags_;
  
  // 🔥 CRITICAL: Output memory management - can be large
  std::mutex output_mtx_;
  std::unordered_map<uintptr_t, std::shared_ptr<AllocatedMemory>> cpu_output_map_;
  std::unordered_map<int64_t, std::unordered_map<uintptr_t, std::shared_ptr<AllocatedMemory>>> gpu_output_map_;
  
  // 🔥 CRITICAL: Updated tensors - key data for scaling analysis
  std::set<std::pair<std::string, IterationCount>> updated_tensors_;
  
  // Response data
  uint32_t response_flags_;
  TRITONSERVER_InferenceResponse* response_;
  const bool preserve_responses_order_;
  
  // Step identification
  size_t step_idx_;
};
```

### 3. TensorData Structure

The `TensorData` structure is central to the scaling issues:

```cpp
struct TensorData {
  struct Metadata {
    std::unique_ptr<InferenceRequest::Input> data_;
    size_t remaining_reference_count_;
    bool parameter_override_;
    InferenceRequest::SequenceId correlation_id_;
    uint32_t flags_;
  };
  
  // 🔥 CRITICAL: Main tensor storage - O(N) iteration target
  std::unordered_map<IterationCount, Metadata> tensor_;
  size_t current_iteration_;
  size_t outgoing_steps_count_;
  size_t batch_size_;
};
```

### 4. EnsembleInfo Structure

The `EnsembleInfo` contains the ensemble configuration:

```cpp
struct EnsembleInfo {
  std::string ensemble_name_;
  bool is_decoupled_;
  bool is_cache_enabled_;
  
  // 🔥 CRITICAL: Output shape definitions
  std::unordered_map<std::string, triton::common::DimsList> ensemble_output_shape_;
  
  // 🔥 CRITICAL: Optional inputs
  std::set<std::string> optional_inputs_;
  
  // 🔥 CRITICAL: Step definitions - affects nested loop complexity
  std::vector<StepInfo> steps_;
  
  // 🔥 CRITICAL: Tensor to step mapping - used in GetNextSteps
  std::unordered_map<std::string, std::set<size_t>> tensor_to_step_;
  std::unordered_map<std::string, size_t> tensor_to_prev_step_;
  
  struct StepInfo {
    ModelIdentifier model_id_;
    int64_t model_version_;
    std::unordered_map<std::string, std::string> input_to_tensor_;
    std::unordered_map<std::string, std::string> output_to_tensor_;
  };
};
```

## Data Flow Analysis

### 1. Input Data Forms

#### EnsembleContext Input
- **Size**: Depends on ensemble complexity
- **Form**: Shared pointer to context object
- **Content**: All ensemble state, tensor data, model handles
- **Expected Size**: 
  - Small ensembles: ~1-10KB
  - Large ensembles: ~100KB-1MB
  - Very large ensembles: >1MB

#### Step Input
- **Size**: Variable, depends on step outputs
- **Form**: Unique pointer to step object (can be nullptr)
- **Content**: Step metadata, output tensors, response data
- **Expected Size**:
  - Small steps: ~1-5KB
  - Large steps: ~10-100KB
  - Steps with large outputs: >100KB

### 2. Data Structure Sizes

#### tensor_data_ Map
```cpp
std::unordered_map<std::string, TensorData> tensor_data_;
```
- **Key**: `std::string` (tensor name) - typically 10-100 bytes
- **Value**: `TensorData` - ~100-1000 bytes per tensor
- **Total Size**: O(N) where N = number of tensors
- **Expected Sizes**:
  - Small ensemble (10 tensors): ~1-10KB
  - Medium ensemble (100 tensors): ~10-100KB
  - Large ensemble (1000 tensors): ~100KB-1MB

#### tensor_to_step_ Map
```cpp
std::unordered_map<std::string, std::set<size_t>>* tensor_to_step_;
```
- **Key**: `std::string` (tensor name) - typically 10-100 bytes
- **Value**: `std::set<size_t>` - variable size based on step count
- **Total Size**: O(N × S) where N = tensors, S = average steps per tensor
- **Expected Sizes**:
  - Small ensemble: ~1-5KB
  - Medium ensemble: ~10-50KB
  - Large ensemble: ~100-500KB

#### handles_ Map
```cpp
std::unordered_map<ModelIdentifier, VersionMap> handles_;
```
- **Key**: `ModelIdentifier` - ~50-200 bytes
- **Value**: `VersionMap` - ~100-500 bytes per model
- **Total Size**: O(M) where M = number of models
- **Expected Sizes**:
  - Small ensemble (5 models): ~1-5KB
  - Medium ensemble (20 models): ~5-20KB
  - Large ensemble (100 models): ~20-100KB

### 3. Critical Data Access Patterns

#### O(N) Iteration Pattern (Primary Bottleneck)
```cpp
// In UpdateEnsembleState - lines 901-905
for (const auto& tensor_data : tensor_data_) {
  if (!tensor_data.second.tensor_.empty()) {
    updated_tensors->emplace(tensor_data.first, 0);
  }
}
```
- **Access Pattern**: Linear iteration through ALL tensor data
- **Complexity**: O(N) where N = total number of tensors
- **Frequency**: Called on every step completion
- **Impact**: Scales linearly with ensemble size

#### Nested Loop Pattern (Secondary Bottleneck)
```cpp
// In GetNextSteps - lines 926-948
for (const auto& updated_tensor : updated_tensors) {
  const auto& step_idx = (*tensor_to_step_)[updated_tensor.first];
  for (const auto& idx : step_idx) {
    for (const auto& input_pair : info_->steps_[idx].input_to_tensor_) {
      // Tensor lookup and iteration count checking
    }
  }
}
```
- **Access Pattern**: Triple nested loops
- **Complexity**: O(U × S × I) where:
  - U = number of updated tensors
  - S = average steps per tensor
  - I = average inputs per step
- **Frequency**: Called on every step completion
- **Impact**: Can become O(N³) in worst case

## Expected Data Sizes by Ensemble Complexity

### Small Ensemble (10-50 tensors, 5-20 steps)
- **tensor_data_**: ~1-50KB
- **tensor_to_step_**: ~1-10KB
- **handles_**: ~1-10KB
- **Total Context**: ~5-100KB
- **Step Data**: ~1-10KB per step

### Medium Ensemble (50-200 tensors, 20-100 steps)
- **tensor_data_**: ~50-500KB
- **tensor_to_step_**: ~10-100KB
- **handles_**: ~10-50KB
- **Total Context**: ~100KB-1MB
- **Step Data**: ~10-100KB per step

### Large Ensemble (200-1000 tensors, 100-500 steps)
- **tensor_data_**: ~500KB-5MB
- **tensor_to_step_**: ~100KB-1MB
- **handles_**: ~50-500KB
- **Total Context**: ~1-10MB
- **Step Data**: ~100KB-1MB per step

### Very Large Ensemble (1000+ tensors, 500+ steps)
- **tensor_data_**: ~5MB+
- **tensor_to_step_**: ~1MB+
- **handles_**: ~500KB+
- **Total Context**: ~10MB+
- **Step Data**: ~1MB+ per step

## Memory Access Patterns

### 1. Sequential Access (Good)
- Iterating through `updated_tensors` set
- Processing step outputs in order
- Linear memory access patterns

### 2. Random Access (Moderate)
- Hash map lookups in `tensor_data_`
- Hash map lookups in `tensor_to_step_`
- Cache-friendly for small maps, cache-unfriendly for large maps

### 3. Nested Access (Poor)
- Triple nested loops in `GetNextSteps`
- Multiple hash map lookups per iteration
- Poor cache locality, especially with large data structures

## Data Structure Optimization Opportunities

### 1. Replace O(N) Iteration
```cpp
// Current (O(N))
for (const auto& tensor_data : tensor_data_) {
  if (!tensor_data.second.tensor_.empty()) {
    updated_tensors->emplace(tensor_data.first, 0);
  }
}

// Optimized (O(1) for common case)
if (completed_step != nullptr) {
  updated_tensors->swap(completed_step->updated_tensors_);
} else {
  // Only for initialization - use more efficient approach
  for (const auto& [name, data] : tensor_data_) {
    if (!data.tensor_.empty()) {
      updated_tensors->emplace(name, 0);
    }
  }
}
```

### 2. Optimize Nested Loops
```cpp
// Current (O(U × S × I))
for (const auto& updated_tensor : updated_tensors) {
  const auto& step_idx = (*tensor_to_step_)[updated_tensor.first];
  for (const auto& idx : step_idx) {
    for (const auto& input_pair : info_->steps_[idx].input_to_tensor_) {
      // Complex logic
    }
  }
}

// Optimized (O(U + S))
// Pre-compute step readiness using bitmasks or more efficient data structures
// Cache step readiness state to avoid recomputation
```

### 3. Improve Data Structure Layout
- Use more cache-friendly data structures
- Reduce memory fragmentation
- Optimize hash map sizes and load factors
- Consider using arrays instead of maps for small, fixed-size data

## Conclusion

The data structures entering `EnsembleContext::Proceed` are well-designed for functionality but have significant scaling issues:

1. **Primary Issue**: `tensor_data_` map requires O(N) iteration on every step completion
2. **Secondary Issue**: Nested loop complexity in `GetNextSteps` creates O(N³) worst-case behavior
3. **Memory Impact**: Large ensembles can consume 10MB+ of memory per context
4. **Cache Impact**: Poor memory access patterns in nested loops

The most impactful optimization would be to eliminate the full tensor data iteration and only process actually updated tensors, changing the complexity from O(N) to O(1) for the common case.
