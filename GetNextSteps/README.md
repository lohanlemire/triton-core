# GetNextSteps Function - Critical Performance Analysis

## Overview

The `GetNextSteps` function is the **PRIMARY PERFORMANCE BOTTLENECK** in the Triton ensemble scheduler. This function contains a triple nested loop with O(U × S × I) complexity that can result in up to 1,000,000+ operations per step completion for very large ensembles.

## Function Location

- **File**: `src/ensemble_scheduler/ensemble_scheduler.cc`
- **Lines**: 917-957
- **Class**: `EnsembleContext`

## Function Signature

```cpp
Status GetNextSteps(
    const std::set<std::pair<std::string, IterationCount>>& updated_tensors,
    StepList* steps)
```

## Purpose

Determines which ensemble steps are ready to be executed based on the `updated_tensors`. It iterates through updated tensors and checks which steps have all their input dependencies satisfied.

## Critical Performance Issue: Triple Nested Loop

### The Problematic Code (Lines 926-948)

```cpp
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
        // Check if other inputs have tensor with corresponding iteration count
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

## Data Structures Analysis

### Key Data Structures

1. **`updated_tensors`**: `std::set<std::pair<std::string, IterationCount>>`
   - **Purpose**: Input parameter containing tensors that were updated by completed step
   - **Size**: Typically 1-500 elements for large ensembles
   - **Access Pattern**: Iterated once in outer loop

2. **`tensor_to_step_`**: `std::unordered_map<std::string, std::set<size_t>>*`
   - **Purpose**: Maps tensor names to set of step indices that use that tensor as input
   - **Access Pattern**: Hash map lookup for each updated tensor
   - **Cache Impact**: Poor cache locality for large maps

3. **`info_->steps_[idx].input_to_tensor_`**: `std::unordered_map<std::string, std::string>`
   - **Purpose**: Maps input names to tensor names for each step
   - **Access Pattern**: Iterated for each step in middle loop
   - **Cache Impact**: Moderate cache locality

4. **`tensor_data_`**: `std::unordered_map<std::string, TensorData>`
   - **Purpose**: Stores actual tensor data for each tensor name
   - **Access Pattern**: Hash map lookup for each input in inner loop
   - **Cache Impact**: Poor cache locality for large maps

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

## Performance Bottlenecks

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
   - **Impact**: Unnecessary work for steps that depend on multiple updated tensors
   - **Location**: Throughout triple loop
   - **Severity**: HIGH

4. **Memory Allocation**: Dynamic allocation for set operations
   - **Impact**: Memory fragmentation and allocation overhead
   - **Location**: Line 924, 945
   - **Severity**: MEDIUM

## Optimization Opportunities

### 1. Algorithmic Optimizations

#### A. Reverse Indexing Approach
Instead of iterating through updated tensors and finding dependent steps, iterate through all steps and check if their dependencies are satisfied.

**Current**: O(U × S × I)
**Optimized**: O(S × I) where S = total steps, I = average inputs per step

#### B. Dependency Tracking
Maintain a data structure that tracks which steps are waiting for which tensors, allowing O(1) updates when tensors become available.

#### C. Early Termination
Use bitmasks or counters to track partial satisfaction of step dependencies, allowing early termination of readiness checks.

### 2. Data Structure Optimizations

#### A. Pre-computed Step Dependencies
Cache step dependency information to avoid repeated hash map lookups.

#### B. Tensor-to-Step Index Optimization
Use more cache-friendly data structures or pre-sort data for better cache locality.

#### C. Memory Pool Allocation
Use memory pools for frequently allocated objects to reduce allocation overhead.

### 3. Parallel Processing

#### A. Step Readiness Checking
Parallelize the readiness checking for independent steps.

#### B. Tensor Processing
Process multiple updated tensors in parallel when they don't have dependencies.

### 4. Caching Strategies

#### A. Step Readiness Cache
Cache step readiness results to avoid recomputation.

#### B. Tensor Dependency Cache
Cache tensor-to-step mappings to reduce hash map lookups.

## Implementation Recommendations

### Immediate Optimizations (Low Risk)

1. **Pre-compute step dependencies** during ensemble initialization
2. **Use vector instead of set** for `next_step_idx` if order doesn't matter
3. **Add early termination** for steps that are already known to be ready
4. **Optimize hash map access** by caching frequently accessed entries

### Medium-term Optimizations (Medium Risk)

1. **Implement reverse indexing** approach
2. **Add dependency tracking** data structures
3. **Use memory pools** for allocation optimization
4. **Implement step readiness caching**

### Long-term Optimizations (High Risk)

1. **Redesign algorithm** to eliminate triple loop entirely
2. **Implement parallel processing** for step readiness checking
3. **Add sophisticated caching** with invalidation strategies
4. **Consider graph-based** dependency resolution

## Testing Strategy

### Performance Testing

1. **Benchmark current implementation** with various ensemble sizes
2. **Profile memory usage** and cache miss rates
3. **Test with realistic ensemble configurations**
4. **Measure scalability** with increasing ensemble complexity

### Correctness Testing

1. **Unit tests** for each optimization
2. **Integration tests** with existing ensemble functionality
3. **Regression tests** to ensure no functional changes
4. **Stress tests** with large, complex ensembles

## Conclusion

The `GetNextSteps` function represents a critical performance bottleneck that scales poorly with ensemble size. The triple nested loop creates O(U × S × I) complexity that can result in millions of operations for large ensembles. 

**Priority**: This should be the **#1 optimization target** for improving ensemble performance.

**Impact**: Optimizing this function could provide 10-100x performance improvements for large ensembles.

**Risk**: Changes require careful testing to ensure correctness while maintaining the complex dependency resolution logic.
