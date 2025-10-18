# Triton Ensemble Scheduler Scaling Analysis

**[index.md](index.md)**

## Executive Summary

This report analyzes the scaling issues in the Triton Inference Server's ensemble scheduler, specifically focusing on the `EnsembleContext::Proceed` and `PrepareSteps` functions. The analysis reveals several critical performance bottlenecks that cause poor scaling when models have many inputs.

## Selected Code Analysis

### Functions Under Analysis

#### 1. `EnsembleContext::Proceed` (Lines 846-855)
```cpp
void EnsembleContext::Proceed(
    const std::shared_ptr<EnsembleContext>& context,
    const std::unique_ptr<Step>& completed_step)
{
  StepList ready_steps;
  Status status = context->PrepareSteps(completed_step, &ready_steps);
  if (status.IsOk()) {
    ScheduleSteps(context, std::move(ready_steps));
  }
}
```

**Purpose**: This is the main entry point for ensemble state transitions. It processes a completed step and determines what steps should be executed next.

#### 2. `EnsembleContext::PrepareSteps` (Lines 857-892)
```cpp
Status EnsembleContext::PrepareSteps(
    const std::unique_ptr<Step>& completed_step, StepList* ready_steps)
{
  {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (ensemble_status_.IsOk()) {
      StepList res;
      std::set<std::pair<std::string, IterationCount>> updated_tensors;
      ensemble_status_ = UpdateEnsembleState(completed_step, &updated_tensors);
      if (ensemble_status_.IsOk()) {
        ensemble_status_ = GetNextSteps(updated_tensors, ready_steps);
      }
      
      if (!ensemble_status_.IsOk()) {
        ensemble_status_ = FinishEnsemble();
      } else {
        std::unique_ptr<InferenceResponse> response;
        if (!updated_tensors.empty()) {
          ensemble_status_ = CheckAndSetEnsembleOutput(updated_tensors, &response);
        }
        ensemble_status_ = FinishEnsemble(std::move(response));
      }
    }
    return ensemble_status_;
  }
}
```

**Purpose**: This function orchestrates the ensemble execution by:
1. Updating ensemble state based on completed steps
2. Determining which steps are ready to execute
3. Checking if ensemble outputs are ready
4. Finishing the ensemble when appropriate

## Critical Scaling Bottlenecks

### 1. **O(N) Tensor Data Iteration in UpdateEnsembleState** ⚠️ **CRITICAL**

**Location**: Lines 901-905 in `UpdateEnsembleState`
```cpp
for (const auto& tensor_data : tensor_data_) {
  if (!tensor_data.second.tensor_.empty()) {
    updated_tensors->emplace(tensor_data.first, 0);
  }
}
```

**Problem**: This iterates through ALL tensor data entries every time a step completes, regardless of how many tensors were actually updated. With many inputs, this becomes a significant bottleneck.

**Impact**: O(N) complexity where N = total number of tensors in the ensemble, executed on every step completion.

### 2. **Nested Loop Complexity in GetNextSteps** ⚠️ **CRITICAL**

**Location**: Lines 926-948 in `GetNextSteps`
```cpp
for (const auto& updated_tensor : updated_tensors) {
  const auto& step_idx = (*tensor_to_step_)[updated_tensor.first];
  for (const auto& idx : step_idx) {
    bool ready = true;
    for (const auto& input_pair : info_->steps_[idx].input_to_tensor_) {
      auto& tensor = tensor_data_[input_pair.second].tensor_;
      if (tensor.empty()) {
        ready = false;
        break;
      } else {
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

**Problem**: Triple nested loop with O(U × S × I) complexity where:
- U = number of updated tensors
- S = average number of steps per tensor
- I = average number of inputs per step

**Impact**: With many inputs, this becomes O(N³) in worst case scenarios.

### 3. **Frequent Tensor Data Lookups** ⚠️ **HIGH**

**Location**: Multiple locations using `tensor_data_.find()`
```cpp
auto it = tensor_data_.find(input->Name());
auto& tensor_data = tensor_data_[input_pair.second];
```

**Problem**: Hash map lookups are performed repeatedly for the same tensors during step processing.

**Impact**: O(1) per lookup but performed many times, accumulating significant overhead.

### 4. **Memory Allocation Overhead in ConsumeResponse** ⚠️ **MEDIUM**

**Location**: Lines 795-797 in `ConsumeResponse`
```cpp
std::unique_ptr<InferenceRequest::Input> tensor(
    new InferenceRequest::Input(
        it->second, TritonToDataType(datatype), shape, dim_count));
```

**Problem**: New tensor objects are allocated for every output from every step.

**Impact**: Memory allocation overhead scales with number of outputs and steps.

### 5. **Lock Contention in PrepareSteps** ⚠️ **MEDIUM**

**Location**: Line 862 in `PrepareSteps`
```cpp
std::lock_guard<std::mutex> lock(mutex_);
```

**Problem**: The entire `PrepareSteps` function is under a single mutex, creating a serialization bottleneck.

**Impact**: All ensemble state updates are serialized, limiting parallelism.

## Data Structure Analysis

### Key Data Structures

#### 1. `tensor_data_` (Line 378)
```cpp
std::unordered_map<std::string, TensorData> tensor_data_;
```
- **Purpose**: Stores tensor data for each ensemble tensor
- **Scaling Issue**: Linear iteration through all entries in `UpdateEnsembleState`

#### 2. `tensor_to_step_` (Line 375)
```cpp
std::unordered_map<std::string, std::set<size_t>>* tensor_to_step_;
```
- **Purpose**: Maps tensor names to sets of step indices that use them
- **Scaling Issue**: Used in nested loops in `GetNextSteps`

#### 3. `TensorData::tensor_` (Line 242)
```cpp
std::unordered_map<IterationCount, Metadata> tensor_;
```
- **Purpose**: Stores tensor metadata indexed by iteration count
- **Scaling Issue**: Hash map lookups for iteration count matching

## Performance Impact Analysis

### Scaling Characteristics

1. **Linear Scaling Issues**:
   - `UpdateEnsembleState` tensor iteration: O(N)
   - Tensor data lookups: O(N) total operations

2. **Quadratic/Cubic Scaling Issues**:
   - `GetNextSteps` nested loops: O(U × S × I) to O(N³)
   - Step readiness checking: O(S × I) per updated tensor

3. **Memory Scaling Issues**:
   - Tensor object allocation: O(O) where O = number of outputs
   - Hash map memory overhead: O(N) for tensor_data_

### Real-World Impact

For an ensemble with:
- 100 input tensors
- 50 steps
- Average 10 inputs per step

The complexity becomes:
- `UpdateEnsembleState`: 100 iterations per step completion
- `GetNextSteps`: Up to 100 × 50 × 10 = 50,000 operations per step completion
- Total: Significant CPU overhead that grows quadratically with input count

## Recommended Optimizations

### 1. **Optimize UpdateEnsembleState** (High Priority)
```cpp
// Instead of iterating all tensors, only process actually updated ones
if (completed_step != nullptr) {
  // Only process tensors that were actually updated
  updated_tensors->swap(completed_step->updated_tensors_);
} else {
  // Only for initialization - use a more efficient approach
  for (const auto& [name, data] : tensor_data_) {
    if (!data.tensor_.empty()) {
      updated_tensors->emplace(name, 0);
    }
  }
}
```

### 2. **Optimize GetNextSteps** (High Priority)
```cpp
// Pre-compute step readiness using bitmasks or more efficient data structures
// Cache step readiness state to avoid recomputation
// Use incremental updates instead of full recalculation
```

### 3. **Reduce Lock Granularity** (Medium Priority)
```cpp
// Use fine-grained locking instead of single mutex
// Lock only specific tensor data during updates
// Use read-write locks for read-heavy operations
```

### 4. **Memory Pool for Tensor Objects** (Medium Priority)
```cpp
// Use object pools for frequently allocated tensor objects
// Reduce memory allocation overhead
```

### 5. **Caching and Memoization** (Low Priority)
```cpp
// Cache step readiness calculations
// Memoize tensor lookup results
// Use more efficient data structures for frequent operations
```

## Conclusion

The ensemble scheduler has several critical scaling bottlenecks that cause poor performance with many inputs:

1. **Primary Issue**: O(N) iteration through all tensor data on every step completion
2. **Secondary Issue**: O(N³) nested loop complexity in step readiness checking
3. **Tertiary Issues**: Lock contention, memory allocation overhead, and frequent hash map lookups

These issues compound to create significant performance degradation as the number of inputs increases. The recommended optimizations focus on eliminating unnecessary iterations, reducing algorithmic complexity, and improving data structure efficiency.

The most impactful fix would be to eliminate the full tensor data iteration in `UpdateEnsembleState` and only process actually updated tensors, which would change the complexity from O(N) to O(1) for the common case.
