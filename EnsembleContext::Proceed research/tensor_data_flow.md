# Tensor Data Flow Analysis

**[index.md](index.md)**

## Overview

This document analyzes how tensor data flows through the ensemble system, from input to output, and how this flow contributes to the scaling issues in `EnsembleContext::Proceed`.

## Tensor Data Flow Architecture

### 1. Data Flow Overview

```
Input Request → EnsembleContext → Step Processing → Tensor Updates → Next Steps → Output Response
     ↓              ↓                    ↓              ↓              ↓              ↓
  Raw Tensors → tensor_data_ → Step Execution → updated_tensors_ → GetNextSteps → Final Response
```

### 2. Key Data Flow Components

#### Input Data Flow
1. **Initial Request**: Raw tensor data enters via `InferenceRequest`
2. **Tensor Data Population**: Data is stored in `tensor_data_` map
3. **Step Initialization**: Steps are created and scheduled
4. **Execution**: Steps execute and produce outputs
5. **State Updates**: Tensor data is updated with new values
6. **Next Step Determination**: System determines which steps are ready
7. **Output Generation**: Final response is constructed

## Detailed Data Flow Analysis

### 1. Initial Data Population

#### EnsembleContext Constructor (Lines 414-574)
```cpp
// Input tensor processing
for (const auto& pr : lrequest->ImmutableInputs()) {
  const InferenceRequest::Input* input = pr.second;
  auto it = tensor_data_.find(input->Name());
  if (it != tensor_data_.end()) {
    auto& tensor_data = it->second;
    // Create tensor with proper shape and data
    std::unique_ptr<InferenceRequest::Input> tensor;
    if (lrequest->BatchSize() != 0) {
      std::vector<int64_t> shape{lrequest->BatchSize()};
      shape.insert(shape.end(), input->Shape().begin(), input->Shape().end());
      tensor.reset(new InferenceRequest::Input(input->Name(), input->DType(), shape));
    } else {
      tensor.reset(new InferenceRequest::Input(input->Name(), input->DType(), input->Shape()));
    }
    tensor->SetData(input->Data());
    tensor_data.AddTensor(std::move(tensor));
  }
}
```

**Data Flow Characteristics:**
- **Input**: Raw tensor data from inference request
- **Processing**: Shape validation, batch size handling, data copying
- **Output**: Populated `tensor_data_` map
- **Complexity**: O(N) where N = number of input tensors
- **Memory Impact**: Creates new tensor objects for each input

### 2. Step Execution and Tensor Updates

#### ConsumeResponse Function (Lines 697-843)
```cpp
// Process response outputs
for (uint32_t idx = 0; idx < count; idx++) {
  // Extract output data
  RETURN_IF_TRITONSERVER_ERROR(TRITONSERVER_InferenceResponseOutput(
      response, idx, &name, &datatype, &shape, &dim_count, &base,
      &byte_size, &memory_type, &memory_type_id, &userp));
  
  auto it = output_to_tensor.find(name);
  if (it != output_to_tensor.end()) {
    // Create new tensor object
    std::unique_ptr<InferenceRequest::Input> tensor(
        new InferenceRequest::Input(it->second, TritonToDataType(datatype), shape, dim_count));
    
    // Handle memory allocation
    if (byte_size != 0) {
      std::lock_guard<std::mutex> output_lk(step_ptr->output_mtx_);
      if (memory_type == TRITONSERVER_MEMORY_GPU) {
        auto& gpu_output_map = step_ptr->gpu_output_map_[memory_type_id];
        auto it = gpu_output_map.find(reinterpret_cast<uintptr_t>(base));
        tensor->SetData(std::move(it->second));
        gpu_output_map.erase(it);
      } else {
        auto it = step_ptr->cpu_output_map_.find(reinterpret_cast<uintptr_t>(base));
        tensor->SetData(std::move(it->second));
        step_ptr->cpu_output_map_.erase(it);
      }
    }
    
    // Add to tensor data
    auto& tensor_data = tensor_data_[it->second];
    step_ptr->updated_tensors_.emplace(
        it->second, tensor_data.AddTensor(std::move(tensor), correlation_id, flags));
  }
}
```

**Data Flow Characteristics:**
- **Input**: Step response with output tensors
- **Processing**: Memory management, tensor object creation, data copying
- **Output**: Updated `tensor_data_` and `updated_tensors_` set
- **Complexity**: O(O) where O = number of outputs per step
- **Memory Impact**: Creates new tensor objects and manages memory allocation

### 3. State Update and Next Step Determination

#### UpdateEnsembleState Function (Lines 895-915)
```cpp
Status UpdateEnsembleState(
    const std::unique_ptr<Step>& completed_step,
    std::set<std::pair<std::string, IterationCount>>* updated_tensors)
{
  updated_tensors->clear();
  if (completed_step == nullptr) {
    // 🔥 CRITICAL BOTTLENECK: O(N) iteration through ALL tensor data
    for (const auto& tensor_data : tensor_data_) {
      if (!tensor_data.second.tensor_.empty()) {
        updated_tensors->emplace(tensor_data.first, 0);
      }
    }
  } else {
    if (completed_step->response_flags_ & TRITONSERVER_RESPONSE_COMPLETE_FINAL) {
      inflight_step_counter_--;
    }
    RETURN_IF_ERROR(ConsumeResponse(completed_step));
    updated_tensors->swap(completed_step->updated_tensors_);
  }
  return Status::Success;
}
```

**Data Flow Characteristics:**
- **Input**: Completed step or nullptr for initialization
- **Processing**: 
  - If nullptr: O(N) iteration through ALL tensor data
  - If step: Process step response and get updated tensors
- **Output**: Set of updated tensor names and iteration counts
- **Complexity**: O(N) for initialization, O(1) for step completion
- **Memory Impact**: Creates set of updated tensor pairs

#### GetNextSteps Function (Lines 918-957)
```cpp
Status GetNextSteps(
    const std::set<std::pair<std::string, IterationCount>>& updated_tensors,
    StepList* steps)
{
  steps->clear();
  std::set<std::pair<size_t, IterationCount>> next_step_idx;
  
  // 🔥 CRITICAL BOTTLENECK: Triple nested loop
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
          // Check iteration count matching
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
  
  // Create step objects
  for (const auto& idx : next_step_idx) {
    steps->emplace_back();
    RETURN_IF_ERROR(InitStep(idx.first, idx.second, &(steps->back())));
  }
  inflight_step_counter_ += steps->size();
  
  return Status::Success;
}
```

**Data Flow Characteristics:**
- **Input**: Set of updated tensor names and iteration counts
- **Processing**: Triple nested loop to determine ready steps
- **Output**: List of ready steps to execute
- **Complexity**: O(U × S × I) where U=updated tensors, S=steps per tensor, I=inputs per step
- **Memory Impact**: Creates new step objects

### 4. Output Generation

#### CheckAndSetEnsembleOutput Function (Lines 1238-1387)
```cpp
Status CheckAndSetEnsembleOutput(
    const std::set<std::pair<std::string, IterationCount>>& updated_tensors,
    std::unique_ptr<InferenceResponse>* response)
{
  // Check if all outputs are ready
  for (const auto& updated_tensor : updated_tensors) {
    if (requested_outputs.find(updated_tensor.first) == requested_outputs.end()) {
      continue;
    }
    
    ready = true;
    iteration_count = updated_tensor.second;
    for (const auto& output : requested_outputs) {
      auto& tensor = tensor_data_[output].tensor_;
      if (tensor.empty()) {
        ready = false;
        break;
      } else if (tensor.find(iteration_count) == tensor.end()) {
        ready = false;
        break;
      }
    }
  }
  
  if (ready) {
    // Create response and copy tensor data
    RETURN_IF_ERROR(lrequest->ResponseFactory()->CreateResponse(response));
    
    for (const auto& output_pair : info_->ensemble_output_shape_) {
      auto& tensor_data = tensor_data_[output_pair.first];
      auto& tensor = tensor_data.tensor_[iteration_count];
      
      // Copy tensor data to response
      // ... memory copying logic ...
    }
  }
  
  return Status::Success;
}
```

**Data Flow Characteristics:**
- **Input**: Set of updated tensor names and iteration counts
- **Processing**: Check output readiness, create response, copy tensor data
- **Output**: Final inference response
- **Complexity**: O(O) where O = number of outputs
- **Memory Impact**: Creates response object and copies tensor data

## Data Flow Scaling Issues

### 1. O(N) Tensor Data Iteration

**Location**: `UpdateEnsembleState` lines 901-905
```cpp
for (const auto& tensor_data : tensor_data_) {
  if (!tensor_data.second.tensor_.empty()) {
    updated_tensors->emplace(tensor_data.first, 0);
  }
}
```

**Problem**: Iterates through ALL tensor data entries regardless of how many were actually updated.

**Impact**: 
- Small ensemble (10 tensors): 10 iterations per step completion
- Medium ensemble (100 tensors): 100 iterations per step completion
- Large ensemble (1000 tensors): 1000 iterations per step completion

**Solution**: Only process actually updated tensors from the completed step.

### 2. Triple Nested Loop Complexity

**Location**: `GetNextSteps` lines 926-948
```cpp
for (const auto& updated_tensor : updated_tensors) {           // U iterations
  const auto& step_idx = (*tensor_to_step_)[updated_tensor.first];
  for (const auto& idx : step_idx) {                          // S iterations
    for (const auto& input_pair : info_->steps_[idx].input_to_tensor_) { // I iterations
      // Tensor lookup and iteration count checking
    }
  }
}
```

**Problem**: O(U × S × I) complexity where:
- U = number of updated tensors
- S = average number of steps per tensor
- I = average number of inputs per step

**Impact**:
- Small ensemble: ~10 × 5 × 3 = 150 operations
- Medium ensemble: ~50 × 20 × 5 = 5,000 operations
- Large ensemble: ~200 × 100 × 10 = 200,000 operations

**Solution**: Pre-compute step readiness or use more efficient data structures.

### 3. Memory Allocation Overhead

**Location**: Multiple locations in tensor data flow
```cpp
// In ConsumeResponse
std::unique_ptr<InferenceRequest::Input> tensor(
    new InferenceRequest::Input(it->second, TritonToDataType(datatype), shape, dim_count));

// In GetNextSteps
steps->emplace_back();
RETURN_IF_ERROR(InitStep(idx.first, idx.second, &(steps->back())));

// In CheckAndSetEnsembleOutput
RETURN_IF_ERROR(lrequest->ResponseFactory()->CreateResponse(response));
```

**Problem**: New objects are allocated for every tensor and step.

**Impact**: Memory allocation overhead scales with number of tensors and steps.

**Solution**: Use object pools or more efficient memory management.

## Data Flow Optimization Strategies

### 1. Eliminate O(N) Iteration

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

### 3. Reduce Memory Allocation

```cpp
// Use object pools for frequently allocated objects
class TensorObjectPool {
  std::queue<std::unique_ptr<InferenceRequest::Input>> available_tensors_;
  std::mutex pool_mutex_;
  
public:
  std::unique_ptr<InferenceRequest::Input> Acquire() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (available_tensors_.empty()) {
      return std::make_unique<InferenceRequest::Input>();
    }
    auto tensor = std::move(available_tensors_.front());
    available_tensors_.pop();
    return tensor;
  }
  
  void Release(std::unique_ptr<InferenceRequest::Input> tensor) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    available_tensors_.push(std::move(tensor));
  }
};
```

## Conclusion

The tensor data flow in the ensemble system has several critical scaling issues:

1. **Primary Issue**: O(N) iteration through all tensor data on every step completion
2. **Secondary Issue**: Triple nested loop complexity in step readiness checking
3. **Memory Issue**: Frequent allocation of tensor and step objects

The most impactful optimization would be to eliminate the full tensor data iteration and only process actually updated tensors, changing the complexity from O(N) to O(1) for the common case. This would significantly improve performance for ensembles with many inputs.
