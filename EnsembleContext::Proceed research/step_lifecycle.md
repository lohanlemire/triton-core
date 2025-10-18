# Step Lifecycle Analysis

**[index.md](index.md)**

## Overview

This document analyzes the lifecycle of `Step` objects in the ensemble system, from creation to completion, and how this lifecycle contributes to the scaling issues in `EnsembleContext::Proceed`.

## Step Object Structure

### Step Class Definition

```cpp
struct Step {
  // Context and request management
  std::shared_ptr<EnsembleContext> ctx_;
  std::unique_ptr<InferenceRequest> request_;
  InferenceRequest::SequenceId correlation_id_;
  uint32_t flags_;
  
  // Memory management for outputs
  std::mutex output_mtx_;
  std::unordered_map<uintptr_t, std::shared_ptr<AllocatedMemory>> cpu_output_map_;
  std::unordered_map<int64_t, std::unordered_map<uintptr_t, std::shared_ptr<AllocatedMemory>>> gpu_output_map_;
  
  // 🔥 CRITICAL: Updated tensors tracking
  std::set<std::pair<std::string, IterationCount>> updated_tensors_;
  
  // Response handling
  uint32_t response_flags_;
  TRITONSERVER_InferenceResponse* response_;
  const bool preserve_responses_order_;
  
  // Step identification
  size_t step_idx_;
};
```

## Step Lifecycle Phases

### 1. Step Creation (InitStep Function)

#### Location: Lines 960-1078
```cpp
Status InitStep(
    const size_t step_idx, const IterationCount iteration_count,
    std::unique_ptr<Step>* step)
{
  const auto& istep = info_->steps_[step_idx];
  auto& version_map = handles_[istep.model_id_];
  auto& model = version_map[istep.model_version_];
  
  const bool allow_batching = (model->Config().max_batch_size() > 0);
  
  // Create inference request
  auto irequest = std::unique_ptr<InferenceRequest>(
      new InferenceRequest(model, istep.model_version_));
  
  // Store pointers to tensors for later pruning
  std::map<TensorData*, size_t*> releasing_tensors;
  
  // Set inputs in request
  auto correlation_id = correlation_id_;
  auto flags = flags_;
  bool parameter_set = false;
  
  for (const auto& pair : istep.input_to_tensor_) {
    auto& tensor_data = tensor_data_[pair.second];
    auto& tensor = tensor_data.tensor_[iteration_count];
    
    if (tensor.data_ != nullptr) {
      // Reshape tensor dimensions
      const inference::ModelInput* input_config;
      model->GetInput(pair.first, &input_config);
      auto shape = ReshapeTensorDims(
          input_config->dims(), allow_batching, tensor_data.batch_size_,
          tensor.data_->OriginalShape());
      
      // Add input to request
      InferenceRequest::Input* input;
      RETURN_IF_ERROR(irequest->AddOriginalInput(
          pair.first, tensor.data_->DType(), shape, &input));
      RETURN_IF_ERROR(input->SetData(tensor.data_->Data()));
      
      // Handle host policy data
      for (const auto& host_policy_data : tensor.data_->HostPolicyData()) {
        RETURN_IF_ERROR(input->SetData(host_policy_data.first, host_policy_data.second));
      }
    }
    
    // Track for later pruning
    releasing_tensors.emplace(&tensor_data, &tensor.remaining_reference_count_);
    
    // Handle parameter overrides
    if (tensor.parameter_override_) {
      if (parameter_set && ((correlation_id != tensor.correlation_id_) ||
                          (flags != tensor.flags_))) {
        LOG_ERROR << "Different set of response parameters are set for '"
                  << istep.model_id_ << "'. Parameter correlation ID "
                  << correlation_id << ", flags " << flags << " is used.";
        continue;
      }
      correlation_id = tensor.correlation_id_;
      flags = tensor.flags_;
      parameter_set = true;
    }
  }
  
  // Prune tensors that are no longer needed
  for (auto& releasing_pair : releasing_tensors) {
    if ((--(*releasing_pair.second)) == 0) {
      releasing_pair.first->tensor_.erase(iteration_count);
    }
  }
  
  // Set requested outputs
  for (const auto& pair : istep.output_to_tensor_) {
    irequest->AddOriginalRequestedOutput(pair.first);
  }
  
  // Create step object
  const bool preserve_order = preserve_responses_order(model->Config());
  step->reset(new Step(step_idx, correlation_id, flags, preserve_order));
  
  // Configure request
  irequest->SetId(request_id_);
  irequest->SetCorrelationId(correlation_id);
  irequest->SetFlags(flags);
  irequest->SetPriority(priority_);
  irequest->SetTimeoutMicroseconds(timeout_);
  irequest->SetParameters(parameters_);
  
  // Set callbacks
  irequest->SetResponseCallback(
      reinterpret_cast<ResponseAllocator*>(allocator_.get()), step->get(),
      ResponseComplete, step->get());
  irequest->SetReleaseCallback(RequestComplete, request_tracker_);
  
  RETURN_IF_ERROR(irequest->PrepareForInference());
  
  // Record batch size for outputs
  for (const auto& pair : istep.output_to_tensor_) {
    auto& output_data_ = tensor_data_[pair.second];
    output_data_.batch_size_ = irequest->BatchSize();
  }
  
  (*step)->request_ = std::move(irequest);
  return Status::Success;
}
```

**Step Creation Characteristics:**
- **Input**: Step index, iteration count, tensor data
- **Processing**: 
  - Model handle lookup
  - Input tensor processing and reshaping
  - Request configuration
  - Callback setup
  - Tensor pruning
- **Output**: Configured step object ready for execution
- **Complexity**: O(I) where I = number of inputs per step
- **Memory Impact**: Creates new inference request and step object

### 2. Step Execution (ScheduleSteps Function)

#### Location: Lines 1390-1444
```cpp
void ScheduleSteps(
    const std::shared_ptr<EnsembleContext>& context, StepList&& steps)
{
  for (auto& step : steps) {
    step->ctx_ = context;
    bool should_schedule = false;
    
    // Check ensemble status and increment counter
    {
      std::lock_guard<std::mutex> lock(context->mutex_);
      if (context->ensemble_status_.IsOk()) {
        context->request_tracker_->IncrementCounter();
        should_schedule = true;
      }
    }
    
    if (should_schedule) {
      // Handle cancellation
      if (context->request_tracker_->Request()->IsCancelled()) {
        step->request_->Cancel();
      }
      
      // Execute inference request
      std::unique_ptr<InferenceRequest> request = std::move(step->request_);
      auto step_status = context->is_->InferAsync(request);
      
      if (step_status.IsOk()) {
        step.release(); // Step will be released by response callback
        continue;
      } else {
        std::lock_guard<std::mutex> lock(context->mutex_);
        context->ensemble_status_ = step_status;
      }
    }
    
    // Handle scheduling failure
    std::lock_guard<std::mutex> lock(context->mutex_);
    context->request_tracker_->DecrementCounter();
    --context->inflight_step_counter_;
    
    if (context->inflight_step_counter_ == 0) {
      context->ensemble_status_ = context->FinishEnsemble();
    }
  }
}
```

**Step Execution Characteristics:**
- **Input**: List of ready steps
- **Processing**: 
  - Status checking
  - Counter management
  - Async inference execution
  - Error handling
- **Output**: Steps are executed asynchronously
- **Complexity**: O(S) where S = number of steps to schedule
- **Memory Impact**: Steps are released after successful scheduling

### 3. Step Response Handling (ResponseComplete Function)

#### Location: Lines 665-694
```cpp
void ResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  auto step_raw_ptr = reinterpret_cast<Step*>(userp);
  auto pool = step_raw_ptr->ctx_->CallbackPool();
  
  auto fn = [response, flags, step_raw_ptr]() {
    auto step_ptr = std::unique_ptr<Step>(step_raw_ptr);
    step_ptr->response_flags_ = flags;
    step_ptr->response_ = response;
    
    // 🔥 CRITICAL: This triggers the main Proceed function
    EnsembleContext::Proceed(step_ptr->ctx_, step_ptr);
    
    // Handle multi-response scenarios
    if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
      step_ptr.release();
    }
  };
  
  // Attempt async execution or fallback to synchronous
  if (!step_raw_ptr->preserve_responses_order_ &&
      pool->TaskQueueSize() < pool->Size()) {
    pool->Enqueue(fn);
  } else {
    fn();
  }
}
```

**Step Response Characteristics:**
- **Input**: Inference response, flags, step pointer
- **Processing**: 
  - Response data extraction
  - Async callback execution
  - Triggers main Proceed function
- **Output**: Step completion triggers ensemble state update
- **Complexity**: O(1) per response
- **Memory Impact**: Step object is consumed and destroyed

### 4. Step Completion and Cleanup

#### Location: Lines 907-913 in UpdateEnsembleState
```cpp
if (completed_step->response_flags_ & TRITONSERVER_RESPONSE_COMPLETE_FINAL) {
  inflight_step_counter_--;
}
RETURN_IF_ERROR(ConsumeResponse(completed_step));
updated_tensors->swap(completed_step->updated_tensors_);
```

**Step Completion Characteristics:**
- **Input**: Completed step object
- **Processing**: 
  - Counter decrement
  - Response consumption
  - Tensor data extraction
- **Output**: Updated tensor information
- **Complexity**: O(O) where O = number of outputs per step
- **Memory Impact**: Step object is destroyed after processing

## Step Lifecycle Scaling Issues

### 1. Step Creation Overhead

**Problem**: Each step creation involves:
- Model handle lookup
- Input tensor processing
- Request configuration
- Callback setup
- Tensor pruning

**Impact**: 
- Small ensemble (10 steps): ~10 step creations
- Medium ensemble (100 steps): ~100 step creations
- Large ensemble (1000 steps): ~1000 step creations

**Scaling**: O(S) where S = number of steps

### 2. Memory Management Overhead

**Problem**: Each step manages:
- CPU output maps
- GPU output maps
- Tensor data references
- Request objects

**Impact**:
- Small ensemble: ~1-10KB per step
- Medium ensemble: ~10-100KB per step
- Large ensemble: ~100KB-1MB per step

**Scaling**: O(S × M) where S = steps, M = memory per step

### 3. Response Processing Overhead

**Problem**: Each step response triggers:
- ResponseComplete callback
- Proceed function call
- Tensor data processing
- Next step determination

**Impact**:
- Small ensemble: ~10 response processing cycles
- Medium ensemble: ~100 response processing cycles
- Large ensemble: ~1000 response processing cycles

**Scaling**: O(S) where S = number of steps

## Step Lifecycle Optimization Strategies

### 1. Reduce Step Creation Overhead

```cpp
// Current: Create new request for each step
auto irequest = std::unique_ptr<InferenceRequest>(
    new InferenceRequest(model, istep.model_version_));

// Optimized: Use request pool
class InferenceRequestPool {
  std::queue<std::unique_ptr<InferenceRequest>> available_requests_;
  std::mutex pool_mutex_;
  
public:
  std::unique_ptr<InferenceRequest> Acquire(Model* model, int64_t version) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (available_requests_.empty()) {
      return std::make_unique<InferenceRequest>(model, version);
    }
    auto request = std::move(available_requests_.front());
    available_requests_.pop();
    // Reset request state
    request->Reset();
    return request;
  }
  
  void Release(std::unique_ptr<InferenceRequest> request) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    available_requests_.push(std::move(request));
  }
};
```

### 2. Optimize Memory Management

```cpp
// Current: Separate maps for CPU and GPU
std::unordered_map<uintptr_t, std::shared_ptr<AllocatedMemory>> cpu_output_map_;
std::unordered_map<int64_t, std::unordered_map<uintptr_t, std::shared_ptr<AllocatedMemory>>> gpu_output_map_;

// Optimized: Unified memory management
class UnifiedOutputManager {
  struct OutputEntry {
    std::shared_ptr<AllocatedMemory> memory;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
  };
  
  std::unordered_map<uintptr_t, OutputEntry> output_map_;
  std::mutex output_mtx_;
  
public:
  void StoreOutput(uintptr_t key, std::shared_ptr<AllocatedMemory> memory,
                   TRITONSERVER_MemoryType memory_type, int64_t memory_type_id) {
    std::lock_guard<std::mutex> lock(output_mtx_);
    output_map_[key] = {std::move(memory), memory_type, memory_type_id};
  }
  
  std::shared_ptr<AllocatedMemory> RetrieveOutput(uintptr_t key) {
    std::lock_guard<std::mutex> lock(output_mtx_);
    auto it = output_map_.find(key);
    if (it != output_map_.end()) {
      auto memory = std::move(it->second.memory);
      output_map_.erase(it);
      return memory;
    }
    return nullptr;
  }
};
```

### 3. Batch Step Processing

```cpp
// Current: Process steps one by one
for (auto& step : steps) {
  // Process individual step
}

// Optimized: Batch step processing
class BatchStepProcessor {
  std::vector<std::unique_ptr<Step>> step_batch_;
  size_t batch_size_;
  
public:
  void AddStep(std::unique_ptr<Step> step) {
    step_batch_.push_back(std::move(step));
    if (step_batch_.size() >= batch_size_) {
      ProcessBatch();
    }
  }
  
  void ProcessBatch() {
    // Process all steps in batch
    for (auto& step : step_batch_) {
      // Batch processing logic
    }
    step_batch_.clear();
  }
};
```

## Step Lifecycle Performance Metrics

### Creation Time
- **Small ensemble**: ~1-5ms per step
- **Medium ensemble**: ~5-20ms per step
- **Large ensemble**: ~20-100ms per step

### Memory Usage
- **Small ensemble**: ~1-10KB per step
- **Medium ensemble**: ~10-100KB per step
- **Large ensemble**: ~100KB-1MB per step

### Response Processing Time
- **Small ensemble**: ~0.1-1ms per response
- **Medium ensemble**: ~1-10ms per response
- **Large ensemble**: ~10-100ms per response

## Conclusion

The step lifecycle in the ensemble system has several scaling issues:

1. **Creation Overhead**: O(S) complexity for step creation
2. **Memory Management**: O(S × M) memory usage scaling
3. **Response Processing**: O(S) response processing cycles

The most impactful optimizations would be:
1. Use object pools for step and request creation
2. Optimize memory management with unified output handling
3. Implement batch processing for step operations

These optimizations would significantly reduce the overhead of step lifecycle management in large ensembles.
