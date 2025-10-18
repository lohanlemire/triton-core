# Cursor Agent Instructions for EnsembleContext::Proceed Research

## Overview

This agent is designed to work with the comprehensive research data in the `EnsembleContext::Proceed research/` folder. The research contains detailed analysis of performance bottlenecks, data structures, and optimization strategies for the Triton Inference Server ensemble scheduler.

## Research Folder Structure

The research is organized in the following structure:

```
EnsembleContext::Proceed research/
├── index.md                           # Main index and overview
├── data_structures.md                 # Input data structures analysis
├── tensor_data_flow.md                # Tensor data flow analysis
├── step_lifecycle.md                  # Step object lifecycle analysis
├── performance_bottlenecks.md         # Detailed bottleneck analysis
├── optimization_recommendations.md    # Specific optimization strategies
└── ensemble_scaling_analysis.md       # Original comprehensive analysis
```

## Agent Responsibilities

### 1. Research Data Usage

When working on ensemble scheduler issues, the agent should:

- **Reference the research**: Always consult the relevant research files before making changes
- **Understand the context**: Use the data structure analysis to understand input forms and expected sizes
- **Identify bottlenecks**: Reference the performance bottleneck analysis to understand scaling issues
- **Apply optimizations**: Use the optimization recommendations as a guide for improvements

### 2. Research Data Updates

When making changes to the ensemble scheduler, the agent should:

- **Update relevant files**: Modify the appropriate research files to reflect changes
- **Maintain accuracy**: Ensure research data remains accurate and up-to-date
- **Document new findings**: Add new insights or discoveries to the research
- **Cross-reference**: Update related files when making changes to one component

### 3. Code Changes and Research Updates

#### When modifying `EnsembleContext::Proceed`:
- Update `data_structures.md` if input parameters change
- Update `tensor_data_flow.md` if data flow patterns change
- Update `performance_bottlenecks.md` if new bottlenecks are identified
- Update `optimization_recommendations.md` if new optimizations are implemented

#### When modifying `UpdateEnsembleState`:
- Update `performance_bottlenecks.md` with new complexity analysis
- Update `optimization_recommendations.md` with new optimization strategies
- Update `tensor_data_flow.md` if tensor processing changes

#### When modifying `GetNextSteps`:
- Update `performance_bottlenecks.md` with new nested loop analysis
- Update `optimization_recommendations.md` with new caching strategies
- Update `step_lifecycle.md` if step processing changes

#### When modifying `Step` class:
- Update `step_lifecycle.md` with new lifecycle patterns
- Update `data_structures.md` with new data structure analysis
- Update `performance_bottlenecks.md` if memory usage changes

## Key Research Findings to Remember

### Critical Performance Issues:
1. **O(N) Tensor Data Iteration** (Lines 901-905 in UpdateEnsembleState)
   - Iterates through ALL tensor data on every step completion
   - **Impact**: 100x-1000x performance degradation for large ensembles
   - **Solution**: Only process actually updated tensors

2. **Triple Nested Loop Complexity** (Lines 926-948 in GetNextSteps)
   - O(U × S × I) complexity where U=updated tensors, S=steps per tensor, I=inputs per step
   - **Impact**: Up to 1,000,000 operations per step for very large ensembles
   - **Solution**: Pre-compute step readiness using caches

3. **Lock Contention** (Line 862 in PrepareSteps)
   - Single mutex serializes all ensemble state updates
   - **Impact**: 20-50% performance degradation for large ensembles
   - **Solution**: Fine-grained locking

### Data Structure Sizes:
- **Small Ensemble** (10-50 tensors): 5-100KB total context
- **Medium Ensemble** (50-200 tensors): 100KB-1MB total context
- **Large Ensemble** (200-1000 tensors): 1-10MB total context
- **Very Large Ensemble** (1000+ tensors): 10MB+ total context

## Agent Workflow

### 1. Before Making Changes
1. Read the relevant research files
2. Understand the current performance characteristics
3. Identify which optimizations are applicable
4. Plan changes to minimize performance impact

### 2. During Implementation
1. Follow the optimization recommendations
2. Implement changes incrementally
3. Test performance impact
4. Document any new findings

### 3. After Making Changes
1. Update the relevant research files
2. Verify accuracy of performance analysis
3. Update optimization recommendations if needed
4. Cross-reference related files

## Research File Update Guidelines

### data_structures.md
- Update when data structures change
- Include new complexity analysis
- Update expected sizes and forms
- Document new data access patterns

### tensor_data_flow.md
- Update when data flow changes
- Include new processing steps
- Update complexity analysis
- Document new memory patterns

### step_lifecycle.md
- Update when step processing changes
- Include new lifecycle phases
- Update performance metrics
- Document new memory usage patterns

### performance_bottlenecks.md
- Update when new bottlenecks are identified
- Include new complexity analysis
- Update performance impact tables
- Document new optimization opportunities

### optimization_recommendations.md
- Update when new optimizations are implemented
- Include new optimization strategies
- Update implementation guidelines
- Document new performance improvements

## Code Change Examples

### Example 1: Optimizing UpdateEnsembleState
```cpp
// Before (O(N) iteration)
for (const auto& tensor_data : tensor_data_) {
  if (!tensor_data.second.tensor_.empty()) {
    updated_tensors->emplace(tensor_data.first, 0);
  }
}

// After (O(1) for common case)
if (completed_step != nullptr) {
  updated_tensors->swap(completed_step->updated_tensors_);
} else {
  // Only for initialization
  for (const auto& [name, data] : tensor_data_) {
    if (!data.tensor_.empty()) {
      updated_tensors->emplace(name, 0);
    }
  }
}
```

**Research Updates Required:**
- Update `performance_bottlenecks.md` to reflect O(1) complexity
- Update `optimization_recommendations.md` to mark this optimization as implemented
- Update `tensor_data_flow.md` to reflect new data flow pattern

### Example 2: Implementing Step Readiness Cache
```cpp
class StepReadinessCache {
  // Implementation details...
};

// Modified GetNextSteps function
Status GetNextSteps(
    const std::set<std::pair<std::string, IterationCount>>& updated_tensors,
    StepList* steps)
{
  // Use cache instead of nested loops
  step_readiness_cache_.UpdateReadiness(updated_tensors, tensor_data_);
  auto ready_steps = step_readiness_cache_.GetReadySteps();
  // ...
}
```

**Research Updates Required:**
- Update `performance_bottlenecks.md` to reflect O(U + S) complexity
- Update `optimization_recommendations.md` to mark this optimization as implemented
- Update `data_structures.md` to include new cache data structure

## Performance Monitoring

The agent should monitor and document:

1. **Algorithmic Complexity Changes**
   - Before and after complexity analysis
   - Performance impact measurements
   - Memory usage changes

2. **Real-World Performance**
   - Small ensemble performance (10-50 tensors)
   - Medium ensemble performance (50-200 tensors)
   - Large ensemble performance (200-1000 tensors)
   - Very large ensemble performance (1000+ tensors)

3. **Memory Usage Patterns**
   - Context memory usage
   - Step memory usage
   - Tensor data memory usage
   - Allocation patterns

## Conclusion

This agent should use the research data as a comprehensive guide for understanding and improving the ensemble scheduler. The research provides detailed analysis of current issues and specific optimization strategies. When making changes, the agent should update the research to maintain its accuracy and usefulness for future development.
