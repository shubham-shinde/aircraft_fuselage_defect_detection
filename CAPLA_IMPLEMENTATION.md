# CAPLA (Class-Aware Adaptive Pseudolabel Assignment) Implementation

## Overview
This document describes the implementation of the CAPLA strategy in the `trainer.py` file for semi-supervised object detection. CAPLA addresses label distribution mismatch between labeled and unlabeled data by dynamically adjusting confidence thresholds on a per-class basis.

## Key Components Implemented

### 1. **CAPLA Parameters Initialization**
- **Location**: `SemiDetectionTrainer.__init__()` method
- **Parameters**:
  - `capla_reliable_threshold`: 0.3 (δ% - proportion of reliable pseudolabels)
  - `capla_update_interval`: 2000 (update thresholds every K iterations)
  - `class_adaptive_thresholds`: Dictionary storing class-specific thresholds
  - `pseudolabel_scores_per_class`: Tracks confidence scores for each class
  - `iteration_count`: Global iteration counter for periodic updates
  - `total_labeled_samples`: N^l (total labeled data)
  - `total_unlabeled_samples`: N^u (total unlabeled data)

### 2. **Dynamic Threshold Calculation**
- **Method**: `calculate_class_adaptive_thresholds()`
- **Formula**: $t_c = P_c^l \cdot \delta\% \cdot n_c^l \cdot \frac{N^u}{N^l}$
- **Parameters**:
  - $P_c^l$: 30th percentile of sorted pseudolabel scores for class c
  - $\delta\%$: 0.3 (30% reliable threshold)
  - $n_c^l$: Number of ground truth labels for class c
  - $\frac{N^u}{N^l}$: Ratio of unlabeled to labeled samples
- **Clipping**: Thresholds are clipped between 0.3 and 0.95 to maintain reasonable bounds

### 3. **Pseudolabel Categorization**
- **Method**: `categorize_pseudolabels()`
- **Categories**:
  - **Reliable**: Confidence score > $t_c$ → Used for standard supervised training
  - **Uncertain**: 0.5 < confidence score ≤ $t_c$ → Used with reduced confidence weight
- **Output**: Separate masks and indices for reliable and uncertain pseudolabels

### 4. **Enhanced Pseudolabel Combination**
- **Method**: `combine_labeled_unlabeled()` (modified)
- **Process**:
  1. Apply CAPLA categorization to teacher predictions
  2. Track confidence scores for each class
  3. Perform NMS separately on reliable and uncertain predictions
  4. Combine labeled data with:
     - Reliable pseudolabels (full weight)
     - Uncertain pseudolabels (0.5× weight)
  5. Store reliability information for loss weighting

### 5. **Periodic Threshold Updates**
- **Location**: `_do_train()` method, training loop
- **Frequency**: Every 2000 iterations (configurable via `capla_update_interval`)
- **Action on Update**:
  1. Recalculate adaptive thresholds based on accumulated scores
  2. Log new thresholds to console
  3. Reset score tracking for next update period

### 6. **CAPLA Initialization in Training Setup**
- **Method**: `_setup_train()` (modified)
- **Initializations**:
  1. Create per-class score tracking dictionary
  2. Count labeled and unlabeled sample sizes
  3. Calculate initial adaptive thresholds
  4. Log initialization information

## Data Flow

```
Teacher Model Predictions
           ↓
   Categorization using
   Adaptive Thresholds (t_c)
           ↓
    ┌──────┴──────┐
    ↓             ↓
Reliable      Uncertain
Pseudolabels  Pseudolabels
(score > t_c) (0.5 < score ≤ t_c)
    ↓             ↓
    └──────┬──────┘
           ↓
    Combined Batch
    (with reliability info)
           ↓
   Student Model Training
    (different loss weights)
```

## Loss Weighting Strategy

- **Reliable pseudolabels**: Full loss contribution
- **Uncertain pseudolabels**: 0.5× loss contribution (reduced weight)
- **Labeled data**: Standard loss (unaffected)

## Addressing Class Imbalance

For imbalanced datasets (e.g., common "scratch" vs. rare "rivet-damage"):

1. **Dynamic thresholding** ensures:
   - Rare classes: Lower threshold → More pseudolabels accepted
   - Common classes: Higher threshold → Stricter filtering

2. **Formula adaptations**:
   - $n_c^l$: Directly incorporates class frequency
   - $\frac{N^u}{N^l}$: Scales threshold based on unlabeled data proportion

## Monitoring and Logging

The implementation logs:
- Initial CAPLA configuration
- Adaptive thresholds at each update (every 2000 iterations)
- Labeled vs. unlabeled data statistics

Example log output:
```
Initialized CAPLA with adaptive thresholds: {0: 0.65, 1: 0.72, 2: 0.58, ...}
Labeled samples: 1024, Unlabeled samples: 5000
Updated CAPLA thresholds at iteration 2000: {0: 0.68, 1: 0.74, 2: 0.61, ...}
```

## Configuration Parameters

In your training script, you can customize:

```python
# In trainer.py initialization
trainer.capla_reliable_threshold = 0.3    # Change δ%
trainer.capla_update_interval = 2000      # Change update frequency K
```

## Integration Notes

1. **Teacher-Student Framework**: Works seamlessly with EMA-based teacher updates
2. **Loss Calculation**: Pseudolabel reliability is stored in batch for potential downstream loss weighting
3. **DDP Support**: Threshold updates are synchronized across all ranks
4. **Backward Compatibility**: Can disable CAPLA by setting very high fixed thresholds

## Performance Considerations

- **Memory**: Minimal overhead (per-class score tracking)
- **Computation**: Threshold calculation is O(n) for n pseudolabels
- **Convergence**: Adaptive thresholds may improve model stability and final accuracy

## Future Enhancements

1. Integrate reliability scores directly into loss function
2. Implement adaptive loss coefficients (λ) per class
3. Add confidence calibration for rare classes
4. Support for hard example mining based on uncertainty
