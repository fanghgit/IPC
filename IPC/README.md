# Experimental code for Implicit Preference Completion

## Description:
- data folder: sub_ml-1m, a subset of ml-1m, positive samples: "train_positive.csv", negative samples: "negsamplesX.csv", test samples: "test.csv", validation samples: "validation.csv".
- util folder: preprocess code for the raw data.
- code folder: primalCRpp.jl, BPR.jl


## Quick start for experiments:
Inital setting: rank = 100, lambda = 0.1, stepsize = 0.01
1. To run BPR with full data:
	```
	julia > include("code/BPR.jl")
	julia > BPR_full("data/sub_ml-1m/train_positive.csv", "data/sub_ml-1m/test.csv", 100, 0.1, 0, 0.01)
	```
2. To run BPR with subsampled data:
	```
	julia > include("code/BPR.jl")
	julia > BPR_subsample("data/sub_ml-1m/train_positive.csv", "data/sub_ml-1m/negsamples1.csv", "data/sub_ml-1m/test.csv", 100, 0.1, 0, 0.01)
	```
3. To run primalCRpp with subsampled data:
	```
	julia > include("code/primalCRpp.jl")
	julia >primalCRpp_subsample("data/sub_ml-1m/train_positive.csv", "data/sub_ml-1m/negsamples1.csv", "data/sub_ml-1m/test.csv", 100, 0.1)
	```
