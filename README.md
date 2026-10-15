# FlowDA: Generative Data Assimilation for Regional XCO₂ Analysis

Implementation code for the paper: **"Generative Data Assimilation for Regional XCO₂ Analysis under Sparse Satellite Observations"**.

## Overview

FlowDA is a conditional flow-matching framework for generating spatially continuous regional XCO₂ analyses from sparse satellite observations.

The framework consists of two main stages:

1. **Model Training**: Learning the background-to-analysis transformation using CT2025 background fields, CAMS reference fields, simulated satellite observations, and auxiliary environmental variables.
2. **Satellite Data Assimilation**: Applying the trained model to real OCO-2, OCO-3, and GOSAT-2 observations without retraining, with independent validation against TCCON measurements.

FlowDA provides an efficient, data-driven approach to regional XCO₂ analysis without requiring online atmospheric transport integration.
