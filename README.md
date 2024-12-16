# Improving FlexTensor

## Installation

Requires: `Python 3.5+`, `Numpy`, `tvm: https://github.com/KnowingNothing/tvm/tree/mirror`

1. Install TVM, follow the [instructions](https://docs.tvm.ai/install/from_source.html).
2. Clone this repo:
   ```sh
   git clone https://github.com/SeonAh-Yoo/FlexTensor.git
   ```
3. Set the environments:
   `export AUTO_HOME=path/to/FlexTensor`
   `export PYTHONPATH=$AUTO_HOME:$PYTHONPATH`

To run the baselines, `PyTorch` is required.

## How to execute

### 1. Reproduce experiment results related to ANNS

0. move into `flextensor` folder

1. p-method without performance model
    ```sh
    ./run_gemm_p_wo_m.sh
    ```
2. q-method without performance model
    ```sh
    ./run_gemm_q_wo_m.sh
    ```
3. p-method with performance model
    ```sh
    ./run_gemm_p_w_m.sh
    ```
4. q-method with performance model
    ```sh
    ./run_gemm_q_w_m.sh
    ```
5. anns
    ```sh
    ./run_gemm_anns.sh
    ```

You can see results in `debug_N_M_K_1210_method.log`.

### 2. A3C Implementation
The implementation of A3C has not yet been completed.
Source codes related to A3C are in `a3c` folder.

