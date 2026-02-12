Background: 
approxMLIR is a project that does approximate compilation in. The approx-runtime, detailed in the `/projects/beqg/approxMLIR/iree/third_party/approxMLIR/runtime`, is an updated version of previous un-centralized scripts. The un-centralized scripts also interfaces with previous version of approxMLIR compiler.

LLM tool benchmark can be refactored such that its invocation of LLM is substituted with the code similar to `selector_tune_llm.py` and its tools invocation is substituted with everything in 
```
haor2@gh-login02:/projects/beqg/approxMLIR/iree/third_party/approxMLIR/runtime/examples/benchmark> ls
benchmark_common.py  example_bm25_tuning.py  example_choose_tuning.py  example_kb_tuning.py  example_kmeans_tuning.py  example_lavamd_tuning.py  example_pagerank_tuning.py  src
```

Goal:
Have the llm-tool benchmark use the latest approx-runtime and the latest approxMLIR compiler for approximation.

Useful resources:
0. Path to llm-tool benchmark: `LLM_tool/jax_tool.py`, `LLM_tool/tuner.py` (both are very obsolete, tuner should be substituted by up-to-date approx-runtime APIs and benchmark can be re-orchestrated).
1. Path to llm-tool dataset and corpus: `LLM_tool/questions.py`, `LLM_tool/input.txt`, `LLM_tool/content.txt` (with the interface and data specified in the previous benchmarking code).
2. Path to dynamic llm inference: selector_tune_llm.py
3. Path to the approx-runtime:/projects/beqg/approxMLIR/iree/third_party/approxMLIR/runtime (hint: read `README.md` files there for more context)
4. Path to the ML compiler (built artifact): `export APPROX_OPT_ML=~/iree-build/tools/iree-opt`
5. Path to the non-ML compiler `export APPROX_OPT_CPP=/work/nvme/beqg/PolygeistSample/build/bin/polygeist-opt`
6. Path to compiler transformations: `/projects/beqg/approxMLIR/iree/third_party/approxMLIR/lib/approx/Passes/TransformApprox.cpp`
7. CPP source code of the llm tools: `/projects/beqg/approxMLIR/iree/third_party/approxMLIR/runtime/examples/benchmark/src`
8. you need to set up environment variables including `CGEIST_PATH=/work/nvme/beqg/PolygeistSample/build/bin/cgeist`, `CGEIST_RESOURCE_DIR=/work/nvme/beqg/PolygeistSample/llvm-project/build/lib/clang/18` and `CGEIST_INCLUDE_DIR=/work/nvme/beqg/PolygeistSample/tools/cgeist/Test/polybench/utilities` for the approxMLIR compiler to work properly.

Experiment setup:
1. ssh haor2@haor2@gh130
2. unset LD_LIBRARY_PATH
3. module load cuda/12.4
4. source myvenv/bin/activate

example command to run the benchmark:
```
ssh haor2@gh130 'unset LD_LIBRARY_PATH; module load cuda/12.4; source /u/haor2/workloads/myvenv/bin/activate; export PYTHONPATH=/u/haor2/workloads:/projects/beqg/approxMLIR/iree/third_party/approxMLIR/runtime:"'$PYTHONPATH; export APPROX_OPT_ML=/u/haor2/iree-build/tools/iree-opt; export APPROX_OPT_CPP=/work/nvme/beqg/PolygeistSample/build/bin/polygeist-opt; cd /u/haor2/workloads; python benchmark/llm_tool_benchmark.py --skip-tuning --use-subset
```