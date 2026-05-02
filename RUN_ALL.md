# Overall Components
* Generator
![alt text](image-1.png)

## Generator
~~~
python gen_test.py \
  --api_key openai api keys \
  --ckpt-dir ./checkpoints/omat24_rattle2 \
  --prompt "Wide-gap semiconductor, wurtzite-like scaffold" \
  --save-dir ./artifacts/generated \
  --device cuda
~~~

## MD Optimize and DFT computation
~~~
python wrap_md_uma.py \
  --gen-path ./artifacts/generated/gen_union.npz \
  --ckpt /path/to/uma.pt \
  --run-dft --orca-command orca --nprocs 8 --maxcore 6000 \
  --preset standard \
  --outdir ./artifacts/mdopt
  <!-- --orcasimpleinput "M062X 6-31G* SP EnGrad D3BJ def2/J RIJCOSX TightSCF NoAutoStart MiniPrint NoPop" \ -->
~~~

## Analyze MD Optimization Trajectory
~~~
python structure_optim_modules/analyze_optimization.py --root ./artifacts/mdopt --pattern "loop_*.traj" --save-dir ./artifacts/mdopt/analysis \
  --export-json ./artifacts/mdopt/analysis/metrics.json --export-csv ./artifacts/mdopt/analysis/metrics.csv --no-show --compare
~~~


~~~
export FAIRCHEM_UMA_CKPT=/path/to/uma.pt
export FAIRCHEM_UMA_CONFIG=/path/to/config.yaml
export ORCA_COMMAND=/path/to/orca
python wrap_md_uma.py \
  --gen-path ./artifacts/generated/gen_union.npz \
  --run-dft --nprocs 8 --maxcore 4000 \
  --preset standard \
  --outdir ./artifacts/mdopt
~~~

