for mode in train inference
do
  python3 benchmark_gluonnlp.py --layout NT --compute_layout NT --mode $mode --use_fp16
done

for mode in train inference
do
  python3 benchmark_gluonnlp.py --layout NT --compute_layout TN --mode $mode --use_fp16
done

for mode in train inference
do
  python3 benchmark_gluonnlp.py --layout TN --compute_layout TN --mode $mode --use_fp16
done
