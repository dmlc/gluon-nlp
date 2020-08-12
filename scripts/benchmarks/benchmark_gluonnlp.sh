for mode in train inference
do
  python3 benchmark_gluonnlp.py --layout NT --compute_layout NT --mode $mode
done

for mode in train inference
do
  python3 benchmark_gluonnlp.py --layout NT --compute_layout TN --mode $mode
done

for mode in train inference
do
  python3 benchmark_gluonnlp.py --layout TN --compute_layout TN --mode $mode
done
