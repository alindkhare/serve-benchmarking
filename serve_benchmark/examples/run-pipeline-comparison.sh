set -ex
ray stop

python pipeline.py --num-replicas 1 --method chain 2>/dev/null
python pipeline.py --num-replicas 1 --method group 2>/dev/null
python pipeline.py --num-replicas 2 --method chain 2>/dev/null
python pipeline.py --num-replicas 2 --method group 2>/dev/null

