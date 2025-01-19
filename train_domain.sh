method=ours

datasets=('GitHub' 'LastFMAsia' 'DeezerEurope')
echo "${datasets[@]}"
n=${#datasets[@]} 
for i in $(seq 1 $n); do
  python train.py --backbone GCN \
       --gpu 0 \
       --ILmode domainIL \
       --inter-task-edges False \
       --minibatch False \
       --method $method \
       --ours_args "'latdim': 512; 'rank': 128; 'sup_coef': 10; 'proto_coef': 0.1; 'buffer_size':4096" \
       --repeats 5 \
       --multi-datasets "${datasets[@]}"
  datasets=(${datasets[-1]} ${datasets[@]:0:${#datasets[@]}-1})
done

datasets=('WikiCS' 'Facebook' 'Chameleon' 'Squirrel')
echo "${datasets[@]}"
n=${#datasets[@]} 
for i in $(seq 1 $n); do
  python train.py --backbone GCN \
       --gpu 0 \
       --ILmode domainIL \
       --inter-task-edges False \
       --minibatch False \
       --method $method \
       --ours_args "'latdim': 512; 'rank': 128; 'sup_coef': 10; 'proto_coef': 1; 'buffer_size':2048" \
       --repeats 5 \
       --multi-datasets "${datasets[@]}"
  datasets=(${datasets[-1]} ${datasets[@]:0:${#datasets[@]}-1})
done

datasets=('Citeseer' 'Pubmed' 'CoauthorCS' 'DBLP')
echo "${datasets[@]}"
n=${#datasets[@]} 
for i in $(seq 1 $n); do
  python train.py --backbone GCN \
       --gpu 0 \
       --ILmode domainIL \
       --inter-task-edges False \
       --minibatch False \
       --method $method \
       --ours_args "'latdim': 512; 'rank': 128; 'sup_coef': 1; 'proto_coef': 0.01; 'buffer_size':2048" \
       --repeats 5 \
       --multi-datasets "${datasets[@]}"
  datasets=(${datasets[-1]} ${datasets[@]:0:${#datasets[@]}-1})
done

datasets=('Pubmed' 'Photo' 'WikiCS' 'Airport')
echo "${datasets[@]}"
n=${#datasets[@]} 
for i in $(seq 1 $n); do
  python train.py --backbone GCN \
       --gpu 0 \
       --ILmode domainIL \
       --inter-task-edges False \
       --minibatch False \
       --method $method \
       --ours_args "'latdim': 512; 'rank': 128; 'sup_coef': 10; 'proto_coef': 1; 'buffer_size':2048" \
       --repeats 5 \
       --multi-datasets "${datasets[@]}"
  datasets=(${datasets[-1]} ${datasets[@]:0:${#datasets[@]}-1})
done

datasets=('CoauthorCS' 'Computer' 'Chameleon' 'DeezerEurope')
echo "${datasets[@]}"
n=${#datasets[@]} 
for i in $(seq 1 $n); do
  python train.py --backbone GCN \
       --gpu 0 \
       --ILmode domainIL \
       --inter-task-edges False \
       --minibatch False \
       --method $method \
       --ours_args "'latdim': 512; 'rank': 128; 'sup_coef': 10; 'proto_coef': 1; 'buffer_size':2048" \
       --repeats 5 \
       --multi-datasets "${datasets[@]}"
  datasets=(${datasets[-1]} ${datasets[@]:0:${#datasets[@]}-1})
done

datasets=('Cora' 'Facebook' 'LastFMAsia' 'Squirrel')
echo "${datasets[@]}"
n=${#datasets[@]} 
for i in $(seq 1 $n); do
  python train.py --backbone GCN \
       --gpu 0 \
       --ILmode domainIL \
       --inter-task-edges False \
       --minibatch False \
       --method $method \
       --ours_args "'latdim': 512; 'rank': 128; 'sup_coef': 10; 'proto_coef': 1; 'buffer_size':2048" \
       --repeats 5 \
       --multi-datasets "${datasets[@]}"
  datasets=(${datasets[-1]} ${datasets[@]:0:${#datasets[@]}-1})
done