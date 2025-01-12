# bash compute_flow.sh $seqname
seqname=$1

if [[ $seqname ]];
then  
  array=(1 2 4 8)
  for i in "${array[@]}"
  do
  python compute_flow.py --datapath ../../../database/processed/JPEGImages/Full-Resolution/$seqname/ --loadmodel ./vcn_rob.pth  --dframe $i
done
fi
