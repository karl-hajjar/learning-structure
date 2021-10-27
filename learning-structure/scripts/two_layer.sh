source ../env/bin/activate  # activate the virtual environment
export PYTHONPATH=$PYTHONPATH:"$PWD"  # add wide-networks library to python path
python3 jobs/training_two_layer.py -n=300 -d=3 -m=500 -b=true -N=50 -pe=5