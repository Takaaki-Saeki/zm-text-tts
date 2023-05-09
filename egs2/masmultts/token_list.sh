##########################
token_type=tphn
##########################

dirname="token_${token_type}"
cd ${dirname}

if [ ${token_type} = "byte" ]; then
    model_token_type=byte
elif [ ${token_type} = "tphn" ]; then
    model_token_type=char
elif [ ${token_type} = "phn" ]; then
    model_token_type=char
elif [ ${token_type} = "bphn" ]; then
    model_token_type=word
else
    echo "Error: token_type must be either byte, tphn, phn, or bphn"
    exit 1
fi

./run.sh --stage 1 --stop-stage 1
mkdir -p dump/raw
./run.sh --stage 4 --stop-stage 4
cat "dump/token_list/${model_token_type}/tokens.txt" > "../token_list_${token_type}.txt"

echo "Updated token_list.txt!"