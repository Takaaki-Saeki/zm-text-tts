./run.sh --stage 1 --stop-stage 1
mkdir -p dump/raw
./run.sh --stage 4 --stop-stage 4
cat "dump/token_list/char/tokens.txt" > "../token_list_tphn.txt"

echo "Updated token_list.txt!"