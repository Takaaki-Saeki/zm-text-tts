if [ $# -ne 1 ]; then
  echo "Usage: $0 <data-dir>"
  exit 1
fi

case_dir=$1
cd $case_dir
ln -s "../tts_pretrain_1/local" "local"
ln -s "../tts_pretrain_1/pyscripts" "pyscripts"
ln -s "../tts_pretrain_1/scripts" "scripts"
ln -s "../tts_pretrain_1/utils" "utils"
ln -s "../tts_pretrain_1/steps" "steps"
ln -s "../tts_pretrain_1/sid" "sid"
ln -s "../tts_pretrain_1/cmd.sh" "cmd.sh"
ln -s "../tts_pretrain_1/path.sh" "path.sh"
ln -s "../tts_pretrain_1/db.sh" "db.sh"
ln -s "../tts_pretrain_1/tts_pretrain.sh" "tts_pretrain.sh"

cp -r "../tts_pretrain_1/conf" .
cp -r "../tts_pretrain_1/run.sh" .

echo "Successfully finished!"