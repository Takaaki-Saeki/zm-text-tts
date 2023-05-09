if [ $# -ne 1 ]; then
  echo "Usage: $0 <data-dir>"
  exit 1
fi

case_dir=$1
cd $case_dir
ln -s "../tts1/local" "local"
ln -s "../tts1/pyscripts" "pyscripts"
ln -s "../tts1/scripts" "scripts"
ln -s "../tts1/utils" "utils"
ln -s "../tts1/steps" "steps"
ln -s "../tts1/sid" "sid"
ln -s "../tts1/cmd.sh" "cmd.sh"
ln -s "../tts1/path.sh" "path.sh"
ln -s "../tts1/db.sh" "db.sh"
ln -s "../tts1/nisqa_results_mailabs.csv" "nisqa_results_mailabs.csv"
ln -s "../tts1/nisqa_results_fleurs.csv" "nisqa_results_fleurs.csv"
ln -s "../tts1/tts.sh" "tts.sh"

cp -r "../tts1/conf" .
cp -r "../tts1/run.sh" .
cp -r "../tts1/decode.sh" .
cp -r "../tts1/lang_set.txt" .
cp -r "../tts1/lang_set_decode.txt" .

echo "Successfully finished!"