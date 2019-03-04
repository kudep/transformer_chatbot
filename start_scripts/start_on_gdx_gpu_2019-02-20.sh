signature=$1
domain=$2
lambda_lm=$3
lambda_risk=$4
# bert_dir='supply/rubert_cased_L-12_H-768_A-12'
bert_dir='supply/rubert_cased_L-12_H-768_A-12_v2'
case "$signature" in

 "bare_test_v40k" )
 python train.py \
  --lr_warmup 16000 \
  --signature bare_test_v40k \
  --segment_embedding 1 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --bare_model 1 \
  --train_from "datasets/${domain}/*.train.txt" \
  --valid_from "datasets/${domain}/*.valid.txt" \
  --batch_split 256 \
  --lm_weight ${lambda_lm} \
  --risk_weight ${lambda_risk} \
 ;;

 "bert_tw1_seg1_v100k" )
 python train.py \
  --lr_warmup 16000 \
  --signature bert_tw1_seg1_v100k \
  --segment_embedding 1 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.txt \
  --tf_bert_model_load_from ${bert_dir}/model.ckpt-710142 \
  --train_from "datasets/${domain}/*.train.txt" \
  --valid_from "datasets/${domain}/*.valid.txt" \
  --batch_split 256 \
  --lm_weight ${lambda_lm} \
  --risk_weight ${lambda_risk} \
 ;;

 "bert_tw1_seg1_v100k_fin" )
 python train.py \
  --lr_warmup 16000 \
  --signature bert_tw1_seg1_v100k_fin \
  --segment_embedding 1 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.txt \
  --tf_bert_model_load_from ${bert_dir}/model.ckpt-710142 \
  --train_from "datasets/${domain}/*.train.txt" \
  --valid_from "datasets/${domain}/*.valid.txt" \
  --batch_split 256 \
  --lm_weight ${lambda_lm} \
  --risk_weight ${lambda_risk} \
 ;;

 "bert_tw0_seg1_v100k" )
 python train.py \
  --lr_warmup 16000 \
  --signature bert_tw0_seg1_v100k \
  --segment_embedding 1 \
  --tie_weights 0 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.txt \
  --tf_bert_model_load_from ${bert_dir}/model.ckpt-710142 \
  --train_from "datasets/${domain}/*.train.txt" \
  --valid_from "datasets/${domain}/*.valid.txt" \
  --batch_split 256 \
  --lm_weight ${lambda_lm} \
  --risk_weight ${lambda_risk} \
 ;;

 "bert_tw1_seg0_v100k" )
 python train.py \
  --lr_warmup 16000 \
  --signature bert_tw1_seg0_v100k \
  --segment_embedding 0 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.txt \
  --tf_bert_model_load_from ${bert_dir}/model.ckpt-710142 \
  --train_from "datasets/${domain}/*.train.txt" \
  --valid_from "datasets/${domain}/*.valid.txt" \
  --batch_split 256 \
  --lm_weight ${lambda_lm} \
  --risk_weight ${lambda_risk} \
 ;;

 "bert_tw1_seg1_v40k" )
 python train.py \
  --lr_warmup 16000 \
  --signature bert_tw1_seg1_v40k \
  --segment_embedding 1 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/model.ckpt-710142 \
  --train_from "datasets/${domain}/*.train.txt" \
  --valid_from "datasets/${domain}/*.valid.txt" \
  --batch_split 256 \
  --lm_weight ${lambda_lm} \
  --risk_weight ${lambda_risk} \
 ;;

 "bert_tw1_seg1_v40k_fin" )
 python train.py \
  --lr_warmup 16000 \
  --signature bert_tw1_seg1_v40k_fin \
  --segment_embedding 1 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/model.ckpt-710142 \
  --train_from "datasets/${domain}/*.train.txt" \
  --valid_from "datasets/${domain}/*.valid.txt" \
  --batch_split 256 \
  --lm_weight ${lambda_lm} \
  --risk_weight ${lambda_risk} \
 ;;

 "bert_tw0_seg1_v40k" )
 python train.py \
  --lr_warmup 16000 \
  --signature bert_tw0_seg1_v40k \
  --segment_embedding 1 \
  --tie_weights 0 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/model.ckpt-710142 \
  --train_from "datasets/${domain}/*.train.txt" \
  --valid_from "datasets/${domain}/*.valid.txt" \
  --batch_split 256 \
  --lm_weight ${lambda_lm} \
  --risk_weight ${lambda_risk} \
 ;;

 "bert_tw1_seg0_v40k" )
 python train.py \
  --lr_warmup 16000 \
  --signature bert_tw1_seg0_v40k \
  --segment_embedding 0 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --tf_bert_model_load_from ${bert_dir}/model.ckpt-710142 \
  --train_from "datasets/${domain}/*.train.txt" \
  --valid_from "datasets/${domain}/*.valid.txt" \
  --batch_split 256 \
  --lm_weight ${lambda_lm} \
  --risk_weight ${lambda_risk} \
 ;;

 "bare_tw1_seg1_v40k" )
 python train.py \
  --lr_warmup 16000 \
  --signature bare_tw1_seg1_v40k \
  --segment_embedding 1 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --bare_model 1 \
  --train_from "datasets/${domain}/*.train.txt" \
  --valid_from "datasets/${domain}/*.valid.txt" \
  --batch_split 128 \
  --lm_weight ${lambda_lm} \
  --risk_weight ${lambda_risk} \
 ;;

 "bare_tw0_seg1_v40k" )
 python train.py \
  --lr_warmup 16000 \
  --signature bare_tw0_seg1_v40k \
  --segment_embedding 1 \
  --tie_weights 0 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --bare_model 1 \
  --train_from "datasets/${domain}/*.train.txt" \
  --valid_from "datasets/${domain}/*.valid.txt" \
  --batch_split 256 \
  --lm_weight ${lambda_lm} \
  --risk_weight ${lambda_risk} \
 ;;

 "bare_tw1_seg0_v40k" )
 python train.py \
  --lr_warmup 16000 \
  --signature bare_tw1_seg0_v40k \
  --segment_embedding 0 \
  --tie_weights 1 \
  --path2bert_vocab ${bert_dir}/std_lm_vocab.40k.txt \
  --bare_model 1 \
  --train_from "datasets/${domain}/*.train.txt" \
  --valid_from "datasets/${domain}/*.valid.txt" \
  --batch_split 256 \
  --lm_weight ${lambda_lm} \
  --risk_weight ${lambda_risk} \
 ;;

esac