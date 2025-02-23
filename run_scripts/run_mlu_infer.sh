set -e

model_path=/workspace/model/favorite/large-scale-models/model-v1/Llama-2-7b-hf
echo "********************************"
echo "**命令行终端形式 仅支持单轮对话**"
echo "********************************"
# 单卡
# 配置板卡
export MLU_VISIBLE_DEVICES=0
python3 -m fastchat.serve.cli --model-path ${model_path}

## 双卡
## 配置板卡
# export MLU_VISIBLE_DEVICES=0,1
# python3 -m fastchat.serve.cli --model-path ${model_path} --num-gpus 2 --gpus "1,2"  
