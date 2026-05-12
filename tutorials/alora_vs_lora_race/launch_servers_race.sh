CUDA_VISIBLE_DEVICES=0 vllm serve ibm-granite/granite-switch-4.1-3b-preview --port 8111 &> vllm_alora.log &
CUDA_VISIBLE_DEVICES=1 vllm serve GrizleeBer/gs-test-2 --port 8112 &> vllm_lora.log &
