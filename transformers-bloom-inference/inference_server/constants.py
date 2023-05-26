from os.path import join, realpath, dirname
# inference method (args.deployment_framework)
HF_ACCELERATE = "hf_accelerate"
HF_CPU = "hf_cpu"
DS_INFERENCE = "ds_inference"
DS_ZERO = "ds_zero"
TOKENISE_DEFAULT = "bigscience/bloom-560m"
# MODEL_NAME = join(dirname(dirname(realpath(__file__))), "/mnt/sources/zalo_test/BLOOM-fine-tuning/output/checkpoint-20")
MODEL_NAME = "/mnt/sources/zalo_test/BLOOM-fine-tuning/output/checkpoint-20"

# GRPC_MAX_MSG_SIZE = 2**30  # 1GB


