from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = "/app/Recognition/arcface_torch/checkpoints"
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 8
config.lr = 0.1
config.verbose = 30
config.dali = False

config.rec = "/app/Recognition/arcface_torch/pepo_ds/train"
config.num_classes = 19
config.num_image = 226
config.num_epoch = 10
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]

#Training time: 101.23045786703005
