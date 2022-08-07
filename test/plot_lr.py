import paddle
import paddle.optimizer as optim
import paddle.optimizer.lr as lr
import pandas as pd
import seaborn as sns

import sys
sys.path.append("..")
from paddle_warmup_lr import WarmupLR

# StepDecay, MultiStepDecay, ExponentialDecay, ReduceOnPlateau

model_parma  = paddle.randn([1, 1], dtype="float32")
model_parma  = paddle.create_parameter(
    shape=model_parma.shape,
    dtype=str(model_parma.dtype).split(".")[1],
    default_initializer=paddle.nn.initializer.Assign(model_parma)
)
lr_base = 0.1
model_parmas = [model_parma]

lr_scheduler = lr.MultiStepDecay(lr_base, milestones=[30,60], gamma=0.1)
lr_scheduler = WarmupLR(lr_scheduler, init_lr=0.001, num_warmup=20, warmup_strategy='cos')


opt = optim.SGD(parameters=model_parmas, 
                learning_rate=lr_scheduler)

# this zero gradient update is needed to avoid a warning message
opt.clear_gradients()
opt.step()

# store lr
lrs = []
x   = []

# The wrapper doesn't affect old scheduler api
# Simply plug and play
for epoch in range(1, 90):
    lr_scheduler.step()
    lr = opt.get_lr()
    print(epoch, lr)
    opt.step()    # backward pass (update network)
    lrs.append(lr)
    x.append(epoch)

data = pd.DataFrame({'lr': lrs, 'epoch': x})
sns_plot = sns.lineplot(x='epoch', y='lr', data=data)
sns_plot.set(xlabel='epoch', ylabel='learning rate')
sns_plot.set_title('learning rate warmup')
sns_plot.get_figure().savefig('output.png')