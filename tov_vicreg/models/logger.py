import wandb

class Logger(object):

    def __init__(self, group=None, type=None, name=None, args=None):
        self.step = 0
        wandb.init(project="Pretrain ViT",
                entity="SOME_ENTITY",
                config=args,
                group=group,
                job_type=type,
                name=name,
                )
        self.plots = {}

    def log_step(self, log_dict):
        wandb.log(log_dict, step=self.step)
        self.step += 1

    def log_line_plot(self, title, id, legends):
        datapoints = self.plots[id]
        wandb.log({id: wandb.plot.line_series(xs=list(range(len(datapoints[0]))),
            ys=datapoints,
            keys=legends,
            title=title)})

    def log_final(self, log_dict):
        wandb.log(log_dict)

    def log_matplot(self, id, fig):
        wandb.log({ id: wandb.Image(fig) })

    def close(self):
        wandb.finish()