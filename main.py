import graphcast
import yaml

with open('/home/wwh/graphcast/config/params.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)
trainer = graphcast.GraphCast(config)
trainer.train()
