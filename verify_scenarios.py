"""Fresh matched trials; deliberately independent of the legacy CSV results."""
import argparse
import json
from pathlib import Path
from modules.simulation import Simulation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',type=int,default=12)
    p.add_argument('--size',type=int,default=30)
    p.add_argument('--alpha',type=float,default=.12)
    p.add_argument('--limit',type=int,default=400)
    p.add_argument('--moving',action='store_true')
    p.add_argument('--output',default='docs/media/validation.json')
    args = p.parse_args()
    if args.seeds < 1: p.error('--seeds must be positive')
    rows = []
    for seed in range(args.seeds):
        sim = Simulation(args.size,args.alpha,seed,args.moving,args.limit).run()
        for a in sim.agents:
            rows.append(dict(seed=seed,bot=a.name,status=a.status,actions=a.finish,moves=a.moves,wall_senses=a.walls,rat_senses=a.senses))
    result = dict(configuration=vars(args),results=rows)
    path = Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Wrote {len(rows)} fresh runs to {path}')


if __name__ == '__main__': main()
