"""Launch the interactive demo or reproduce an execution recording."""
import argparse
import json
from modules.simulation import Comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=30)
    parser.add_argument('--alpha',type=float,default=.12)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--limit',type=int,default=400)
    parser.add_argument('--export',metavar='GIF')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--snapshot',metavar='PNG')
    parser.add_argument('--at',type=int,default=130)
    args = parser.parse_args()
    try:
        sim = Comparison(args.size,args.alpha,args.seed,args.limit)
    except ValueError as error:
        parser.error(str(error))
    if args.at < 0:
        parser.error('--at must be nonnegative')
    if args.export or args.snapshot or not args.headless:
        from modules.dashboard import Dashboard
        app = Dashboard(sim)
        if args.snapshot:
            from pathlib import Path
            import pygame
            while sim.tick < args.at and not sim.done:
                sim.step()
            app.cursor = sim.tick
            path = Path(args.snapshot)
            path.parent.mkdir(parents=True,exist_ok=True)
            pygame.image.save(app.render(),str(path))
        elif args.export:
            app.export(args.export)
        else:
            app.run()
    else:
        sim.run()
    if args.export or args.headless:
        print(json.dumps(sim.results(),indent=2))


if __name__ == '__main__':
    main()
